"""
    注意:
        1. agent 的局部坐标系为，正前方为x, 右手边为y, 正上方为z
"""



import habitat_sim
import yaml
import numpy as np
import math
import json
import os
import imageio.v2 as imageio
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from magnum import Vector3
from scipy.spatial.transform import Rotation as R


def env_config(config, scene_id):
    env_cfg = habitat_sim.SimulatorConfiguration()
    env_cfg.scene_id = os.path.join(config["scene_root_path"], scene_id, scene_id + ".glb")
    env_cfg.enable_physics = False
    env_cfg.allow_sliding = True
    return env_cfg


def agent_config(config):
    # 1. 配置 agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = config["agent"]["height"] 
    agent_cfg.radius = config["agent"]["radius"]    
    agent_cfg.body_type = config["agent"]["body_type"]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=0)
        ),
    }
    # 2. 配置agent 上的传感器
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "rgb_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.position = config["agent"]["rgb_sensor"]["position"]
    rgb_sensor_spec.orientation = Vector3(config["agent"]["rgb_sensor"]["orientation"][0], config["agent"]["rgb_sensor"]["orientation"][1], config["agent"]["rgb_sensor"]["orientation"][2])
    rgb_sensor_spec.resolution = [config["agent"]["rgb_sensor"]["height"], config["agent"]["rgb_sensor"]["width"]]
    rgb_sensor_spec.hfov = config["agent"]["rgb_sensor"]["hfov"]
    # 3. 将传感器加入到agent身上
    agent_cfg.sensor_specifications = [rgb_sensor_spec]
    return agent_cfg


def environment(config, scene_id):
    env_cfg = env_config(config, scene_id)
    agent_cfg = agent_config(config)
    sim_cfg = habitat_sim.Configuration(env_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(sim_cfg)
    return sim, env_cfg, agent_cfg


def load_task(config, data_type):
    with open(os.path.join(config["task_root_path"], "r2r_ce", data_type, data_type + ".json"), 'r', encoding='utf-8') as file:
        task = json.load(file)
    grouped_tasks = {}
    for episode in task['episodes']:
        scene_id = episode['scene_id']
        parts = scene_id.split('/')
        scene_hash = parts[1] if len(parts) > 1 else parts[0].split('.')[0]  # 备用方案
        if scene_hash not in grouped_tasks:
            grouped_tasks[scene_hash] = []
        grouped_tasks[scene_hash].append(episode)
    return grouped_tasks


def count_delta_yaw(prev_rotation, curr_rotation):
    q1 = np.array([prev_rotation.x, prev_rotation.y, prev_rotation.z, prev_rotation.w])
    q2 = np.array([curr_rotation.x, curr_rotation.y, curr_rotation.z, curr_rotation.w])
    q1_conj = np.array([-q1[0], -q1[1], -q1[2], q1[3]])
    q_diff = np.array([
        q2[3]*q1_conj[0] + q2[0]*q1_conj[3] + q2[1]*q1_conj[2] - q2[2]*q1_conj[1],
        q2[3]*q1_conj[1] - q2[0]*q1_conj[2] + q2[1]*q1_conj[3] + q2[2]*q1_conj[0],
        q2[3]*q1_conj[2] + q2[0]*q1_conj[1] - q2[1]*q1_conj[0] + q2[2]*q1_conj[3],
        q2[3]*q1_conj[3] - q2[0]*q1_conj[0] - q2[1]*q1_conj[1] - q2[2]*q1_conj[2]
    ])
    yaw = 2 * math.atan2(q_diff[1], q_diff[3])
    return yaw


def action_point(simulator, agent_cfg, distance, angle):
    """angle : 弧度 (左转正数，右转负数)"""
    # 1. 配置移动步距
    agent_cfg.action_space["move_forward"].actuation.amount = distance
    # 2. 获取移动前状态
    agent = simulator.get_agent(0)
    pre_state = agent.get_state()
    # 3. 实际旋转
    if angle >= 0: # turn left
        agent_cfg.action_space["turn_left"].actuation.amount = math.degrees(angle)
        simulator.step("turn_left")
    else:           # turn right
        agent_cfg.action_space["turn_right"].actuation.amount = -math.degrees(angle)
        simulator.step("turn_right")
    # 4. 计算实际旋转弧度
    turn_state = agent.get_state()
    yaw_delta = count_delta_yaw(pre_state.rotation, turn_state.rotation)
    # 5. 实际移动
    simulator.step("move_forward")
    # 6. 计算实际移动距离
    move_state = agent.get_state()
    dis_delta = np.linalg.norm(np.array(pre_state.position) - np.array(move_state.position))
    return dis_delta, yaw_delta


def get_yaw_and_dist(sim, target_pos):
    def _quat_conj(q):
        return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)
    def _quat_mul(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ], dtype=np.float32)
    def _rotate_vec_by_quat(v, q_xyzw):
        # 把向量 v 视作纯四元数 [vx, vy, vz, 0]，用 q * v * q_conj 旋转
        vq = np.array([v[0], v[1], v[2], 0.0], dtype=np.float32)
        return _quat_mul(_quat_mul(q_xyzw, vq), _quat_conj(q_xyzw))[:3]
    
    agent = sim.get_agent(0)
    s = agent.get_state()
    cur_pos = np.asarray(s.position, dtype=np.float32)
    q = np.array([s.rotation.x, s.rotation.y, s.rotation.z, s.rotation.w], dtype=np.float32)
    target_pos = np.asarray(target_pos, dtype=np.float32)
    dir_vec = target_pos - cur_pos            
    dist_3d = float(np.linalg.norm(dir_vec))
    dir_flat = dir_vec.copy()
    dir_flat[1] = 0.0
    dist_planar = float(np.linalg.norm(dir_flat))
    local_forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    world_forward = _rotate_vec_by_quat(local_forward, q)
    a = np.array([world_forward[0], 0.0, world_forward[2]], dtype=np.float32)
    b = np.array([dir_vec[0],      0.0, dir_vec[2]],      dtype=np.float32)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        yaw_err = 0.0
    else:
        a /= na; b /= nb
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        ang = math.acos(dot)
        cross_y = np.cross(a, b)[1]
        yaw_err = ang if cross_y >= 0.0 else -ang

    return yaw_err, dist_planar, dist_3d


def interpolate_path(path, step):
    path = np.array(path)  
    interpolated_points = [path[0]] 
    for i in range(1, len(path)):
        p1, p2 = path[i-1], path[i]
        dist = np.linalg.norm(p2 - p1)  
        num_new_points = int(dist // step)  
        for j in range(1, num_new_points + 1):
            t = j / (num_new_points + 1)
            new_point = p1 + t * (p2 - p1)
            interpolated_points.append(new_point)
        interpolated_points.append(p2)  
    return np.array(interpolated_points)


def smooth_path(path, window_size, sigma):
    smoothed_path = path.copy()
    # 用高斯滤波平滑路径
    for i in range(3):
        smoothed_path[:, i] = gaussian_filter1d(path[:, i], sigma=sigma)
    return smoothed_path


def draw_change(original_path, interpolation_path, smoothed_path, save_path):
    original_path = np.array(original_path)
    interpolation_path = np.array(interpolation_path)
    smoothed_path = np.array(smoothed_path)
    original_x = original_path[:, 0]
    original_y = original_path[:, 2]  
    smoothed_x = smoothed_path[:, 0]
    smoothed_y = smoothed_path[:, 2]
    interpolation_x = interpolation_path[:, 0]
    interpolation_y = interpolation_path[:, 2]  

    plt.figure(figsize=(8, 6))
    plt.plot(original_x, original_y, label="Original Path", color='blue', marker='o', markersize=5)
    plt.plot(interpolation_x, interpolation_y, label="Interpolated Path", color='green', linestyle='--', marker='s', markersize=5)
    plt.plot(smoothed_x, smoothed_y, label="Smoothed Path", color='red', linestyle='--', marker='x', markersize=5)
    plt.title("Path Before and After Smoothing", fontsize=14)
    plt.xlabel("X Position", fontsize=12)
    plt.ylabel("Y Position", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "change.png"), format='png')  
    plt.close() 


def find_more_paths_for_task(sim, reference_path):
    all_paths = []
    for i in range(len(reference_path) - 1):
        start_position = reference_path[i]
        end_position = reference_path[i + 1]
        shortest_path = habitat_sim.ShortestPath()
        shortest_path.requested_start = np.array(start_position)
        shortest_path.requested_end = np.array(end_position)
        success = sim.pathfinder.find_path(shortest_path)
        path = shortest_path.points 
        if success and shortest_path.points:
            if all_paths:
                # 如果已经有路径，去除当前路径的第一个点（避免重复）
                points = [point.tolist() for point in shortest_path.points[1:]] 
                all_paths.extend(points)
            else:
                # 如果是第一个路径，直接添加
                all_paths.extend([point.tolist() for point in shortest_path.points]) 
    return all_paths


def main():
    # 1. 加载基础配置
    with open('data/preprocess/config.yaml', 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)
    # 2. 加载任务
    tasks = load_task(cfg, cfg["type"])
    # 3. 提取数据
    for scene_id in tasks:
        ## 3.1 创建虚拟环境
        simulator, _, agent_cfg = environment(cfg, scene_id)
        for id, task in enumerate(tasks[scene_id]):
            action = []
            ## 3.2 配置保存目录
            save_path = os.path.join(cfg["output_path"], cfg["type"], "mp3d_" + "r2r_ce_" + scene_id + "_" + str(task["episode_id"]))
            if os.path.exists(save_path):
                continue
            os.makedirs(save_path, exist_ok=True)
            ## 3.3 保存 指令
            with open(os.path.join(save_path, "instruction.txt"), 'w', encoding='utf-8') as file:
                cleaned_text = task["instruction"]["instruction_text"].replace('\n', ' ').replace('\r', ' ')
                file.write(cleaned_text)
            ## 3.4 保存 观测
            with imageio.get_writer(os.path.join(save_path, "observation.mp4"), fps=5, codec='libx264') as writer:
                ### 3.4.0 
                task["reference_path"] = find_more_paths_for_task(simulator, task["reference_path"])
                ### 3.4.1 将参考路径加密 + 平滑处理
                interpolation_path = interpolate_path(task["reference_path"], cfg["trajectory"]["interval"])
                reference_path = smooth_path(interpolation_path, cfg["trajectory"]["smooth"][0], cfg["trajectory"]["smooth"][1])
                # reference_path = task["reference_path"]
                # draw_change(task["reference_path"], interpolation_path, reference_path, save_path)
                ### 3.4.2 初始化机器人
                agent = simulator.initialize_agent(0)
                initial_state = habitat_sim.AgentState()
                initial_state.position = reference_path[0]
                initial_state.rotation = task["start_rotation"]
                agent.set_state(initial_state)
                ### 3.4.3 移动 + 记录观测
                for u, ref_point in enumerate(reference_path[1:]):
                    yaw_err, _, dist_planar = get_yaw_and_dist(simulator, ref_point)
                    if u == 0:
                        action_point(simulator, agent_cfg, 0, yaw_err)
                        obs = simulator.get_sensor_observations()
                        rgb = obs["rgb_sensor"][:, :, :3]
                        writer.append_data(rgb)
                        delta_dis, delta_yaw = action_point(simulator, agent_cfg, dist_planar, 0)
                        obs = simulator.get_sensor_observations()
                        rgb = obs["rgb_sensor"][:, :, :3]
                        writer.append_data(rgb)
                        action.append([delta_dis, delta_yaw])
                    else:
                        #### 移动
                        delta_dis, delta_yaw = action_point(simulator, agent_cfg, dist_planar, yaw_err)
                        obs = simulator.get_sensor_observations()
                        rgb = obs["rgb_sensor"][:, :, :3]
                        writer.append_data(rgb)
                        action.append([delta_dis, delta_yaw])
            # 3.5 保存 动作
            np.save(os.path.join(save_path, "action.npy"), np.array(action))

        simulator.close()

if __name__ == "__main__":
    main()