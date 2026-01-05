import habitat_sim
import os
from magnum import Vector3
import math
import numpy as np
from data.utils.image_utils import resize_with_padding
import torch

def env_config(config, scene_id):
    env_cfg = habitat_sim.SimulatorConfiguration()
    env_cfg.scene_id = os.path.join(config.simulator.path.scene_root, scene_id, scene_id + ".glb")
    env_cfg.enable_physics = True
    env_cfg.allow_sliding = True
    return env_cfg


def agent_config(config):
    # 1. 配置 agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = config.simulator.agent.height
    agent_cfg.radius = config.simulator.agent.radius  
    agent_cfg.body_type = config.simulator.agent.body_type
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
    rgb_sensor_spec.position = config.simulator.rgb_sensor.position
    rgb_sensor_spec.orientation = Vector3(config.simulator.rgb_sensor.orientation[0], config.simulator.rgb_sensor.orientation[1], config.simulator.rgb_sensor.orientation[2])
    rgb_sensor_spec.resolution = [config.simulator.rgb_sensor.height, config.simulator.rgb_sensor.width]
    rgb_sensor_spec.hfov = config.simulator.rgb_sensor.hfov
    # 3. 将传感器加入到agent身上
    agent_cfg.sensor_specifications = [rgb_sensor_spec]
    return agent_cfg


def environment(config, scene_id):
    env_cfg = env_config(config, scene_id)
    agent_cfg = agent_config(config)
    sim_cfg = habitat_sim.Configuration(env_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(sim_cfg)
    return sim, env_cfg, agent_cfg


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



def get_observation(simulator, config):
    obs = simulator.get_sensor_observations()
    rgb = obs["rgb_sensor"][:, :, :3]
    frames_np = np.expand_dims(rgb, axis=0) 
    resized = [resize_with_padding(frames_np[i], (config.main.predicted_frame_height, config.main.predicted_frame_width)) for i in range(frames_np.shape[0])]
    frames_np = np.stack(resized, axis=0)
    current_frame = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0  # (1, C, H, W)
    current_frame = current_frame.to("cuda", dtype=torch.bfloat16)
    return current_frame


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