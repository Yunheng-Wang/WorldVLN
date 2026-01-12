import numpy as np
import habitat_sim
import math
from scipy.ndimage import gaussian_filter1d

def smooth_path(path, window_size, sigma):
    smoothed_path = path.copy()
    for i in range(3):
        smoothed_path[:, i] = gaussian_filter1d(path[:, i], sigma=sigma)
    return smoothed_path


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