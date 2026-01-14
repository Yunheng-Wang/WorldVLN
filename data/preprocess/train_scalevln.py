import sys, os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_path)

import shutil  
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
import torch
from tqdm import tqdm

from model.utils.wan.modules.t5 import T5EncoderModel
from utils.promot import Video_Model_Prompt
from data.preprocess.utils.sim import environment_hm3d
from data.preprocess.utils.action import smooth_path, interpolate_path, get_yaw_and_dist, count_delta_yaw, action_point, find_more_paths_for_task


def load_task(config):
    grouped_tasks = {}
    with open(os.path.join(config["data_root"], "tasks", "scalevln", "scalevln.json"), 'r', encoding='utf-8') as file:
        task = json.load(file)
    for episode in task['episodes']:
        scene_id = episode['scene_id']
        parts = scene_id.split('/')
        scene_hash = parts[1] if len(parts) > 1 else parts[0].split('.')[0]  # 备用方案
        if scene_hash not in grouped_tasks:
            grouped_tasks[scene_hash] = []
        grouped_tasks[scene_hash].append(episode)
    return grouped_tasks


def main():
    # 1. 加载基础配置
    with open('data/preprocess/config.yaml', 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)
    # 2. 加载训练集任务
    tasks = load_task(cfg)
    # 3. 加载t5模型
    t5_model = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device = torch.device("cuda"), checkpoint_path=os.path.join(cfg["wan_root"], "models_t5_umt5-xxl-enc-bf16.pth"), tokenizer_path=os.path.join(cfg["wan_root"], 'google/umt5-xxl'))
    # 4. 提取训练集数据
    for i, scene_id in enumerate(tqdm(tasks, desc="Scenes", unit="scene")):
        ## 4.1 创建虚拟环境
        simulator, _, agent_cfg = environment_hm3d(cfg, scene_id)
        for id, task in enumerate(tqdm(tasks[scene_id], desc="Tasks", unit="task", leave=False)):
            action = []
            ## 4.2 配置保存目录
            save_path = os.path.join(cfg["data_root"], "cache", "train", "hm3d_" + "scalevln" + "_" + scene_id.split('-')[1] + "_" + str(task["episode_id"]))
            if os.path.exists(os.path.join(save_path, 'action.npy')) and os.path.exists(os.path.join(save_path, 'instruction.txt')) and os.path.exists(os.path.join(save_path, 'observation.mp4')) and os.path.exists(os.path.join(save_path, 'instruction.pth')):
                continue
            os.makedirs(save_path, exist_ok=True)
            ## 4.3 保存 指令 及其 embed
            with open(os.path.join(save_path, "instruction.txt"), 'w', encoding='utf-8') as file:
                instruction = task["instruction"]["instruction_text"].replace('\n', ' ').replace('\r', ' ')
                file.write(instruction)
            with torch.no_grad():
                instruction_embed = t5_model(Video_Model_Prompt["user"].format(instruction), torch.device("cuda"))
                torch.save(instruction_embed[0], os.path.join(save_path, "instruction.pth"))
            ## 4.4 保存 观测
            with imageio.get_writer(os.path.join(save_path, "observation.mp4"), fps=5, codec='libx264') as writer:
                ### 4.4.0 
                task["reference_path"] = find_more_paths_for_task(simulator, task["reference_path"])

                ### 4.4.1 将参考路径加密 + 平滑处理
                if len(task["reference_path"]) == 0:
                    if os.path.exists(save_path):
                        shutil.rmtree(save_path)
                    continue
                interpolation_path = interpolate_path(task["reference_path"], cfg["trajectory"]["interval"])
                reference_path = smooth_path(interpolation_path, cfg["trajectory"]["smooth"][0], cfg["trajectory"]["smooth"][1])
                # draw_change(task["reference_path"], interpolation_path, reference_path, save_path)
                ### 4.4.2 初始化机器人
                agent = simulator.initialize_agent(0)
                initial_state = habitat_sim.AgentState()
                initial_state.position = reference_path[0]
                initial_state.rotation = task["start_rotation"]
                agent.set_state(initial_state)
                ### 4.4.3 移动 + 记录观测
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
            # 4.5 保存 动作
            np.save(os.path.join(save_path, "action.npy"), np.array(action))
        simulator.close()

if __name__ == "__main__":
    main()