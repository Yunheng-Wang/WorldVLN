import sys, os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_path)

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
from data.preprocess.utils.sim import environment
from data.preprocess.utils.action import smooth_path, interpolate_path, get_yaw_and_dist, count_delta_yaw, action_point, find_more_paths_for_task


def load_task_val(config):
    grouped_tasks= {}
    with open(os.path.join(config["data_root"], "tasks", "r2r_ce", "val_unseen/val_unseen.json"), 'r', encoding='utf-8') as file:
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
    val_tasks = load_task_val(cfg)
    # 3. 加载t5模型
    t5_model = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device = torch.device("cuda"), checkpoint_path=os.path.join(cfg["wan_root"], "models_t5_umt5-xxl-enc-bf16.pth"), tokenizer_path=os.path.join(cfg["wan_root"], 'google/umt5-xxl'))
    # 4. 提取验证集数据
    for i, scene_id in enumerate(tqdm(val_tasks, desc="Scenes", unit="scene")):
        simulator, _, agent_cfg = environment(cfg, scene_id)
        for id, task in enumerate(tqdm(val_tasks[scene_id], desc="Tasks", unit="task", leave=False)):
            save_path = os.path.join(cfg["data_root"], "cache", "val_unseen", "mp3d" + "_r2r_ce_" + scene_id + "_" + str(task["episode_id"]))
            if os.path.exists(os.path.join(save_path, 'instruction.pth')):
                    continue
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "instruction.txt"), 'w', encoding='utf-8') as file:
                instruction = task["instruction"]["instruction_text"].replace('\n', ' ').replace('\r', ' ')
                file.write(instruction)
            with torch.no_grad():
                instruction_embed = t5_model(Video_Model_Prompt["user"].format(instruction), torch.device("cuda"))
                torch.save(instruction_embed[0], os.path.join(save_path, "instruction.pth"))
        simulator.close()

if __name__ == "__main__":
    main()