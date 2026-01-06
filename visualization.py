import habitat_sim
import torch
import random
import yaml
import logging
import json
import numpy as np
import os
import multiprocessing as mp
from accelerate import Accelerator
from omegaconf import OmegaConf
import argparse
import imageio.v2 as imageio

from model.WorldVLNConfig import WorldVLNConfig
from model.WorldVLN import WorldVLN
from utils.load import load_r2r_ce_task
from utils.habitat_sim import environment, action_point, get_yaw_and_dist, get_observation_visual, find_more_paths_for_task, get_observation


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format=f'%(asctime)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')



def eval_val_unseen(checkpoint_path):
    # 1. 加载配置
    config = OmegaConf.load('visualization.yaml')
    # 2. 加载模型
    model_config = WorldVLNConfig(
        # common setting
        batch_size = config.main.batch_size,
        dtype = torch.bfloat16 if config.main.dtype == "bf16" else torch.float32,
        predict_frame_num = config.main.prediction_steps,
        history_frame_num = config.main.history_steps,
        predict_frame_h = config.main.predicted_frame_height,
        predict_frame_w = config.main.predicted_frame_width,
        # video setting
        video_wan_root = config.model.video_model.video_wan_root_path,
        video_wan_precision = "bfloat16" if config.main.dtype == "bf16" else "float32",
        video_block_num = config.model.video_model.video_block_num,
        video_model_dim = config.model.video_model.video_model_dim,
        # action setting
        action_dim = config.model.action_model.action_dim,
        action_model_dim = config.model.action_model.action_model_dim,
        action_model_ffn_dim = config.model.action_model.action_model_ffn_dim,
        action_block_num = config.model.action_model.action_block_num,
        action_encoder_depth = config.model.action_model.action_encoder_mlp_depth,
        action_decoder_depth = config.model.action_model.action_decoder_mlp_depth,
        action_registers_num = config.model.action_model.action_registers_num,
        action_eps = config.model.action_model.action_eps,
        # understand setting
        understand_model_dim = config.model.understand_model.understand_model_dim,
        understand_model_ffn_dim = config.model.understand_model.understand_model_ffn_dim,
        understand_block_num = config.model.understand_model.understand_block_num,
        understand_vlm_root = config.model.understand_model.understand_vlm_root_path,
        understand_vlm_token_dim = config.model.understand_model.understand_vlm_token_dim,
        understand_vlm_projector_mlp_depth = config.model.understand_model.understand_vlm_projector_mlp_depth,
        understand_eps = config.model.understand_model.understand_eps,
        # loss
        video_loss_weight = config.model.loss.video_loss_weight,
        action_loss_weight = config.model.loss.action_loss_weight,
        understand_loss_weight = config.model.loss.understand_loss_weight
    )
    model = WorldVLN(model_config)
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    accelerator.load_state(checkpoint_path)
    model.eval()
    # 3. 加载任务
    tasks_p = load_r2r_ce_task(config, "val_unseen")
    tasks = {}
    for scene_id in tasks_p:
        random_task= random.choice(tasks_p[scene_id])
        tasks[scene_id] = [random_task]
    # 4. 执行
    results = {}
    for scene_id in tasks:
        simulator, _, agent_cfg = environment(config, scene_id)
        save_path = os.path.join(config.simulator.path.output, "visualization", str(scene_id), str(tasks[scene_id][0]["episode_id"]))
        os.makedirs(save_path, exist_ok=True)
        with imageio.get_writer(os.path.join(save_path, "video_action.mp4"), fps=5, codec='libx264') as writer_1, imageio.get_writer(os.path.join(save_path, "video_predict.mp4"), fps=5, codec='libx264') as writer_2:
            for id, task in enumerate(tasks[scene_id]):
                # 4.1. 初始化agent
                agent = simulator.initialize_agent(0)
                initial_state = habitat_sim.AgentState()
                initial_state.position = task["start_position"]
                initial_state.rotation = task["start_rotation"]
                agent.set_state(initial_state)
                # 4.2. 先旋转到正确方向（临时）
                tmp_frame = get_observation_visual(simulator, config)
                writer_1.append_data(tmp_frame)
                task["reference_path"] = find_more_paths_for_task(simulator, task["reference_path"])
                yaw_err, _, dist_planar = get_yaw_and_dist(simulator, task["reference_path"][1])
                action_point(simulator, agent_cfg, 0, yaw_err)
                tmp_frame = get_observation_visual(simulator, config)
                writer_1.append_data(tmp_frame)
                # 4.3. 模型执行
                history = []
                instruction = task["instruction"]["instruction_text"]
                L = 0
                step = 0
                his_pos = []
                while step < config.main.max_execution_steps:
                    # 4.3.1. 获取当前观测
                    current_frame = get_observation(simulator, config)
                    # 4.3.2. 加载历史观测
                    if len(history) == 0:
                        his_frames = torch.tensor(history)
                        history.append(current_frame)
                    else:
                        indices = sorted(random.sample(range(len(history)), 8))
                        sampled_history = [history[i] for i in indices]
                        his_frames = torch.cat(sampled_history, dim=0).unsqueeze(0)
                        history.append(current_frame)
                    # 4.3.3 模型推理
                    with torch.no_grad():
                        predicted_frames, predicted_actions, predicted_stop_flag = model.inference_step([instruction],  current_frame,  his_frames, 10)
                        predicted_actions = predicted_actions.squeeze(0).tolist()[:config.main.execution_steps]
                        predicted_frames = predicted_frames.squeeze(0)[:config.main.execution_steps]
                        for frame in predicted_frames:
                            writer_2.append_data(frame.permute(1, 2, 0).cpu().numpy())
                    # 4.3.4 物理执行
                    for idx, action in enumerate(predicted_actions):
                        prev_pos = np.array(simulator.get_agent(0).get_state().position, dtype=float)
                        action_point(simulator, agent_cfg, action[0], action[1])
                        new_pos = np.array(simulator.get_agent(0).get_state().position, dtype=float)
                        his_pos.append(new_pos)
                        L += float(np.linalg.norm(new_pos - prev_pos))
                        tmp_frame = get_observation_visual(simulator, config)
                        writer_1.append_data(tmp_frame)
                        if idx < len(predicted_actions) - 1:
                            history.append(get_observation(simulator, config))
                    # 4.3.5 到达终点
                    step += 1
                    if predicted_stop_flag == 1:
                        break
            # 4.4. 评估 SR
            SR = 0
            final_pos = np.array(agent.get_state().position, dtype=float)
            gt_pos = np.array(task["goals"][0]["position"], dtype=float)
            d = np.linalg.norm(final_pos - gt_pos) 
            if d <= 3:
                SR = 1
            # 4.5 评估 SPL
            SPL = SR * (task['info']['geodesic_distance'] / max(L, task['info']['geodesic_distance']))
            # 4.6 评估 OSR
            OSR = 0
            for i in his_pos:
                if np.linalg.norm(i - gt_pos) <= 3:
                    OSR = 1
                    break
            results[str(task["episode_id"])] = {"Step": step, "TL": L, "NE": d, "OSR": OSR, "SR": SR, "SPL": SPL}
            # 4.6 保存刷新结果
            with open(save_path + "results.json", "w") as f:
                json.dump(results, f, indent=4)

        simulator.close()

if __name__ == "__main__":
    checkpoint_path = "/home/CONNECT/yfang870/yunhengwang/WorldVLN/log/2026-01-04_23/checkpoint_step_10_6761"
    eval_val_unseen(checkpoint_path)

