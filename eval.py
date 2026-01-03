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

from model.WorldVLNConfig import WorldVLNConfig
from model.WorldVLN import WorldVLN
from utils.load import load_r2r_ce_task
from utils.habitat_sim import environment, action_point, get_yaw_and_dist, get_observation


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format=f'%(asctime)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Run eval_val_unseen")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument('--gpu', type=int, required=True, help="GPU device id")
    return parser.parse_args()


def eval_val_unseen(checkpoint_path, gpu_id):
    # 设置 CUDA 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Using GPU {gpu_id} for evaluation")
    # 1. 加载配置
    config = OmegaConf.load('eval.yaml')
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
    tasks = load_r2r_ce_task(config, "val_unseen")
    if gpu_id == 0:
        key = list(tasks.keys())[0]  
        value = tasks[key]     
        tasks = {key: value}
    elif gpu_id == 1:
        key = list(tasks.keys())[1]  
        value = tasks[key]     
        tasks = {key: value}
    elif gpu_id == 2:
        key_1 = list(tasks.keys())[2]  
        value_1 = tasks[key_1]  
        key_2 = list(tasks.keys())[9]  
        value_2 = tasks[key_2]
        key_3 = list(tasks.keys())[10]  
        value_3 = tasks[key_3]   
        tasks = {key_1: value_1, key_2: value_2, key_3: value_3}
    elif gpu_id == 3:
        key = list(tasks.keys())[3]  
        value = tasks[key]     
        tasks = {key: value}
    elif gpu_id == 4:
        key = list(tasks.keys())[4]  
        value = tasks[key]     
        tasks = {key: value}
    elif gpu_id == 5:
        key = list(tasks.keys())[5]  
        value = tasks[key]     
        tasks = {key: value}
    elif gpu_id == 6:
        key_1 = list(tasks.keys())[6]  
        value_1 = tasks[key_1]  
        key_2 = list(tasks.keys())[7]  
        value_2 = tasks[key_2]
        tasks = {key_1: value_1, key_2: value_2}
    elif gpu_id == 7:
        key = list(tasks.keys())[8]  
        value = tasks[key]     
        tasks = {key: value}
    # 4. 执行
    results = {}
    for scene_id in tasks:
        simulator, _, agent_cfg = environment(config, scene_id)
        for id, task in enumerate(tasks[scene_id]):
            # 4.1. 初始化agent
            agent = simulator.initialize_agent(0)
            initial_state = habitat_sim.AgentState()
            initial_state.position = task["start_position"]
            initial_state.rotation = task["start_rotation"]
            agent.set_state(initial_state)
            # 4.2. 先旋转到正确方向（临时）
            yaw_err, _, dist_planar = get_yaw_and_dist(simulator, task["reference_path"][1])
            action_point(simulator, agent_cfg, 0, yaw_err)
            # 4.3. 模型执行
            history = []
            instruction = task["instruction"]["instruction_text"]
            L = 0
            step = 0
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
                    predicted_actions = predicted_actions.squeeze(0).tolist()
                # 4.3.4 物理执行
                for idx, action in enumerate(predicted_actions):
                    prev_pos = np.array(simulator.get_agent(0).get_state().position, dtype=float)
                    action_point(simulator, agent_cfg, action[0], action[1])
                    new_pos = np.array(simulator.get_agent(0).get_state().position, dtype=float)
                    L += float(np.linalg.norm(new_pos - prev_pos))
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
            results[str(task["episode_id"])] = {"Step": step, "TL": L, "NE": d, "SR": SR, "SPL": SPL}
            # 4.6 保存刷新结果
            save_path = os.path.join(config.simulator.path.output, os.path.basename(checkpoint_path) + "_" + str(gpu_id) + ".json")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(results, f, indent=4)

        simulator.close()

if __name__ == "__main__":
    args = parse_args()
    eval_val_unseen(args.checkpoint_path, args.gpu)
