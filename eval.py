import torch
import random
import yaml
import logging
import os 
import json
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import habitat_sim
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf
from datetime import datetime

from model.WorldVLNConfig import WorldVLNConfig
from model.WorldVLN import WorldVLN
from data.Dataset_Random import Dataset_Random
from data.Dataset_Normal import Dataset_Normal_Train, Dataset_Normal_Val
from data.utils.load import load_video_num
from utils.scheduler_linear import LambdaLinearScheduler
from utils.save import save_model_hook, save_checkpoint
from utils.load import load_checkpoint
from accelerate.utils import InitProcessGroupKwargs
from datetime import datetime, timedelta
from utils.tool import print_model_size
from utils.habitat_sim import environment_multi_agents, find_more_paths_for_task, action_point, get_yaw_and_dist, get_all_agent_observation, get_agent_id_observation
from accelerate.utils import gather_object

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)


logger = logging.getLogger(__name__)
logging.getLogger("accelerate").setLevel(logging.ERROR)


def setup_logging(rank, save_path):
    logging.basicConfig(level=logging.INFO, format=f'[Rank {rank}] %(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter(f'[Rank {rank}] %(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log_file = os.path.join(save_path, 'evaling.log')
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter) 
    logging.getLogger().addHandler(file_handler)  
    

def build_model_and_optimizer(config):
    # 1. 加载模型
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

        # strategy
        t5_text_encoder = False

    )
    model = WorldVLN(model_config)
    return model


def build_dataloader(config, world_size, rank):
    def seed_worker(worker_id):
        worker_seed = 42 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    # 1. 加载数据
    val_unseen_dataset = Dataset_Normal_Val(os.path.join(config.main.data_root, "val_unseen"), config.main.prediction_steps, config.main.history_steps, config.main.predicted_frame_height, config.main.predicted_frame_width, config, "all")

    # 2. 配置加载数据分布式
    if world_size > 1:
        val_seen_sampler = DistributedSampler(val_unseen_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    else:
        val_seen_sampler = None
    # 3. 创建数据迭代器
    val_unseen_dataloader = DataLoader(
        val_unseen_dataset,
        batch_size = 1,
        shuffle = False,
        sampler = val_seen_sampler,
        num_workers = config.main.cpu_workers_num,
        pin_memory = True,
        drop_last = False,
        worker_init_fn = seed_worker,
    )

    return val_unseen_dataloader


def eval(checkpoint_path, batch_size, max_execution_steps, min_execution_steps):
    # 1. 加载配置参数 & 加载保存根目录
    config = OmegaConf.load(os.path.join(checkpoint_path, 'config.json'))
    config.main.batch_size = batch_size
    config.eval.run_max_execution_steps = max_execution_steps
    config.eval.run_min_execution_steps = min_execution_steps
    # 2. 配置分布式
    accelerator = Accelerator(
        gradient_accumulation_steps = config.main.gradient.grad_accumulation_steps,
        mixed_precision = config.main.dtype,
        project_dir = config.main.save_root,
        project_config = ProjectConfiguration(total_limit= 20),
        kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=3600))]
    )
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    # 
    save_path = os.path.dirname(checkpoint_path)
    setup_logging(rank, save_path)
    # 3. 加载模型
    if rank == 0:
        print("Loading WorldVLN Model ... ")
    model = build_model_and_optimizer(config)
    # 4. 加载数据
    if rank == 0:
        print("Loading Data ... ")
    val_unseen_dataloader = build_dataloader(config, world_size, rank)
    # 5. 配置模型保存设置
    # 6. 分布式分发
    model = accelerator.prepare(model)
    # 7. 评估
    if rank == 0:
        print("Start Evaling ...")
    accelerator.load_state(checkpoint_path)
    model.eval()
    results = {}
    for step_global, batch_val in enumerate(val_unseen_dataloader):
        # 7.8.0 整合指令
        instruction = []
        instruction_embed = []
        # 7.8.1 创建虚拟环境
        simulator, _ = environment_multi_agents(config, batch_val[0]["scene_id"][0], len(batch_val), accelerator.device)
        # 7.8.2 创建agent
        for agent_id in range(len(batch_val)):
            instruction.append(batch_val[agent_id]['instruction'][0])
            instruction_embed.append(batch_val[agent_id]["instruction_embed"].squeeze(0).to(device = accelerator.device))
            agent = simulator.initialize_agent(agent_id)
            initial_state = habitat_sim.AgentState()
            initial_state.position = batch_val[agent_id]["start_position"].tolist()[0]
            initial_state.rotation = batch_val[agent_id]["start_rotation"].tolist()[0]
            agent.set_state(initial_state)
            batch_val[agent_id]["reference_path"] = find_more_paths_for_task(simulator, batch_val[agent_id]["reference_path"].tolist()[0])
            yaw_err, _, dist_planar = get_yaw_and_dist(simulator, batch_val[agent_id]["reference_path"][1], agent_id)
            action_point(simulator, 0, yaw_err, agent_id)
        # 7.8.3 核心执行
        all_hist_obers = [[] for i in range(len(simulator.agents))]
        step = 0
        L = [ 0 for i in range(len(simulator.agents))]
        stop = [ 0 for i in range(len(simulator.agents))] # 1 则停止
        his_pose = [[] for i in range(len(simulator.agents))]
        while step < config.eval.run_max_execution_steps:
            if all(sto == 1 for sto in stop):
                break  
            # 7.8.3.1 获取当前观测
            curr_obers = get_all_agent_observation(simulator, config, accelerator.device)
            # 7.8.3.2 获取历史观测
            input_hist_obers = []
            for agent in range(len(simulator.agents)):
                if stop[agent] == 0:
                    if len(all_hist_obers[agent]) < config.eval.history_frames:
                        num_missing_frames = config.eval.history_frames - len(all_hist_obers[agent])
                        zero_frame = torch.zeros_like(curr_obers[agent], device = accelerator.device) 
                        zero_frames = zero_frame.unsqueeze(0).repeat(num_missing_frames, 1, 1, 1)
                        input_hist_obers.append(torch.cat([torch.tensor(all_hist_obers[agent], device = accelerator.device), zero_frames], dim=0))
                        all_hist_obers[agent].append(curr_obers[agent])
                    elif len(all_hist_obers[agent]) == config.eval.history_frames:
                        input_hist_obers.append(torch.stack(all_hist_obers[agent], dim=0).to(accelerator.device))
                        all_hist_obers[agent].append(curr_obers[agent])
                    else:
                        his_frames = random.sample(all_hist_obers[agent], config.eval.history_frames)
                        input_hist_obers.append(torch.stack(his_frames, dim=0).to(accelerator.device))
                        all_hist_obers[agent].append(curr_obers[agent])
            # 7.8.3.3 模型推理
            model = model.module if hasattr(model, 'module') else model
            if config.eval.run_min_execution_steps > step:
                with torch.no_grad():
                    predicted_frames, predicted_actions, predicted_stop_flag = model.inference_step([instr for instr, sto in zip(instruction, stop) if sto == 0], torch.stack([instr_embed for instr_embed, sto in zip(instruction_embed, stop) if sto == 0], dim=0), curr_obers[torch.tensor(stop) == 0],  torch.stack(input_hist_obers, dim=0), config.main.inference.steps_for_denoising, False, False)
                    predicted_actions = predicted_actions[:, :config.eval.agent_execution_steps, :].tolist()
            else:
                with torch.no_grad():
                    predicted_frames, predicted_actions, predicted_stop_flag = model.inference_step([instr for instr, sto in zip(instruction, stop) if sto == 0], torch.stack([instr_embed for instr_embed, sto in zip(instruction_embed, stop) if sto == 0], dim=0), curr_obers[torch.tensor(stop) == 0],  torch.stack(input_hist_obers, dim=0), config.main.inference.steps_for_denoising, True, False)
                    predicted_actions = predicted_actions[:, :config.eval.agent_execution_steps, :].tolist()
            # 7.8.3.4 判断是否到达目的地
            if config.eval.run_min_execution_steps <= step:
                effective_idx = [i for i, x in enumerate(stop) if x == 0]
                for idx, sig_stop in enumerate(predicted_stop_flag):
                    if sig_stop == 1:
                        stop[effective_idx[idx]] = 1
                if all(sto == 1 for sto in stop):
                    break 
            # 7.8.3.5 物理执行
            effective_idx = [i for i, x in enumerate(stop) if x == 0]
            for agent in range(len(simulator.agents)):
                if stop[agent] == 0: 
                    for idx, action in enumerate(predicted_actions[effective_idx.index(agent)]):
                        prev_pos = np.array(simulator.get_agent(agent).get_state().position, dtype=float)
                        action_point(simulator, action[0], action[1], agent)
                        new_pos = np.array(simulator.get_agent(agent).get_state().position, dtype=float)
                        his_pose[agent].append(new_pos)
                        L[agent] += float(np.linalg.norm(new_pos - prev_pos))
                        if idx < config.eval.agent_execution_steps - 1:
                            all_hist_obers[agent].append(get_agent_id_observation(simulator, config, agent, accelerator.device))
            step += 1
        # 7.8.4 评估SR & NE
        SR = [0 for _ in range(len(simulator.agents))]
        final_pos = [np.array(simulator.get_agent(agent).get_state().position[[0, 2]], dtype=float) for agent in range(len(simulator.agents))]
        gt_pos = [np.array(batch_val[agent]["goal_position"].cpu(), dtype=float).flatten()[[0, 2]] for agent in range(len(simulator.agents))]
        d = [np.linalg.norm(final_pos[agent] - gt_pos[agent]) for agent in range(len(simulator.agents))]
        for i in range(len(d)):
            if d[i] <= 3:
                SR[i] = 1
        # 7.8.5 评估 SRL
        SPL = [SR[agent] * (batch_val[agent]['reference_distant'] / max(L[agent], batch_val[agent]['reference_distant'])) for agent in range(len(simulator.agents))]
        # 7.8.6 评估 OSR
        OSR = [0 for _ in range(len(simulator.agents))]
        for agent in range(len(simulator.agents)):
            for i in his_pose[agent]:
                if np.linalg.norm(i[[0, 2]] - gt_pos[agent]) <= 3:
                    OSR[agent] = 1
                    break
        # 7.8.7 整合
        for i in range(len(batch_val)):
            results[str(batch_val[i]["episode_id"].item())] = {"TL": L[i], "NE": d[i], "OSR": OSR[i], "SR": SR[i], "SPL": float(SPL[i])}
        # 7.8.8 打印日志
        logger.info(
            f'Validation (Accumulated), Step: {step_global}, '
            f'SR: {sum(v.get("SR", 0) for v in results.values()) / len(results):.4f}, '
            f'SPL: {sum(v.get("SPL", 0) for v in results.values()) / len(results):.4f}, '
            f'NE: {sum(v.get("NE", 0) for v in results.values()) / len(results):.4f}, '
            f'TL: {sum(v.get("TL", 0) for v in results.values()) / len(results):.4f}, '
        )
        # 7.8.10 消除环境
        simulator.close()
    # 8. 收集所有 GPU 的字典结果并保存
    accelerator.wait_for_everyone()
    gathered_results_list = gather_object([results])
    if rank == 0:
        final_results = {}
        for res in gathered_results_list:
            final_results.update(res)
        total_count = len(final_results)
        if total_count > 0:
            avg_sr = sum(v.get("SR", 0) for v in final_results.values()) / total_count
            avg_spl = sum(v.get("SPL", 0) for v in final_results.values()) / total_count
            avg_ne = sum(v.get("NE", 0) for v in final_results.values()) / total_count
            avg_tl = sum(v.get("TL", 0) for v in final_results.values()) / total_count
            avg_osr = sum(v.get("OSR", 0) for v in final_results.values()) / total_count
            logger.info("==================================================")
            logger.info(f"Final Evaluation Results (Total Episodes: {total_count}):")
            logger.info(f"SR:  {avg_sr:.4f}")
            logger.info(f"SPL: {avg_spl:.4f}")
            logger.info(f"NE:  {avg_ne:.4f}")
            logger.info(f"TL:  {avg_tl:.4f}")
            logger.info(f"OSR: {avg_osr:.4f}")
            logger.info("==================================================")
        final_file_path = os.path.join(save_path, "evaluation" + os.path.basename(checkpoint_path) + ".json")
        with open(final_file_path, "w") as f:
            json.dump(final_results, f, indent=4)
        print(f"All results saved to {final_file_path}")
        
    # 9. 清理资源
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()



if __name__ == "__main__":
    checkpoint_path = "/home/CONNECT/yfang870/yunhengwang/WorldVLN/log/2026-01-16_10/checkpoint_step_2_20763"
    batch_size = 32
    man_execution_steps = 10
    min_execution_steps = 5
    eval(checkpoint_path, batch_size, man_execution_steps, min_execution_steps)
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu --num_processes 8 --num_machines 1 --mixed_precision fp16 --dynamo_backend no eval.py
