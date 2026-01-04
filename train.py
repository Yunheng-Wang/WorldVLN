import torch
import random
import yaml
import logging
import os 
import json
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf
from datetime import datetime

from model.WorldVLNConfig import WorldVLNConfig
from model.WorldVLN import WorldVLN
from utils.model_size import model_size
from data.Dataset import Dataset
from data.utils.load import load_video_num
from utils.scheduler import create_scheduler
from utils.save import save_model_hook, save_checkpoint
from utils.load import load_checkpoint

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

logger = logging.getLogger(__name__)


def setup_logging(rank, save_path = None):
    logging.basicConfig(level=logging.INFO, format=f'[Rank {rank}] %(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter(f'[Rank {rank}] %(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if rank == 0:
        log_file = os.path.join(save_path, 'training.log')
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
        understand_loss_weight = config.model.loss.understand_loss_weight
    )
    model = WorldVLN(model_config)
    # 2. 加载学习率
    base_lr = float(config.optimizer.action_understand_lr)
    wan_lr = float(config.optimizer.video_model_lr)
    # 3. 为不同层分配不同的学习率
    wan_params = [p for p in model.video_model.wan_model.parameters() if p.requires_grad]
    all_trainable = [p for p in model.parameters() if p.requires_grad]
    wan_param_ids = {id(p) for p in wan_params}
    other_params = [p for p in all_trainable if id(p) not in wan_param_ids]
    param_groups = []
    param_groups.append({'params': other_params, 'lr': base_lr})
    param_groups.append({'params': wan_params, 'lr': wan_lr})
    # 4. 构建优化函数
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.optimizer.weight_decay,
        betas=(0.9, 0.95)
    )
    # 5. 学习率变化策略
    scheduler = create_scheduler(optimizer, config)

    return model, optimizer, scheduler


def build_dataloader(config, world_size, rank):
    def seed_worker(worker_id):
        worker_seed = 42 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    # 1. 加载数据
    train_dataset = Dataset(os.path.join(config.main.data_root, "train"), config.main.prediction_steps, config.main.history_steps, config.main.predicted_frame_height, config.main.predicted_frame_width)
    val_unseen_dataset = Dataset(os.path.join(config.main.data_root, "val_unseen"), config.main.prediction_steps, config.main.history_steps, config.main.predicted_frame_height, config.main.predicted_frame_width)
    # 2. 配置加载数据分布式
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_seen_sampler = DistributedSampler(val_unseen_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_seen_sampler = None
    # 3. 创建数据迭代器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = config.main.batch_size,
        shuffle = True if train_sampler is None else False,
        sampler = train_sampler,
        num_workers = config.main.cpu_workers_num,
        pin_memory = True,
        drop_last = True,
        worker_init_fn = seed_worker,
    )
    val_unseen_dataloader = DataLoader(
        val_unseen_dataset,
        batch_size = config.main.batch_size,
        shuffle = False,
        sampler = val_seen_sampler,
        num_workers = config.main.cpu_workers_num,
        pin_memory = True,
        drop_last = False,
        worker_init_fn = seed_worker,
    )

    return train_dataloader, val_unseen_dataloader


def learning():
    # 1. 加载配置参数 & 加载保存根目录
    config = OmegaConf.load('train.yaml')
    os.makedirs(config.main.save_root, exist_ok=True)
    # 2. 配置分布式
    accelerator = Accelerator(
        gradient_accumulation_steps = config.main.gradient.grad_accumulation_steps,
        mixed_precision = config.main.dtype,
        project_dir = config.main.save_root,
        project_config = ProjectConfiguration(total_limit= 20),
    )
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    if rank == 0:
        save_path = os.path.join(config.main.save_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(save_path, exist_ok=True)
        setup_logging(rank, save_path)
    else:
        setup_logging(rank)
    # 3. 加载模型
    if rank == 0:
        print("Loading WorldVLN Model ... ")
    model, optimizer, scheduler = build_model_and_optimizer(config)
    # 4. 加载数据
    if rank == 0:
        print("Loading Data ... ")
    train_dataloader, val_unseen_dataloader = build_dataloader(config, world_size, rank)
    # 5. 配置模型保存设置
    accelerator.register_save_state_pre_hook(lambda models, weights, output_dir: save_model_hook(models, weights, output_dir, accelerator))
    # 6. 分布式分发
    model, optimizer, train_dataloader, val_unseen_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_unseen_dataloader, scheduler)
    # 7. 训练
    if rank == 0:
        print("Start Training ...")
    epoch = 0
    signal = 0
    global_step = 0
    data_iter = iter(train_dataloader)
    while (global_step < config.main.max_steps):
        ## 7.0. 配置为训练模式
        model.train()
        optimizer.zero_grad()
        ## 7.1. 加载一个batch数据
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        ## 7.2 预处理batch
        cur_frame = batch['cur_frame'].to("cuda", dtype = torch.bfloat16)  
        his_video = batch['his_frames'].to("cuda", dtype = torch.bfloat16) 
        pred_video = batch['pred_frames'].to("cuda", dtype = torch.bfloat16) 
        action = batch['action'].to("cuda", dtype = torch.bfloat16) 
        instruction = batch['instruction']
        stop_label = batch['stop_label'].to("cuda", dtype = torch.bfloat16) 
        ## 7.3. 前向推理
        model = model.module if hasattr(model, 'module') else model
        with autocast(dtype=torch.float32):
            total_loss, video_loss, action_loss, understand_loss = model.training_step(instruction, cur_frame, his_video, pred_video, action, stop_label)
        ## 7.4. 梯度同步 & 反向传播
        accelerator.backward(total_loss)
        ## 7.5. 梯度裁剪
        grad_clip_norm = config.main.gradient.grad_clip_norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        ## 7.6. 参数更新 & 学习率调度
        optimizer.step()
        scheduler.step()
        global_step += 1 
        ## 7.7. 刷新训练loss结果
        logger.info(f"Step: {global_step}/{config.main.max_steps}, Epoch: {epoch}, Total Loss: {total_loss:.4f}, Video Loss: {video_loss:.4f}, Action Loss: {action_loss:.4f}, Understand Loss: {understand_loss:.4f}")
        ## 7.8 模型保存 & 验证 val_unseen 结果（每过一个epoch）
        if epoch != signal:
            model.eval()
            val_loss = {"video_mse_loss": [], "action_mse_loss": [], "action_mse_loss_std": [], "action_l2_loss": [], "action_l2_loss_std": [], "stop_accuracy": []}
            for step, batch_val in enumerate(val_unseen_dataloader):
                ### 7.9.1 整理验证集数据
                val_cur_frame = batch_val['cur_frame'].to("cuda", dtype = torch.bfloat16)  
                val_his_video = batch_val['his_frames'].to("cuda", dtype = torch.bfloat16) 
                gt_pred_video = batch_val['pred_frames'].to("cuda", dtype = torch.bfloat16) 
                gt_action = batch_val['action'].to("cuda", dtype = torch.bfloat16) 
                val_instruction = batch_val['instruction']
                stop_label = batch_val['stop_label'].to("cuda", dtype = torch.bfloat16) 
                ## 7.9.2 推理
                with torch.no_grad(): 
                    predicted_frames, predicted_actions, predicted_stop_flag = model.inference_step(val_instruction, val_cur_frame, val_his_video, config.main.inference.steps_for_denoising)
                ## 7.9.3 计算均方误差（模型准确性）
                val_loss["video_mse_loss"].append(F.mse_loss(predicted_frames, gt_pred_video, reduction='mean').item())
                action_mse_loss = F.mse_loss(predicted_actions, gt_action, reduction='none').float()
                val_loss["action_mse_loss"].append(action_mse_loss.reshape(action_mse_loss.shape[0], -1).mean(1).mean().item())
                action_l2_loss = action_mse_loss.sqrt() / (1 + 1e-3)
                action_l2_loss_per_sample = action_l2_loss.reshape(predicted_actions.shape[0], -1).mean(1)
                val_loss["action_l2_loss"].append(action_l2_loss.reshape(predicted_actions.shape[0], -1).mean(1).mean().item())
                ## 7.9.4 计算误差标准差（模型稳定性）
                val_loss["action_mse_loss_std"].append(action_mse_loss.reshape(action_mse_loss.shape[0], -1).mean(1).std().item())
                val_loss["action_l2_loss_std"].append(action_l2_loss.reshape(predicted_actions.shape[0], -1).mean(1).std().item())
                ## 7.9.5 计算停止符准确性
                val_stop_accuracy = (predicted_stop_flag == stop_label.view(-1)).float().mean()
                val_loss["stop_accuracy"].append(val_stop_accuracy.item())

            if dist.is_initialized():
                gathered_losses = {}
                for key in val_loss:
                    local_tensor = torch.tensor(val_loss[key], dtype=torch.float32, device=accelerator.device)
                    gathered_tensor = [torch.zeros_like(local_tensor) for _ in range(world_size)]
                    dist.all_gather(gathered_tensor, local_tensor)
                    all_values = torch.cat(gathered_tensor, dim=0)
                    gathered_losses[key] = all_values.cpu().numpy().tolist()
            else:
                gathered_losses = val_loss
            
            logger.info(
                f"Evaluation Validation, "
                f"Video Loss: {np.mean(gathered_losses['video_mse_loss']):.4f}, "
                f"Action Loss: {np.mean(gathered_losses['action_mse_loss']):.4f}, "
                f"Action MSE Loss Std: {np.mean(gathered_losses['action_mse_loss_std']):.4f}, "
                f"Action L2 Loss: {np.mean(gathered_losses['action_l2_loss']):.4f}, "
                f"Action L2 Loss Std: {np.mean(gathered_losses['action_l2_loss_std']):.4f}, "
                f"Stop Accuracy: {np.mean(gathered_losses['stop_accuracy']):.4f}"
            )           
            model.train()
            dist.barrier()

            save_checkpoint(accelerator, config, global_step, epoch, save_path)
            signal = epoch
            dist.barrier()
    

    # 8. 清理资源
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    learning()

    # CUDA_VISIBLE_DEVICES=0,4 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision fp16 --dynamo_backend no train.py

