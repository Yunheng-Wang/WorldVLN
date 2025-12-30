import torch
import random
import yaml
import logging
import os 
import json
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf

from model.WorldVLNConfig import WorldVLNConfig
from model.WorldVLN import WorldVLN
from utils.model_size import model_size
from data.Dataset import Dataset
from data.utils.load import load_video_num
from utils.scheduler import create_scheduler

logger = logging.getLogger(__name__)


def setup_logging(rank):
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def build_model_and_optimizer(config):
    # 1. 加载模型
    model_config = WorldVLNConfig(
        batch_size = config.main.batch_size,
        dtype = torch.bfloat16 if config.model.dtype == "bf16" else torch.float32,
        predict_frame_num = config.main.prediction_steps,

        predict_frame_h = config.model.video_model.future_frame_height,
        predict_frame_w = config.model.video_model.predicted_frame_width,
        video_block_num = config.model.video_model.video_block_num

    )
    model = WorldVLN(model_config)
    # 2. 加载学习率
    base_lr = float(config.main.optimizer.learning_rate)
    wan_lr = float(config.main.optimizer.wan_learning_rate)
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
        weight_decay=config.main.optimizer.weight_decay,
        betas=(0.9, 0.95)
    )
    # 5. 学习率变化策略
    scheduler = create_scheduler(optimizer, config)
    return model, optimizer, scheduler


def build_dataloader(config, world_size, rank):
    # 1. 加载数据
    train_dataset = Dataset(os.path.join(config.main.dataloader.data_root, "train"), config.main.predict_frames_num, config.main.history_frames_num, config.main.image_height, config.main.image_width)
    val_seen_dataset = Dataset(os.path.join(config.main.dataloader.data_root, "val_seen"), config.main.predict_frames_num, config.main.history_frames_num, config.main.image_height, config.main.image_width)
    # 2. 配置加载数据分布式
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=config.main.dataloader.shuffle, drop_last=config.main.dataloader.drop_last)
        val_seen_sampler = DistributedSampler(val_seen_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_seen_sampler = None
    # 3. 创建数据迭代器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = config.main.batch_size,
        shuffle = config.main.dataloader.shuffle if train_sampler is None else False,
        sampler = train_sampler,
        num_workers = config.main.dataloader.cpu_workers_num,
        pin_memory = config.main.dataloader.pin_memory,
        drop_last = config.main.dataloader.drop_last,
    )
    val_seen_dataloader = DataLoader(
        val_seen_dataset,
        batch_size = config.main.batch_size,
        shuffle = False,
        sampler = val_seen_sampler,
        num_workers = config.main.dataloader.cpu_workers_num,
        pin_memory = config.main.dataloader.pin_memory,
        drop_last=False,
    )

    return train_dataloader, val_seen_dataloader


def save_checkpoint(accelerator, config, step):
    # 1. 保存权重
    checkpoint_dir = os.path.join(config.main.save.checkpoints_output, "checkpoint_step_" + str(step))
    accelerator.save_state(str(checkpoint_dir))
    logger.info(f"Checkpoint saved to {checkpoint_dir}")
    # # 2. 保存配置
    # cfg = OmegaConf.to_container(config, resolve=True)
    # with open(checkpoint_dir / "config.json", "w") as f:
    #     json.dump(filtered, f, indent=2)


def learning():
    # 1. 加载配置参数
    config = OmegaConf.load('train.yaml')
    # 2. 配置分布式
    accelerator = Accelerator(
        gradient_accumulation_steps = config.main.gradient.grad_accumulation_steps,
        mixed_precision = config.main.precision,
        project_dir = config.main.save.checkpoints_output,
        project_config = ProjectConfiguration(total_limit= config.main.save.checkpoints_history_num),
    )
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    setup_logging(rank)
    # 3. 加载模型
    if rank == 0:
        print("Loading WorldVLN Model ... ")
    model, optimizer, scheduler = build_model_and_optimizer(config)
    # 4. 加载数据
    if rank == 0:
        print("Loading Data ... ")
    train_dataloader, val_seen_dataloader = build_dataloader(config, world_size, rank)
    
    def save_model_hook(models, weights, output_dir):
        """Custom save hook to save model safely and avoid NCCL timeouts."""
        if accelerator.is_main_process:
            logger.info(f"Saving model to {output_dir}")
            for i, model_to_save in enumerate(models):
                unwrapped_model = accelerator.unwrap_model(model_to_save)
                model_save_path = os.path.join(output_dir, f"pytorch_model_{i}.bin")
                torch.save(unwrapped_model.state_dict(), model_save_path)
                logger.info(f"Model {i} saved to {model_save_path}")
    
    accelerator.register_save_state_pre_hook(save_model_hook)
    
    
    
    # 5. 分布式分发
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    # 
    logger.info(f"================ Start Training ================")
    epoch = 0
    global_step = 0
    data_iter = iter(train_dataloader)
    while global_step < config.main.max_steps:
        # 0. 配置为训练模式
        model.train()
        optimizer.zero_grad()
        # 1. 整理数据
        try:
            batch = next(data_iter)
            print(batch['cur_frame'].shape[0])
        except StopIteration:
            # End of epoch, restart dataloader
            epoch += 1
            if hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        # 
        cur_frame = batch['cur_frame'].to("cuda", dtype = torch.bfloat16)  
        his_video = batch['his_frames'].to("cuda", dtype = torch.bfloat16) 
        pred_video = batch['pred_frames'].to("cuda", dtype = torch.bfloat16) 
        action = batch['action'].to("cuda", dtype = torch.bfloat16) 
        instruction = batch['instruction']
        # 2. 前向推理
        model = model.module if hasattr(model, 'module') else model
        with autocast(dtype=torch.float32):
            total_loss, video_loss, action_loss = model.training_step(instruction, cur_frame, his_video, pred_video, action)
        # 3. 梯度同步 & 反向传播
        accelerator.backward(total_loss)
        # 4. 梯度裁剪
        grad_clip_norm = config.main.gradient.grad_clip_norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        # 5. 参数更新 & 学习率调度
        optimizer.step()
        scheduler.step()
        global_step += 1 
        # 6. 记录
        if rank == 0:
            logger.info(f"Step {global_step}/{config.main.max_steps}, Total Loss: {total_loss:.4f}, Video Loss: {video_loss:.4f}, Action Loss: {action_loss:.4f}")
        
        # 7. 验证 val_unseen 
        if global_step % config.interval.eval_val_unseen == 0:
            # 7.1 主进程评估验证集
            if rank == 0:
                model.eval()
                val_loss = {"video_mse_loss": [], "action_mse_loss": [], "action_mse_loss_std": [], "action_l2_loss": [], "action_l2_loss_std": []}
                for step, batch_val in enumerate(val_seen_dataloader):
                    ## 整理验证集数据
                    val_cur_frame = batch['cur_frame'].to("cuda", dtype = torch.bfloat16)  
                    val_his_video = batch['his_frames'].to("cuda", dtype = torch.bfloat16) 
                    gt_pred_video = batch['pred_frames'].to("cuda", dtype = torch.bfloat16) 
                    gt_action = batch['action'].to("cuda", dtype = torch.bfloat16) 
                    val_instruction = batch['instruction']
                    ## 验证集推理
                    with torch.no_grad(): 
                        predicted_frames, predicted_actions = model.inference_step(val_instruction, val_cur_frame, val_his_video, config.inference.steps_for_denoising)
                    ## 计算均方误差（模型准确性）
                    val_loss["video_mse_loss"].append(F.mse_loss(predicted_frames, gt_pred_video, reduction='mean').item())
                    action_mse_loss = F.mse_loss(predicted_actions, gt_action, reduction='none').float()
                    val_loss["action_mse_loss"].append(action_mse_loss.reshape(action_mse_loss.shape[0], -1).mean(1).mean().item())
                    action_l2_loss = action_mse_loss.sqrt() / (1 + 1e-3)
                    action_l2_loss_per_sample = action_l2_loss.reshape(predicted_actions.shape[0], -1).mean(1)
                    val_loss["action_l2_loss"].append(action_l2_loss.reshape(predicted_actions.shape[0], -1).mean(1).mean().item())
                    ## 计算误差标准差（模型稳定性）
                    val_loss["action_mse_loss_std"].append(action_mse_loss.reshape(action_mse_loss.shape[0], -1).mean(1).std().item())
                    val_loss["action_l2_loss_std"].append(action_l2_loss.reshape(predicted_actions.shape[0], -1).mean(1).std().item())
                ## 打印
                logger.info(f"Evaluation Validation, Video Loss: {np.mean(val_loss['video_mse_loss']):.4f}, "
                            f"Action Loss: {np.mean(val_loss['action_mse_loss']):.4f}, "
                            f"Action MSE Loss Std: {np.mean(val_loss['action_mse_loss_std']):.4f}, "
                            f"Action L2 Loss: {np.mean(val_loss['action_l2_loss']):.4f}, "
                            f"Action L2 Loss Std: {np.mean(val_loss['action_l2_loss_std']):.4f}")                
                model.train()

            ## 7.1 同步进程
            dist.barrier()

        # 8. 验证 val_seen 准确率

        # 8. 保存模型权重
        # save_checkpoint(accelerator, config, global_step)






if __name__ == "__main__":
    learning()

    # CUDA_VISIBLE_DEVICES=0,4 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision fp16 --dynamo_backend no train.py

