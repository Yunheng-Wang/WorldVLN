from accelerate import Accelerator
import os
import json
import re


def load_checkpoint(checkpoint_path, reset_scheduler, accelerator):
    # 1. 加载步数
    step_match = re.search(r'step_(\d+)', checkpoint_path)
    global_step = int(step_match.group(1))
    # 2. 加载模型 & 优化器 & 调度器
    accelerator.load_state(checkpoint_path)

    
    # Reset scheduler with new config if requested
    # if reset_scheduler and self.config is not None and self.scheduler is not None:
    #     logger.info("Resetting scheduler to new configuration (not using checkpoint scheduler state)...")
        
    #     # Unwrap scheduler if it's wrapped by accelerator
    #     unwrapped_scheduler = self.scheduler
    #     if hasattr(self.scheduler, 'module'):
    #         unwrapped_scheduler = self.scheduler.module
        
    #     # Check if it's our custom LambdaLinearScheduler
    #     if hasattr(unwrapped_scheduler, 'warm_up_steps'):
    #         # Update scheduler parameters with new config
    #         unwrapped_scheduler.warm_up_steps = self.config.training.warmup_steps
    #         unwrapped_scheduler.cycle_length = self.config.training.cycle_length
    #         unwrapped_scheduler.f_max = self.config.training.f_max
    #         unwrapped_scheduler.f_min = self.config.training.f_min
    #         # Update base_lrs for all parameter groups
    #         unwrapped_scheduler.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
            
    #         # Reset step_count to 0 so scheduler starts warmup from beginning
    #         unwrapped_scheduler.step_count = 0
            
    #         logger.info(f"Updated scheduler config: warmup={unwrapped_scheduler.warm_up_steps}, "
    #                     f"cycle_length={unwrapped_scheduler.cycle_length}, "
    #                     f"f_max={unwrapped_scheduler.f_max}, f_min={unwrapped_scheduler.f_min}")
    #         logger.info(f"Base learning rates: {[f'{lr:.2e}' for lr in unwrapped_scheduler.base_lrs]}")
            
    #         # Don't directly modify optimizer's lr! Let scheduler update it naturally on next step
    #         # Only log the target lr that scheduler will set
    #         initial_lrs = [base_lr * unwrapped_scheduler.f_max for base_lr in unwrapped_scheduler.base_lrs]
    #         logger.info(f"Reset scheduler step_count to 0 (will start warmup from next step)")
    #         logger.info(f"Target initial learning rates: {[f'{lr:.2e}' for lr in initial_lrs]}")
    #         logger.info(f"Learning rate will be updated by scheduler on first training step")
        
    #     # Log current learning rate (from checkpoint)
    #     current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.optimizer.param_groups[0]['lr']
    #     logger.info(f"Current learning rate after checkpoint load (will be overridden by scheduler): {current_lr:.2e}")
    # elif self.scheduler is not None:
    #     # If not resetting scheduler, sync scheduler progress with global_step
    #     unwrapped_scheduler = self.scheduler
    #     if hasattr(self.scheduler, 'module'):
    #         unwrapped_scheduler = self.scheduler.module

    #     # Case 1: our custom LambdaLinearScheduler
    #     if hasattr(unwrapped_scheduler, 'step_count'):
    #         old_step_count = unwrapped_scheduler.step_count
    #         unwrapped_scheduler.step_count = self.global_step
    #         logger.info(f"Synchronized scheduler step_count: {old_step_count} -> {self.global_step}")

    #     # Case 2: diffusers_cosine wrapper with inner scheduler
    #     if hasattr(unwrapped_scheduler, 'inner') and hasattr(unwrapped_scheduler.inner, 'last_epoch'):
    #         try:
    #             old_epoch = int(getattr(unwrapped_scheduler.inner, 'last_epoch', -1))
    #         except Exception:
    #             old_epoch = -1
    #         # Align inner scheduler epoch with current global_step so schedule continues
    #         unwrapped_scheduler.inner.last_epoch = int(self.global_step)
    #         logger.info(f"Aligned diffusers scheduler last_epoch: {old_epoch} -> {self.global_step}")

    #     # Log current optimizer LR (authoritative)
    #     current_lr = self.optimizer.param_groups[0]['lr']
    #     logger.info(f"Current learning rate after checkpoint load (optimizer): {current_lr:.2e}")




def load_r2r_ce_task(config, data_type):
    with open(os.path.join(config.simulator.path.task_root, "r2r_ce", data_type, data_type + ".json"), 'r', encoding='utf-8') as file:
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