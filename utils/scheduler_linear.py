#!/usr/bin/env python3
"""
Learning rate schedulers for joint video-action training.
"""

import torch
from typing import Optional
import math
from diffusers.optimization import get_scheduler as hf_get_scheduler


class LambdaLinearScheduler:
    """在前 warm_up_steps 步，学习率从初始值 f_start 线性增加到最大值 f_max; 
    后续，学习率从最大值 f_max 线性衰减到最小值 f_min，直到达到设定的总步数 cycle_length"""
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.warm_up_steps = config.scheduler.warmup_steps
        self.cycle_length = config.scheduler.cycle_length
        self.f_max = config.scheduler.f_max
        self.f_min = config.scheduler.f_min
        self.f_start = config.scheduler.f_start
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.base_lr = self.base_lrs[0]
        self.step_count = 0

        
    def step(self):
        self.step_count += 1
        lr_multiplier = self.get_lr_multiplier(self.step_count)

        # Apply per-group base lr scaling
        for idx, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[idx] if idx < len(self.base_lrs) else self.base_lr
            param_group['lr'] = base_lr * lr_multiplier
    
    def get_lr_multiplier(self, step: int) -> float:
        """Calculate learning rate multiplier for given step"""
        if step <= 0:
            return self.f_start
        elif step <= self.warm_up_steps:
            # Warmup: linear increase from f_start to f_max
            return self.f_start + (self.f_max - self.f_start) * step / self.warm_up_steps
        elif step < self.cycle_length:
            # Main phase: linear decay from f_max to f_min
            remaining_steps = self.cycle_length - step
            decay_steps = self.cycle_length - self.warm_up_steps
            return self.f_min + (self.f_max - self.f_min) * remaining_steps / decay_steps
        else:
            # After cycle ends, maintain minimum learning rate
            return self.f_min
    
    def get_last_lr(self):
        """Return current learning rates for all parameter groups"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Return scheduler state for checkpointing"""
        return {
            'step_count': self.step_count,
            'base_lr': self.base_lr,
            'base_lrs': self.base_lrs,
            'warm_up_steps': self.warm_up_steps,
            'cycle_length': self.cycle_length,
            'f_max': self.f_max,
            'f_min': self.f_min,
            'f_start': self.f_start,
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint"""
        self.step_count = state_dict['step_count']
        self.base_lr = state_dict.get('base_lr', self.base_lr)
        self.base_lrs = state_dict.get('base_lrs', self.base_lrs)
        self.warm_up_steps = state_dict['warm_up_steps']
        self.cycle_length = state_dict['cycle_length']
        self.f_max = state_dict['f_max']
        self.f_min = state_dict['f_min']
        self.f_start = state_dict['f_start']


