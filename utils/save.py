import os
import json
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
import logging

logger = logging.getLogger(__name__)


def save_model_hook(models, weights, output_dir, accelerator):
    """Custom save hook to save model safely and avoid NCCL timeouts."""
    if accelerator.is_main_process:
        logger.info(f"Saving model to {output_dir}")
        for i, model_to_save in enumerate(models):
            unwrapped_model = accelerator.unwrap_model(model_to_save)
            model_save_path = os.path.join(output_dir, f"pytorch_model_{i}.bin")
            torch.save(unwrapped_model.state_dict(), model_save_path)
            logger.info(f"Model {i} saved to {model_save_path}")


def save_checkpoint(accelerator, config, step):
    # 1. 保存权重 & 状态
    checkpoint_dir = os.path.join(config.main.save.checkpoints_output, "checkpoint_step_" + str(step))
    accelerator.save_state(str(checkpoint_dir))
    logger.info(f"Checkpoint saved to {checkpoint_dir}")
    # 2. 保存配置
    cfg = OmegaConf.to_container(config, resolve=True)
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)