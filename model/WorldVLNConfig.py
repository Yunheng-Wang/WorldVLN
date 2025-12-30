import torch
from dataclasses import dataclass



@dataclass 
class WorldVLNConfig:
    batch_size: int = 4
    predict_frame_num: int = 8
    predict_frame_h: int = 384
    predict_frame_w: int = 320

    dtype: torch.dtype = torch.bfloat16

    video_block_num: int = 30

    video_loss_weight: int = 1
    action_loss_weight: int = 1
    
    # 1. Video Model Setting
    video_model_dim: int = 3072

    wan_root: str = "./checkpoints/Wan2.2-TI2V-5B"
    wan_precision: str = "bfloat16"

    # 2. Action Model Setting
    action_model_dim: int = 1024

    action_dim: int = 2
    action_chunk_size: int = 8

    action_encoder_depth: int = 3
    
    action_registers_num: int = 4


    # 3. Understand Model Setting
    vlm_root: str = "./checkpoints/Qwen3-VL-2B-Instruct"
    vlm_projector_input_dim: int = 2048
    vlm_projector_mlp_depth: int = 3
