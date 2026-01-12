import torch
from dataclasses import dataclass


@dataclass 
class WorldVLNConfig:
    # 1. common setting
    batch_size: int = 4
    predict_frame_num: int = 8
    history_frame_num: int = 8
    predict_frame_h: int = 384
    predict_frame_w: int = 320
    dtype: torch.dtype = torch.bfloat16

    # 1. video setting
    video_block_num: int = 30
    video_model_dim: int = 3072
    video_wan_root: str = "./checkpoints/Wan2.2-TI2V-5B"
    video_wan_precision: str = "bfloat16"

    # 2. action setting
    action_dim: int = 2
    action_model_dim: int = 1024
    action_model_ffn_dim: int = 4096
    action_block_num: int = 30
    action_encoder_depth: int = 3
    action_decoder_depth: int = 1
    action_registers_num: int = 4
    action_eps: float = 1e-6

    # 3. understand setting
    understand_vlm_root: str = "./checkpoints/Qwen3-VL-2B-Instruct"
    understand_model_dim: int = 512
    understand_model_ffn_dim: int = 2048
    understand_block_num: int = 30

    understand_vlm_token_dim: int = 2048
    understand_vlm_projector_mlp_depth: int = 3
    understand_eps: float = 1e-5

    # 4. loss
    video_loss_weight: int = 1
    action_loss_weight: int = 1

    # 5. strategy 
    t5_text_encoder: bool = True