import torch.nn as nn
import torch

from .action_config import ActionModelConfig
from ..utils.wan.modules.model import WanRMSNorm, WanLayerNorm, sinusoidal_embedding_1d, rope_apply


class ActionBlock(nn.Module):
    def __init__(self, config: ActionModelConfig, wan_config: dict):
        super().__init__()
        self.config = config
        
        # Layer norms (WAN style) - only need one for joint attention and one for FFN
        self.norm1 = WanLayerNorm(config.dim, eps=config.eps)  # For trimodal joint attention
        self.norm2 = WanLayerNorm(config.dim, eps=config.eps)  # For FFN
        
        # WAN-side action projections and norms (MoT: action -> WAN head space for trimodal joint attention)
        self.wan_num_heads = wan_config['num_heads']
        self.wan_head_dim = wan_config['head_dim']
        self.wan_dim = wan_config['dim']
        assert self.wan_num_heads * self.wan_head_dim == self.wan_dim
        self.wan_action_qkv = nn.Parameter(
            torch.randn(3, self.wan_num_heads, config.dim, self.wan_head_dim)
            / (config.dim * self.wan_head_dim) ** 0.5
        )
        self.video_to_action_projector = nn.Linear(self.wan_dim, config.dim, bias=False)
        # normalize Q/K in WAN unified dim
        self.wan_action_norm_q = WanRMSNorm(self.wan_dim, eps=config.eps)
        self.wan_action_norm_k = WanRMSNorm(self.wan_dim, eps=config.eps)
        
        # FFN (Action Expert's own)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(config.ffn_dim, config.dim)
        )
        
        # Timestep modulation (AdaLN style, 6 parameters)
        # 3 params each for: self-attn residual (WAN-action), FFN (alpha/beta/gamma)
        # self.modulation = nn.Parameter(torch.zeros(1, 6, config.dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, config.dim) / config.dim**0.5)
