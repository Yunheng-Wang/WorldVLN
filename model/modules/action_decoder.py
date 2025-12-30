import torch.nn as nn
import torch
import numpy as np

from ..utils.wan.modules.model import WanRMSNorm, WanLayerNorm, sinusoidal_embedding_1d, rope_apply
from .tool.build_network import build_mlp
from .action_config import ActionModelConfig

class ActionDecoder(nn.Module):
    def __init__(self, config: ActionModelConfig):
        super().__init__()
        self.config = config

        self.norm = WanLayerNorm(config.dim, eps=config.eps)
        self.action_head = build_mlp(config.dim, config.action_dim, config.decoder_mlp_depth)

        self.modulation = nn.Parameter(torch.randn(1, 2, config.dim) / config.dim**0.5)
    

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e0, e1 = (self.modulation.unsqueeze(0) + time_embedding.unsqueeze(2)).chunk(2, dim=2)
            z = self.norm(x) * (1 + e1.squeeze(2)) + e0.squeeze(2)
            return self.action_head(z)



