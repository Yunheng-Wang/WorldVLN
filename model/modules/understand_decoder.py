import torch.nn as nn
import torch
import numpy as np

from ..utils.wan.modules.model import WanLayerNorm
from .tool.build_network import build_mlp, build_binary_mlp
from .understand_config import UnderstandModelConfig
import torch.nn.functional as F


class AttentionPooler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score_proj = nn.Linear(dim, 1)  # [B,T,D] -> [B,T,1]

    def forward(self, x):
        scores = self.score_proj(x).squeeze(-1)  # [B,T]
        attn = F.softmax(scores, dim=1)          # [B,T]
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)  # [B,D]
        return pooled


class UnderstandnDecoder(nn.Module):
    def __init__(self, config: UnderstandModelConfig):
        super().__init__()
        self.config = config
        self.norm = WanLayerNorm(config.dim, eps=config.eps)
        self.pooler = AttentionPooler(config.dim)
        self.understand_decoder = build_binary_mlp(config.dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = self.norm(x)
            x = self.pooler(x)
            x = self.understand_decoder(x)
            return x