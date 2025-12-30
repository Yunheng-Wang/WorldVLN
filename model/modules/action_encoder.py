import torch.nn as nn
import torch
import numpy as np

from .action_config import ActionModelConfig
from .tool.build_network import build_mlp
from .tool.build_embed import get_1d_pos_embed


class ActionEncoder(nn.Module):
    def __init__(self, config: ActionModelConfig):
        super().__init__()
        self.config = config
        # 1. encoder 主体 - MLP
        self.encoder_block = build_mlp(config.action_dim, config.dim, config.encoder_mlp_depth)
        # 2. 位置编码（不可训练）
        pos_embed = get_1d_pos_embed(config.dim, np.arange(config.chunk_size + config.num_registers))
        self.register_buffer('pos_embedding', pos_embed.unsqueeze(0))


    def forward(self, actions, registers):
        # 1. 过 Encoder 的主体 MLP
        action_encoded = self.encoder_block(actions)
        # 2. 合并 registers token
        encoded = torch.cat([action_encoded, registers], dim=1)
        # 3. 与位置编码相加
        seq_len = encoded.shape[1]
        encoded = encoded + self.pos_embedding[:, :seq_len, :]
        return encoded