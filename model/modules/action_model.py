import torch.nn as nn
import torch
from dataclasses import dataclass

from .action_config import ActionModelConfig
from .action_block import ActionBlock
from .action_encoder import ActionEncoder
from .action_decoder import ActionDecoder
    

class ActionModel(nn.Module):
    def __init__(self, config: ActionModelConfig, wan_config):
        super().__init__()
        self.config = config
        self.freq_dim = 256

        # 1. 动作 encoder 层
        self.encoder = ActionEncoder(config)

        # 2. 时间步编码层 (同 WAN)
        self.time_embedding = nn.Sequential(
            nn.Linear(256, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, config.dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.dim, config.dim * 6)  # 6 parameters: 3 for WAN-Action joint attn + 3 for FFN
        )

        # 3. 创建 block 块 
        self.blocks = nn.ModuleList([
            ActionBlock(config, wan_config) for _ in range(config.num_layers)
        ])

        # 4. 创建 registers token
        self.registers = nn.Parameter(
            torch.empty(1, config.num_registers, config.dim).normal_(std=0.02)
        )

        # 5. 动作 decoder 层 
        self.decoder = ActionDecoder(config)
