import torch.nn as nn
import torch
from dataclasses import dataclass


@dataclass
class ActionModelConfig:
    dim: int = 1024 
    ffn_dim: int = 4096     
    num_layers: int = 30

    action_dim: int = 2
    chunk_size: int = 8

    # action encoder
    encoder_mlp_depth: int = 3  
    decoder_mlp_depth: int = 1

    num_registers: int = 4

    # LayerNorm
    eps: float = 1e-6
