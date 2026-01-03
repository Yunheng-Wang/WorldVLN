from dataclasses import dataclass


@dataclass
class UnderstandModelConfig:
    dim: int = 512
    ffn_dim: int = 2048
    num_layers: int = 30
    vlm_projector_input_dim: int = 2048
    vlm_projector_mlp_depth: int = 3
    eps: float = 1e-5