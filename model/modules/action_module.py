import torch
import torch.nn as nn

from .action_model import ActionModel
from ..utils.wan.modules.model import sinusoidal_embedding_1d


class ActionModule(nn.Module):
    def __init__(self, action_model: ActionModel, config, dtype, device):
        super().__init__()
        self.action_model = action_model
        self.config = config
        self.dtype = dtype
        self.device = device

    def time_embedding(self, t: torch.Tensor, token_num: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = t.unsqueeze(1).expand(t.size(0), token_num)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            b = t.size(0)
            t_flat = t.flatten()
            
            t_emb = self.action_model.time_embedding(
                sinusoidal_embedding_1d(self.action_model.freq_dim, t_flat).unflatten(0, (b, token_num)).float()
            )
            t_emb_proj = self.action_model.time_projection(t_emb).unflatten(2, (6, self.config.action_model_dim))
        return t_emb, t_emb_proj
    

    def compute_adaln_modulation(self, action_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        """Compute AdaLN modulation parameters for 6 components (3 for WAN-Action joint attn + 3 for FFN)."""
        action_layer = self.action_model.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (
                action_layer.modulation.unsqueeze(0)
                + action_adaln_params
            ).chunk(6, dim=2)
        return modulation