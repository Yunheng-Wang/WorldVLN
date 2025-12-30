import torch
import torch.nn as nn
from ..utils.wan.modules.model import sinusoidal_embedding_1d
from ..utils.wan.modules.model import sinusoidal_embedding_1d


class VideoModule(nn.Module):

    def __init__(self, video_model, config, dtype, device):
        super().__init__()
        self.video_model = video_model
        self.config = config
        self.dtype = dtype
        self.device = device
    

    def latent_to_token(self, noisy_video_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        video_patched = self.video_model.wan_model.patch_embedding(noisy_video_latent)
        video_token = video_patched.flatten(2).transpose(1, 2)
        return video_token


    def time_embedding(self, t: torch.Tensor, token_num: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = t.unsqueeze(1).expand(t.size(0), token_num)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            b = t.size(0)
            t_flat = t.flatten()
            
            t_emb = self.video_model.wan_model.time_embedding(
                sinusoidal_embedding_1d(self.video_model.wan_model.freq_dim, t_flat).unflatten(0, (b, token_num)).float()
            )
            t_emb_proj = self.video_model.wan_model.time_projection(t_emb).unflatten(2, (6, self.config.video_model_dim))
        return t_emb, t_emb_proj
    

    def compute_adaln_modulation(self, video_adaln_params: torch.Tensor, layer_idx: int) -> tuple:
        wan_layer = self.video_model.wan_model.blocks[layer_idx]
        with torch.amp.autocast('cuda', dtype=torch.float32):
            modulation = (
                wan_layer.modulation.unsqueeze(0)
                + video_adaln_params
            ).chunk(6, dim=2)
        return modulation

