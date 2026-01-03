import torch.nn as nn
from .tool.build_network import build_mlp
from transformers import Qwen3VLForConditionalGeneration, AutoConfig, AutoProcessor
from .understand_config import UnderstandModelConfig
from .understand_block import UndModelBlock
from .understand_decoder import UnderstandnDecoder

class UnderstandModel(nn.Module):
    def __init__(self, config: UnderstandModelConfig, wan_config, dtype, vlm_path, device):
        super().__init__()
        self.config = config
        self.freq_dim = 256
        # 1. VLM 映射层（VLM 的 token 转成 WAN 的 token 维度）
        self.vlm_projector = build_mlp(config.vlm_projector_input_dim, config.dim,config.vlm_projector_mlp_depth)
        self.vlm_projector.to(device).to(dtype)
        # 2. 加载 VLM 模型 & 冻结
        self.vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            vlm_path,
            dtype=dtype,
            trust_remote_code=True
        )
        self.vlm_model.to(device)
        for param in self.vlm_model.parameters():
            param.requires_grad = False
        # 3. 加载 VLM 数据处理器
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_path)
        # 4. 加载 block
        self.blocks = nn.ModuleList([UndModelBlock(config, wan_config) for _ in range(config.num_layers)])
        self.blocks.to(device)
        # 5. 加载decoder （二分类 停止/继续）
        self.decoder = UnderstandnDecoder(config)

