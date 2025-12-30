import torch
import torch.nn as nn
import os
from transformers import Qwen3VLForConditionalGeneration, AutoConfig, AutoProcessor

from .WorldVLNConfig import WorldVLNConfig
from .modules.video_model import WanVideoModel
from .modules.action_model import ActionModel, ActionModelConfig
from .modules.video_module import VideoModule
from .modules.action_module import ActionModule
from .utils.wan.utils.fm import FlowMatchScheduler
from .modules.understand_model import UnderstandModelConfig, UnderstandModel
from .modules.understand_module import UnderstandModule
from .utils.padding import padding_t5_text

class WorldVLN(nn.Module):
    def __init__(self, config: WorldVLNConfig):
        super().__init__()

        # 1. 配置
        self.config = config
        self.dtype = config.dtype

        # 2. 加载 Video Model
        self.video_model = WanVideoModel.from_pretrained(
                root_path = config.wan_root,
                precision = config.wan_precision
            )
        self.device = next(self.video_model.parameters()).device
        self.video_module = VideoModule(self.video_model, self.config, self.dtype, self.device)
        self.video_model_latent_size = torch.tensor([1 + config.predict_frame_num // 4, config.predict_frame_h // 32, config.predict_frame_w // 32], dtype=torch.long, device=self.device).unsqueeze(0).expand(config.batch_size, -1)


        # 获取必要参数配置
        wan_config = {
            'dim': getattr(self.video_model.wan_model.config, 'dim'),
            'num_heads': getattr(self.video_model.wan_model.config, 'num_heads'), 
            'head_dim': getattr(self.video_model.wan_model.config, 'dim') // getattr(self.video_model.wan_model.config, 'num_heads')
        }


        # 3. 加载 Understand Model
        understand_config = UnderstandModelConfig()
        self.understand_model = UnderstandModel(understand_config, wan_config,self.dtype, config.vlm_root, self.device)
        self.understand_model.to(device=self.device, dtype=self.dtype)
        self.understand_module = UnderstandModule(self.understand_model, self.config, self.dtype, self.device)

        # 4. 加载 Action Model
        action_config = ActionModelConfig()
        self.action_model = ActionModel(action_config, wan_config)
        self.action_model.to(device=self.device, dtype=self.dtype)
        self.action_module = ActionModule(self.action_model, self.config, self.dtype, self.device)


        # 定义 video model 的 Flow-Matching
        self.fm_train_scheduler_video = FlowMatchScheduler(
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000
        )
        self.fm_train_scheduler_video.set_timesteps(num_inference_steps=1000, training=True)
        # 定义 action model 的 Flow-Matching
        self.fm_train_scheduler_action = FlowMatchScheduler(
            shift=5.0,
            sigma_min=0.0,
            extra_one_step=True,
            num_train_timesteps=1000
        )
        self.fm_train_scheduler_action.set_timesteps(num_inference_steps=1000, training=True)


    def training_step(
        self,
        instruction: str,           # string 
        cur_frame: torch.Tensor,    # [B, C, H, W]
        his_frame: torch.Tensor,    # [B, N, C, H, W]
        tar_frame: torch.Tensor,    # [B, N, C, H, W]
        actions: torch.Tensor        # [B, chunk_size, action_dim]
    ):
        B = cur_frame.shape[0]

        # 1. Video Model
        ## 1.1 将 当前帧+未来帧 VAE 编码成 latent（VAE 不会被更新）
        cur_frame_norm = (cur_frame * 2.0 - 1.0).unsqueeze(2)  # [B, C, 1, H, W]
        tar_frame_norm = (tar_frame * 2.0 - 1.0).permute(0, 2, 1, 3, 4)  # [B, C, num_frames, H, W]
        video = torch.cat([cur_frame_norm, tar_frame_norm], dim=2)  # [B, C, frames+1, H, W]
        with torch.no_grad():
            video_latent = self.video_model.encode_video(video.to(self.dtype))  # [B, 48, latent_frames, H', W']
            cur_frame_latent = self.video_model.encode_video(cur_frame_norm.to(self.dtype))  # [B, 48, 1, H', W']
        ## 1.2 时间编码
        timestep_id = torch.randint(0, self.fm_train_scheduler_video.num_train_timesteps, (B,))
        video_t_embed = self.fm_train_scheduler_video.timesteps[timestep_id].to(dtype=self.dtype, device=self.device)  # [B]
        ## 1.3 未来帧的latent 加噪
        sigma = self.fm_train_scheduler_video.sigmas[timestep_id].to(dtype=self.dtype, device=self.device).view(B, 1, 1, 1, 1)
        video_noise = torch.randn_like(video_latent, dtype=self.dtype)
        noisy_video_latent = video_latent * (1 - sigma) + video_noise * sigma
        noisy_video_latent[:, :, 0:1] = cur_frame_latent
        ## 1.4 监督信号
        video_target = video_noise - video_latent
        video_target[:, :, 0:1] = 0
        ## 1.5 带噪的未来帧 latent token化
        video_tokens = self.video_module.latent_to_token(noisy_video_latent.to(self.dtype))
        ## 1.6 获得时间嵌入 和 adaln 调制参数
        video_head_time_emb, video_adaln_params  = self.video_module.time_embedding(video_t_embed, video_tokens.shape[1])
        ## 1.7 获取指令的t5编码
        with torch.no_grad():
            video_text_embedding = self.video_model.t5(instruction, self.device)
            video_text_embedding = padding_t5_text(video_text_embedding, 512)
            video_text_tokens = self.video_model.wan_model.text_embedding(video_text_embedding)
        
        # 2. Action Model 
        ## 2.1 时间编码
        timestep_id_action = torch.randint(0, self.fm_train_scheduler_action.num_train_timesteps, (B,))
        action_t_embed = self.fm_train_scheduler_action.timesteps[timestep_id_action].to(dtype=self.dtype, device=self.device)  # [B]
        ## 2.2 动作加噪
        sigma_action = self.fm_train_scheduler_action.sigmas[timestep_id_action].to(dtype=self.dtype, device=self.device).view(B, 1, 1)
        action_noise = torch.randn_like(actions, dtype=self.dtype)
        noisy_actions = actions * (1 - sigma_action) + action_noise * sigma_action
        ## 2.3 监督信号
        action_target = action_noise - actions
        ## 2.4 带噪的动作 token 化
        registers = self.action_model.registers.expand(B, -1, -1)
        action_tokens = self.action_model.encoder(noisy_actions, registers)
        ## 2.5 获得时间嵌入 和 adaln 调制参数
        action_head_time_emb, action_adaln_params = self.action_module.time_embedding(action_t_embed, action_tokens.shape[1])
        
        # 3. Understand Model
        ## 3.1 VLM 编码 语言 和 历史/当前帧 为 token
        understand_tokens = self.understand_module.encoder_vlm_context_tokens(instruction, his_frame, cur_frame)
        
        # 4. MoT forward
        with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
            for layer_idx in range(self.config.video_block_num):
                ## 4.1 通过归一化层
                video_tokens_processed = self.video_model.wan_model.blocks[layer_idx].norm1(video_tokens)
                action_tokens_processed = self.action_model.blocks[layer_idx].norm1(action_tokens)
                understand_tokens_processed = self.understand_model.blocks[layer_idx].norm1(understand_tokens)
                ## 4.2 通过AdaLN层
                video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                video_tokens_processed = video_tokens_processed * (1 + video_adaln_modulation[1].squeeze(2)) + video_adaln_modulation[0].squeeze(2)
                action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
                action_tokens_processed = action_tokens_processed* (1 + action_adaln_modulation[1].squeeze(2)) + action_adaln_modulation[0].squeeze(2)
                ## 4.3 通过Self-Attention 层
                ### 4.3.1 获取基础参数
                video_tokens_dim = int(video_tokens_processed.shape[2])
                video_tokens_num = int(video_tokens_processed.shape[1])
                batch_size = video_tokens_processed.shape[0]
                head_num = self.video_model.wan_model.num_heads
                head_dim = video_tokens_dim // head_num
                ### 4.3.2 获取 action model 的qkv
                action_tokens_num = action_tokens_processed.shape[1]
                action_qkv = torch.einsum("BTD,KNDE->KBTNE", action_tokens_processed, self.action_model.blocks[layer_idx].wan_action_qkv)
                action_q_h, action_k_h, action_v_h = action_qkv[0], action_qkv[1], action_qkv[2]
                action_q = self.action_model.blocks[layer_idx].wan_action_norm_q(action_q_h.flatten(-2)).view(batch_size, action_tokens_num, head_num, head_dim)
                action_k = self.action_model.blocks[layer_idx].wan_action_norm_k(action_k_h.flatten(-2)).view(batch_size, action_tokens_num, head_num, head_dim)
                action_v = action_v_h.view(batch_size, action_tokens_num, head_num, head_dim)
                ### 4.3.3 获取 understand model 的qkv
                understand_tokens_num = understand_tokens_processed.shape[1]
                understand_qkv = torch.einsum("BTD,KNDE->KBTNE", understand_tokens_processed, self.understand_model.blocks[layer_idx].wan_und_qkv)
                understand_q_h, understand_k_h, understand_v_h = understand_qkv[0], understand_qkv[1], understand_qkv[2]
                understand_q = self.understand_model.blocks[layer_idx].wan_und_norm_q(understand_q_h.flatten(-2)).view(batch_size, understand_tokens_num, head_num, head_dim)
                understand_k = self.understand_model.blocks[layer_idx].wan_und_norm_k(understand_k_h.flatten(-2)).view(batch_size, understand_tokens_num, head_num, head_dim)
                understand_v = understand_v_h.view(batch_size, understand_tokens_num, head_num, head_dim)
                ### 4.3.4 计算联合 self-attention
                freqs = self.video_model.wan_model.freqs.to(self.device)
                seq_lens = torch.full((batch_size,), video_tokens_num + action_tokens_num + understand_tokens_num, dtype=torch.long, device=self.device)
                video_output, action_output, understand_output = self.video_model.wan_model.blocks[layer_idx].self_attn(video_tokens_processed, seq_lens, self.video_model_latent_size, freqs, action_q, action_k, action_v, understand_q, understand_k, understand_v)
                ## 4.4 映射回原有维度
                understand_output = self.understand_model.blocks[layer_idx].video_to_understand_projector(understand_output.flatten(2))
                action_output = self.action_model.blocks[layer_idx].video_to_action_projector(action_output.flatten(2))
                ## 4.5 结合adaln调制参数做残差计算
                video_tokens = video_tokens + video_output * video_adaln_modulation[2].squeeze(2)
                action_tokens = action_tokens + action_output * action_adaln_modulation[2].squeeze(2)
                understand_tokens = understand_tokens + understand_output 
                ## 4.6 通过 cross-attention 层
                ### 4.6.1 过归一化层
                video_tokens = self.video_model.wan_model.blocks[layer_idx].norm3(video_tokens)
                ### 4.6.2 过cross attention层
                video_cross_atten_output = self.video_model.wan_model.blocks[layer_idx].cross_attn(video_tokens, video_text_tokens, None)
                ### 4.6.3 残差计算
                video_tokens = video_tokens + video_cross_atten_output
                ## 4.7 通过AdaLN层 & FFN层
                video_adaln_output = self.video_model.wan_model.blocks[layer_idx].norm2(video_tokens).float() * (1 + video_adaln_modulation[4].squeeze(2)) + video_adaln_modulation[3].squeeze(2)
                video_ffn_output = self.video_model.wan_model.blocks[layer_idx].ffn(video_adaln_output)
                action_adaln_output = self.action_model.blocks[layer_idx].norm2(action_tokens).float() * (1 + action_adaln_modulation[4].squeeze(2)) + action_adaln_modulation[3].squeeze(2)
                action_ffn_output = self.action_model.blocks[layer_idx].ffn(action_adaln_output)
                understand_ffn_output = self.understand_model.blocks[layer_idx].ffn(self.understand_model.blocks[layer_idx].norm2(understand_tokens))
                ## 4.8 残差计算
                video_tokens = video_tokens + video_ffn_output * video_adaln_modulation[5].squeeze(2)
                action_tokens = action_tokens + action_ffn_output * action_adaln_modulation[5].squeeze(2)
                understand_tokens = understand_tokens + understand_ffn_output
        
        # 5. 通过 Video Diffusion Head
        video_predict_head = self.video_model.wan_model.head(video_tokens, video_head_time_emb)
        video_predict = self.video_model.wan_model.unpatchify(video_predict_head, self.video_model_latent_size)
        video_predict = torch.stack([u.float() for u in video_predict], dim=0)
        video_predict[:, :, 0:1] = 0

        # 6. 通过 Action Diffusion Head
        action_predict = self.action_model.decoder(action_tokens, action_head_time_emb)
        action_predict = action_predict[:, :action_predict.shape[1] - self.action_model.config.num_registers, :]
        
        # 7. 计算损失
        video_loss = torch.nn.functional.mse_loss(video_predict, video_target, reduction='mean')
        action_loss = torch.nn.functional.mse_loss(action_predict, action_target, reduction='mean')
        total_loss = self.config.video_loss_weight * video_loss + self.config.action_loss_weight * action_loss
        
        return total_loss, video_loss, action_loss


    def inference_step(
        self,
        instruction: str,           # string 
        cur_frame: torch.Tensor,    # [B, C, H, W]
        his_frame: torch.Tensor,    # [B, N, C, H, W]
        inference_steps_num
    ):
    
        B = cur_frame.shape[0]

        # 1. Video Model
        ## 1.1 将 当前帧 VAE 编码成 latent（VAE 不会被更新）
        cur_frame_norm = (cur_frame * 2.0 - 1.0).unsqueeze(2)  # [B, C, 1, H, W]
        with torch.no_grad():
            cur_frame_latent = self.video_model.encode_video(cur_frame_norm.to(self.dtype))  # [B, 48, 1, H', W']
        ## 1.2 随机初始化未来图像噪声latent
        B, C_latent, f_latent, H_latent, W_latent = cur_frame_latent.shape
        num_total_latent_frames = 1 + self.config.predict_frame_num // 4
        video_latent = torch.randn((B, C_latent, num_total_latent_frames, H_latent, W_latent), device=self.device, dtype=self.dtype)
        video_latent[:, :, 0:1] = cur_frame_latent
        ## 1.3 将 指令编码
        with torch.no_grad():
            video_text_embedding = self.video_model.t5(instruction, self.device)
            video_text_embedding = padding_t5_text(video_text_embedding, 512)
            video_text_tokens = self.video_model.wan_model.text_embedding(video_text_embedding)

        # 3. Action Model
        ## 3.1 随机初始化未来动作序列噪声latent
        action_latent = torch.randn((B, self.config.action_chunk_size, self.config.action_dim), device=self.device, dtype=self.dtype)

        # 4. 去噪
        timesteps = torch.linspace(1.0, 0.0, inference_steps_num + 1, device=self.device, dtype=self.dtype)
        for i in range(inference_steps_num):
            ## 4.1 定义时间步
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            video_t_scaled = (t * 1000).expand(B).to(self.dtype)
            action_t_scaled = (t * 1000).expand(B).to(self.dtype)
            ## 4.2 video latent token化 & adaln 调制参数
            video_tokens = self.video_module.latent_to_token(video_latent.to(self.dtype))
            video_head_time_emb, video_adaln_params  = self.video_module.time_embedding(video_t_scaled, video_tokens.shape[1])
            ## 4.3 action latent token化 & adaln 调制参数
            registers = self.action_model.registers.expand(B, -1, -1)
            action_tokens = self.action_model.encoder(action_latent, registers)
            action_head_time_emb, action_adaln_params = self.action_module.time_embedding(action_t_scaled, action_tokens.shape[1])
            ## 4.4 历史帧 & 指令 token 化 
            understand_tokens = self.understand_module.encoder_vlm_context_tokens(instruction, his_frame, cur_frame)
            ## 4.5 MoT 主结构
            with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                for layer_idx in range(self.config.video_block_num):
                    ### 4.5.1 通过归一化层
                    video_tokens_processed = self.video_model.wan_model.blocks[layer_idx].norm1(video_tokens)
                    action_tokens_processed = self.action_model.blocks[layer_idx].norm1(action_tokens)
                    understand_tokens_processed = self.understand_model.blocks[layer_idx].norm1(understand_tokens)
                    ## 4.5.2 通过AdaLN层
                    video_adaln_modulation = self.video_module.compute_adaln_modulation(video_adaln_params, layer_idx)
                    video_tokens_processed = video_tokens_processed * (1 + video_adaln_modulation[1].squeeze(2)) + video_adaln_modulation[0].squeeze(2)
                    action_adaln_modulation = self.action_module.compute_adaln_modulation(action_adaln_params, layer_idx)
                    action_tokens_processed = action_tokens_processed* (1 + action_adaln_modulation[1].squeeze(2)) + action_adaln_modulation[0].squeeze(2)
                    ## 4.5.3 通过Self-Attention 层
                    ### 4.5.3.1 获取基础参数
                    video_tokens_dim = int(video_tokens_processed.shape[2])
                    video_tokens_num = int(video_tokens_processed.shape[1])
                    batch_size = video_tokens_processed.shape[0]
                    head_num = self.video_model.wan_model.num_heads
                    head_dim = video_tokens_dim // head_num
                    ### 4.5.3.2 获取 action model 的qkv
                    action_tokens_num = action_tokens_processed.shape[1]
                    action_qkv = torch.einsum("BTD,KNDE->KBTNE", action_tokens_processed, self.action_model.blocks[layer_idx].wan_action_qkv)
                    action_q_h, action_k_h, action_v_h = action_qkv[0], action_qkv[1], action_qkv[2]
                    action_q = self.action_model.blocks[layer_idx].wan_action_norm_q(action_q_h.flatten(-2)).view(batch_size, action_tokens_num, head_num, head_dim)
                    action_k = self.action_model.blocks[layer_idx].wan_action_norm_k(action_k_h.flatten(-2)).view(batch_size, action_tokens_num, head_num, head_dim)
                    action_v = action_v_h.view(batch_size, action_tokens_num, head_num, head_dim)
                    ### 4.5.3.3 获取 understand model 的qkv
                    understand_tokens_num = understand_tokens_processed.shape[1]
                    understand_qkv = torch.einsum("BTD,KNDE->KBTNE", understand_tokens_processed, self.understand_model.blocks[layer_idx].wan_und_qkv)
                    understand_q_h, understand_k_h, understand_v_h = understand_qkv[0], understand_qkv[1], understand_qkv[2]
                    understand_q = self.understand_model.blocks[layer_idx].wan_und_norm_q(understand_q_h.flatten(-2)).view(batch_size, understand_tokens_num, head_num, head_dim)
                    understand_k = self.understand_model.blocks[layer_idx].wan_und_norm_k(understand_k_h.flatten(-2)).view(batch_size, understand_tokens_num, head_num, head_dim)
                    understand_v = understand_v_h.view(batch_size, understand_tokens_num, head_num, head_dim)
                    ### 4.5.3.4 计算联合 self-attention
                    freqs = self.video_model.wan_model.freqs.to(self.device)
                    seq_lens = torch.full((batch_size,), video_tokens_num + action_tokens_num + understand_tokens_num, dtype=torch.long, device=self.device)
                    video_output, action_output, understand_output = self.video_model.wan_model.blocks[layer_idx].self_attn(video_tokens_processed, seq_lens, self.video_model_latent_size, freqs, action_q, action_k, action_v, understand_q, understand_k, understand_v)
                    ## 4.5.4 映射回原有维度
                    understand_output = self.understand_model.blocks[layer_idx].video_to_understand_projector(understand_output.flatten(2))
                    action_output = self.action_model.blocks[layer_idx].video_to_action_projector(action_output.flatten(2))
                    ## 4.5.5 结合adaln调制参数做残差计算
                    video_tokens = video_tokens + video_output * video_adaln_modulation[2].squeeze(2)
                    action_tokens = action_tokens + action_output * action_adaln_modulation[2].squeeze(2)
                    understand_tokens = understand_tokens + understand_output 
                    ## 4.5.6 通过 cross-attention 层
                    ### 4.5.6.1 过归一化层
                    video_tokens = self.video_model.wan_model.blocks[layer_idx].norm3(video_tokens)
                    ### 4.5.6.2 过cross attention层
                    video_cross_atten_output = self.video_model.wan_model.blocks[layer_idx].cross_attn(video_tokens, video_text_tokens, None)
                    ### 4.5.6.3 残差计算
                    video_tokens = video_tokens + video_cross_atten_output
                    ## 4.5.7 通过AdaLN层 & FFN层
                    video_adaln_output = self.video_model.wan_model.blocks[layer_idx].norm2(video_tokens).float() * (1 + video_adaln_modulation[4].squeeze(2)) + video_adaln_modulation[3].squeeze(2)
                    video_ffn_output = self.video_model.wan_model.blocks[layer_idx].ffn(video_adaln_output)
                    action_adaln_output = self.action_model.blocks[layer_idx].norm2(action_tokens).float() * (1 + action_adaln_modulation[4].squeeze(2)) + action_adaln_modulation[3].squeeze(2)
                    action_ffn_output = self.action_model.blocks[layer_idx].ffn(action_adaln_output)
                    understand_ffn_output = self.understand_model.blocks[layer_idx].ffn(self.understand_model.blocks[layer_idx].norm2(understand_tokens))
                    ## 4.8 残差计算
                    video_tokens = video_tokens + video_ffn_output * video_adaln_modulation[5].squeeze(2)
                    action_tokens = action_tokens + action_ffn_output * action_adaln_modulation[5].squeeze(2)
                    understand_tokens = understand_tokens + understand_ffn_output

            # 4.6 通过 Video Diffusion Head
            video_predict_head = self.video_model.wan_model.head(video_tokens, video_head_time_emb)
            video_predict = self.video_model.wan_model.unpatchify(video_predict_head, self.video_model_latent_size)
            video_predict = torch.stack([u.float() for u in video_predict], dim=0)
            # 4.7 通过 Action Diffusion Head
            action_predict = self.action_model.decoder(action_tokens, action_head_time_emb)
            action_predict = action_predict[:, :action_predict.shape[1] - self.action_model.config.num_registers, :]
            # 4.5 准备下一次去噪
            video_latent = video_latent + video_predict * dt
            video_latent[:, :, 0:1] = cur_frame_latent
            video_latent = video_latent.to(self.dtype)
            action_latent = action_latent + action_predict * dt
            action_latent = action_latent.to(self.dtype)
        
        # 5. 图像解码
        with torch.no_grad():
            ## 5.1 解码
            decoded_frames = self.video_model.decode_video(video_latent)
            ## 5.2 去掉第一帧
            predicted_frames = decoded_frames[:, :, 1:] 
            ## 5.3 标准化
            predicted_frames = (predicted_frames + 1.0) / 2.0 
            predicted_frames = torch.clamp(predicted_frames, 0, 1).float()
            predicted_frames = predicted_frames.permute(0, 2, 1, 3, 4)
        
        # 6. 标准化动作
        predicted_actions = action_latent.float()

        return predicted_frames, predicted_actions
