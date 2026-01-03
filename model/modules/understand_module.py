import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from qwen_vl_utils import process_vision_info


class UnderstandModule(nn.Module):
    def __init__(self, udnerstand_model, config, dtype, device):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device
        self.undstand_model = udnerstand_model
    

    def encoder_vlm_context_tokens(self, instruction_batch, history_video, current_frame):
        # 1. 数据预处理
        B = current_frame.shape[0]
        current_frame_cpu = current_frame.detach().cpu()
        history_video_cpu = history_video.detach().cpu()
        curr_imgs_batch = [Image.fromarray((current_frame_cpu[b].float().permute(1, 2, 0).numpy() * 255).astype(np.uint8), mode='RGB') for b in range(current_frame_cpu.shape[0])]
        hist_vide_batch = [[Image.fromarray((frame.float().permute(1, 2, 0).numpy() * 255).astype(np.uint8), mode='RGB') for frame in video] for video in history_video_cpu]

        # 2. 构建VLM输入
        ## 2.1 第一步（无历史帧）
        if len(hist_vide_batch) == 0:
            input_ids_list = []
            attention_mask_list = []
            pixel_values_list = []
            image_grid_thw_list = []
            for index in range(B):
                curr_imgs = curr_imgs_batch[index]
                messages = [
                    {
                        "role": "user",
                        "content": (
                            [{"type": "image", "image": curr_imgs}]
                            +
                            [{"type": "text", "text": instruction_batch[index]}]
                        )
                    }
                ]
                text = self.undstand_model.vlm_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.undstand_model.vlm_processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids_list.append(inputs["input_ids"])          
                attention_mask_list.append(inputs["attention_mask"])
                pixel_values_list.append(inputs["pixel_values"])  
                image_grid_thw_list.append(inputs["image_grid_thw"]) 
        ## 2.1 非第一步（有历史帧）
        else:
            input_ids_list = []
            attention_mask_list = []
            pixel_values_list = []
            image_grid_thw_list = []
            for index in range(B):
                curr_imgs = curr_imgs_batch[index]
                hist_vide = hist_vide_batch[index]
                messages = [
                    {
                        "role": "user",
                        "content": (
                            [{"type": "image", "image": img} for img in hist_vide]
                            +
                            [{"type": "image", "image": curr_imgs}]
                            +
                            [{"type": "text", "text": instruction_batch[index]}]
                        )
                    }
                ]
                text = self.undstand_model.vlm_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.undstand_model.vlm_processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                input_ids_list.append(inputs["input_ids"])          
                attention_mask_list.append(inputs["attention_mask"])
                pixel_values_list.append(inputs["pixel_values"])  
                image_grid_thw_list.append(inputs["image_grid_thw"]) 
        
        # 3. batch 之间补齐
        max_len = max(x.shape[1] for x in input_ids_list)
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(input_ids_list, attention_mask_list):
            pad_len = max_len - ids.shape[1]
            if pad_len > 0:
                pad_id = getattr(getattr(self.undstand_model.vlm_processor, "tokenizer", None), "pad_token_id", 0)
                ids_pad = torch.full((ids.shape[0], pad_len), pad_id, dtype=ids.dtype)
                mask_pad = torch.zeros((mask.shape[0], pad_len), dtype=mask.dtype)
                ids = torch.cat([ids, ids_pad], dim=1)
                mask = torch.cat([mask, mask_pad], dim=1)
            padded_ids.append(ids)
            padded_masks.append(mask)

        # 4. 构建输入batch
        input_ids_batch = torch.cat(padded_ids, dim=0).to(self.device)              # [B, L]
        attention_mask_batch = torch.cat(padded_masks, dim=0).to(self.device)       # [B, L]
        pixel_values_batch = torch.cat(pixel_values_list, dim=0).to(self.device)    # [B, ...]
        image_grid_thw_batch = torch.cat(image_grid_thw_list, dim=0).to(self.device)

        # 5. 构建transformer输入embedding
        inputs_embeds = self.undstand_model.vlm_model.get_input_embeddings()(input_ids_batch)
        image_embeds, deepstack_image_embeds = self.undstand_model.vlm_model.get_image_features(pixel_values_batch, image_grid_thw_batch)
        image_embeds = torch.cat(image_embeds, dim=0).to(self.device, self.dtype)

        image_mask, _ = self.undstand_model.vlm_model.model.get_placeholder_mask(
            input_ids_batch, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]

        # 6. 位置编码
        position_ids, _rope_deltas = self.undstand_model.vlm_model.model.get_rope_index(
            input_ids=input_ids_batch,
            image_grid_thw=image_grid_thw_batch,
            video_grid_thw=None, 
            attention_mask=attention_mask_batch
        )

        # 7. 构造最终输入
        vlm_kwargs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask_batch,
            'position_ids': position_ids,
            'visual_pos_masks': visual_pos_masks,
            'deepstack_visual_embeds': deepstack_image_embeds,
            'past_key_values': None,
            'use_cache': False,
            'output_attentions': False,
            'output_hidden_states': True,
            'return_dict': True
        }

        # 8. 输入
        with torch.no_grad():
            vlm_output = self.undstand_model.vlm_model.model.language_model(**vlm_kwargs)
        
        # 9. 提取最后一层的输出
        last_layer_features = vlm_output.hidden_states[-1]

        # 10. 映射 为 token
        tokens = self.undstand_model.vlm_projector(last_layer_features)

        return tokens
