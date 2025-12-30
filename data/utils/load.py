from decord import VideoReader, cpu
import torch
import numpy as np
from .image_utils import resize_with_padding

def load_video_num(video_path):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    return len(vr)


def load_video_frames(video_path, frame_indices, target_size):
    # 1. 加载帧
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    batch = vr.get_batch(frame_indices)
    frames_np = batch.asnumpy()
    # 2. 根据索引抽帧
    batch = vr.get_batch(frame_indices)
    frames_np = batch.asnumpy()
    # 3. 缩放帧
    th, tw = target_size
    _, h, w, _ = frames_np.shape
    if (h, w) != (th, tw):
        # Apply resize_with_padding to each frame (OpenCV implementation; no distortion)
        resized = [resize_with_padding(frames_np[i], target_size) for i in range(frames_np.shape[0])]
        frames_np = np.stack(resized, axis=0)
    # 4. 转换为张量 (归一化)
    video_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
    return video_tensor


def load_action(action_path, action_indices):
    action = np.load(action_path)
    action_tensor = torch.tensor(action)
    action_selected = action_tensor[action_indices]
    return action_selected


def load_instruction(instruction_path):
    with open(instruction_path, 'r') as file:
        instructions = file.read()
    return instructions
