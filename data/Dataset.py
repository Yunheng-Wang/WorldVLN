import torch.utils.data as data
import torch
import os
import random
from decord import VideoReader, cpu
from data.utils.load import load_action, load_video_frames, load_video_num, load_instruction


class Dataset(data.Dataset):
    def __init__(self, dataset_root, predict_num, history_num, height, weight):
        self.dataset_root = dataset_root
        self.predict_num = predict_num
        self.history_num = history_num
        self.image_size = (height, weight)
        self.all_episodes = self._load_episodes()


    def _load_episodes(self):
        episodes = []
        for folder_name in os.listdir(self.dataset_root):
            folder_path = os.path.join(self.dataset_root, folder_name)
            if os.path.isdir(folder_path):
                files = {}
                files["episode_name"] = folder_name
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if file_name == "instruction.txt":
                        files['instruction'] = file_path
                    elif file_name == "observation.mp4":
                        files['observation'] = file_path
                    elif file_name.endswith(('.npy')) and 'action' in file_name:
                        files['action'] = file_path
            episodes.append(files)
        return episodes
    

    def __len__(self):
        return len(self.all_episodes)


    def __getitem__(self, idx):
        while(True):
            try:
                # 1. 随机选择一个episode
                episode = random.choice(self.all_episodes)
                # 2. 获取当前帧 & 历史帧 & 未来帧 & 动作 下标
                ## 2.1 加载视频数量
                obser_fra_num = load_video_num(episode['observation'])
                ## 2.2 加载t时刻的观测下标
                current_frames_idx = random.randint(0, obser_fra_num - self.predict_num - 1)
                ## 2.3 加载历史时刻的观测下标
                available_history_idx = list(range(0, current_frames_idx)) 
                if len(available_history_idx) <= self.history_num:
                    history_frames_idx = available_history_idx
                else:
                    history_frames_idx = sorted(random.sample(available_history_idx, self.history_num))
                ## 2.4 加载未来时刻的观测下标
                predict_frames_idx = list(range(current_frames_idx + 1, current_frames_idx + self.predict_num + 1))
                ## 2.5 加载动作下标
                action_idx = list(range(current_frames_idx, current_frames_idx + self.predict_num))
                # 3. 从下标加载数据
                cur_frames = load_video_frames(episode['observation'], [current_frames_idx], self.image_size).squeeze(0)
                his_frames = load_video_frames(episode['observation'], history_frames_idx, self.image_size)
                ##* 若不足, 则补帧 保持bach的形状一致
                if len(history_frames_idx) < self.history_num:
                    num_missing_frames = self.history_num - len(history_frames_idx)
                    zero_frame = torch.zeros_like(his_frames[0])  # 创建与现有帧相同形状的零帧
                    zero_frames = zero_frame.unsqueeze(0).repeat(num_missing_frames, 1, 1, 1)
                    his_frames = torch.cat([zero_frames, his_frames], dim=0)
                pred_frames = load_video_frames(episode['observation'], predict_frames_idx, self.image_size)
                actions = load_action(episode['action'], action_idx)
                # 4. 加载 instruction
                instruction = load_instruction(episode['instruction'])

                if current_frames_idx == obser_fra_num - self.predict_num - 1:
                    stop_label = 1
                else:
                    stop_label = 0
                
                return {"cur_frame": cur_frames,
                        "his_frames": his_frames,
                        "pred_frames": pred_frames,
                        "action": actions, 
                        "instruction": instruction,
                        "stop_label": stop_label}
                    
            except Exception as e:
                continue