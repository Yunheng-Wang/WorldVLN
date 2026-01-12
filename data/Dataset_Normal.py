import torch.utils.data as data
import torch
import os
import random
import json
from decord import VideoReader, cpu
from data.utils.load import load_action, load_video_frames, load_video_num, load_instruction, load_instruction_embedding


class Dataset_Normal_Train(data.Dataset):
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
                    elif file_name == "instruction.pth":
                        files['instruction_embed'] = file_path
            episodes.append(files)
        return episodes
    

    def __len__(self):
        return len(self.all_episodes)


    def __getitem__(self, idx):
        # 1. 按顺序选择一个episode
        episode = self.all_episodes[idx]
        # 2. 获取当前帧 & 历史帧 & 未来帧 & 动作 下标
        ## 2.1 加载视频数量
        obser_fra_num = load_video_num(episode['observation'])
        ## 2.2 加载t时刻的观测下标
        if obser_fra_num  > self.predict_num:
            current_frames_idx = random.randint(0, obser_fra_num - self.predict_num - 1)
        else:
            new_idx = random.randint(0, len(self.all_episodes) - 1)
            return self.__getitem__(new_idx)
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
        #4. 补历史帧 保持bach的形状一致
        if len(history_frames_idx) < self.history_num:
            num_missing_frames = self.history_num - len(history_frames_idx)
            zero_frame = torch.zeros_like(cur_frames)  # 创建与现有帧相同形状的零帧
            zero_frames = zero_frame.unsqueeze(0).repeat(num_missing_frames, 1, 1, 1)
            his_frames = torch.cat([his_frames, zero_frames], dim=0)
        pred_frames = load_video_frames(episode['observation'], predict_frames_idx, self.image_size)
        actions = load_action(episode['action'], action_idx)
        # 4. 加载 instruction
        instruction = load_instruction(episode['instruction'])
        # 5. 加载 instruction embeding(t5)
        instruction_embed = load_instruction_embedding(episode["instruction_embed"], 512)
        
        if current_frames_idx == obser_fra_num - self.predict_num - 1:
            stop_label = 1
        else:
            stop_label = 0
        
        return {"cur_frame": cur_frames,
                "his_frames": his_frames,
                "pred_frames": pred_frames,
                "action": actions, 
                "instruction": instruction,
                "instruction_embed": instruction_embed,
                "stop_label": stop_label}




class Dataset_Normal_Val(data.Dataset):
    def __init__(self, dataset_root, predict_num, history_num, height, weight, config):
        self.dataset_root = dataset_root
        self.predict_num = predict_num
        self.history_num = history_num
        self.image_size = (height, weight)
        self.config = config
        self.batch = self._load_batch()


    def __len__(self):
        return len(self.batch)


    def _load_batch(self):
        # 加载任务数据
        with open(os.path.join(self.config.eval.data_root, self.config.eval.tasks_type, "val_unseen", "val_unseen.json"), 'r', encoding='utf-8') as file:
            task = json.load(file)
        random.seed(42)
        task['episodes'] = random.sample(task['episodes'], self.config.eval.num)
        grouped_tasks = {}
        for episode in task['episodes']:
            scene_id = episode['scene_id']
            parts = scene_id.split('/')
            scene_hash = parts[1] if len(parts) > 1 else parts[0].split('.')[0]  # 备用方案
            intruc_t5_embed_path = os.path.join(self.config.main.data_root, "val_unseen", "mp3d_" + self.config.eval.tasks_type + "_" + scene_hash + "_" + str(episode["episode_id"]), "instruction.pth")
            episode["instruction_embed"] = load_instruction_embedding(intruc_t5_embed_path, 512)
            if scene_hash not in grouped_tasks:
                grouped_tasks[scene_hash] = []
            grouped_tasks[scene_hash].append(episode)
        
        flattened_batches = []
        for scene_hash, episodes in grouped_tasks.items():
            for i in range(0, len(episodes), self.config.main.batch_size):
                batch_chunk = episodes[i : i + self.config.main.batch_size]
                if len(batch_chunk) == self.config.main.batch_size: 
                    flattened_batches.append(batch_chunk)
        return flattened_batches


    def __getitem__(self, idx):
        batch_data = self.batch[idx]
        scene = batch_data[0]["scene_id"]
        scene = scene.split('/')
        scene = scene[1] if len(scene) > 1 else scene[0].split('.')[0]

        output = []
        for i in range(self.config.main.batch_size):
            task = {
                'episode_id': batch_data[i]['episode_id'],
                'scene_id': scene,
                'instruction': batch_data[i]["instruction"]['instruction_text'],
                'instruction_embed': batch_data[i]["instruction_embed"],
                'start_position': torch.tensor(batch_data[i]["start_position"]),
                'start_rotation': torch.tensor(batch_data[i]["start_rotation"]),
                'goal_position': torch.tensor(batch_data[i]["goals"][0]["position"]),
                'reference_distant': torch.tensor(batch_data[i]["info"]["geodesic_distance"]),
                'reference_path': torch.tensor(batch_data[i]["reference_path"]),
            }
            output.append(task)
        return output
        
    
