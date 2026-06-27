import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from core.config import *  # noqa: F401,F403  (DATASET_PATH / pred_horizon / obs_horizon 等)
# 7D(axis-angle) <-> 10D(rotation_6d) 旋转表示转换
from core.myrotation_transformer_final import RotationTransformer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from tqdm import tqdm
class MinMaxNormalizer:
    def __init__(self, data=None, min_val=None, max_val=None):
        """
        可以传入 data 自动计算 min/max，也可以直接传入已知的 min/max
        """
        if data is not None:
            # 假设 data 是 (N, Dim)
            self.min_val = np.min(data, axis=0)
            self.max_val = np.max(data, axis=0)
        else:
            self.min_val = np.array(min_val)
            self.max_val = np.array(max_val)
            
        # 防止除以零（如果某维度没有变化）
        self.scale = self.max_val - self.min_val
        self.scale[self.scale == 0] = 1.0 

    def normalize(self, x):
        # 归一化到 [0, 1]
        norm = (x - self.min_val) / self.scale
        # 映射到 [-1, 1]
        return norm * 2 - 1

    def denormalize(self, x):
        # 从 [-1, 1] 映射回 [0, 1]
        denorm = (x + 1) / 2
        # 还原到原始范围
        return denorm * self.scale + self.min_val
#----------------------------------------------------------
class RobomimicDataset(Dataset):
    def __init__(self, 
                 dataset_path, 
                 obs_keys, 
                 pred_horizon, 
                 obs_horizon, 
                 action_horizon):
        
        # 参数记录
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.obs_keys = obs_keys
        
        # --- 新增：初始化旋转变换器 ---
        # 用于将 Axis-Angle (3) 转为 Rotation 6D (6)
        self.rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep='rotation_6d')

        # 1. 读取数据到内存
        print(f"正在加载数据集: {dataset_path}")
        with h5py.File(dataset_path, 'r') as f:
            demos = f['data']
            self.all_obs = []
            self.all_actions = [] # 这里将存储转换后的 10维动作
            
            for key in tqdm(demos.keys(), desc="Loading"):
                demo = demos[key]
                
                # --- 处理 Observation (保持不变) ---
                obs_list = []
                for k in self.obs_keys:
                    obs_data = demo['obs'][k][:] 
                    obs_list.append(obs_data)
                obs_seq = np.concatenate(obs_list, axis=-1)
                self.all_obs.append(obs_seq)
                
                # --- 修改重点：处理 Action (7维 -> 10维) ---
                raw_actions = demo['actions'][:].astype(np.float32) # (T, 7)
                
                # 1. 拆分动作
                pos = raw_actions[:, :3]     # (T, 3)
                rot = raw_actions[:, 3:6]    # (T, 3) Axis-Angle
                gripper = raw_actions[:, 6:] # (T, 1)
                
                # 2. 旋转变换 (Axis-Angle -> Rotation 6D)
                # Transformer 接收 Tensor，返回 Tensor，需转回 Numpy
                rot_tensor = torch.from_numpy(rot)
                rot_6d_tensor = self.rotation_transformer.forward(rot_tensor)
                rot_6d = rot_6d_tensor.numpy() # (T, 6)
                
                # 3. 拼接成绝对位置控制所需的格式 (T, 10)
                # 3 pos + 6 rot + 1 gripper = 10 dim
                new_actions = np.concatenate([pos, rot_6d, gripper], axis=-1)
                
                self.all_actions.append(new_actions)

        # 2. 计算统计数据并初始化归一化器
        # 把所有数据拼在一起
        all_obs_concat = np.concatenate(self.all_obs, axis=0)
        all_action_concat = np.concatenate(self.all_actions, axis=0)
        
        # --- 修改重点：使用类 MinMaxNormalizer ---
        # 此时 all_action_concat 已经是 10维的数据了
        self.obs_normalizer = MinMaxNormalizer(data=all_obs_concat)
        self.action_normalizer = MinMaxNormalizer(data=all_action_concat)
        
        # 打印一下验证维度
        print(f"原始 Action 维度: 7, 转换后 Action 维度: {all_action_concat.shape[-1]}")
        
        # 3. 预处理：归一化所有数据并建立索引
        self.normalized_obs = []
        self.normalized_actions = []
        self.indices = []
        
        for i in range(len(self.all_obs)):
            # --- 修改重点：调用实例方法进行归一化 ---
            n_obs = self.obs_normalizer.normalize(self.all_obs[i])
            n_action = self.action_normalizer.normalize(self.all_actions[i])
            
            self.normalized_obs.append(n_obs)
            self.normalized_actions.append(n_action)
            
            # 建立索引 (逻辑不变)
            episode_len = n_obs.shape[0]
            # 为了保证剩下的长度够 pred_horizon，需要减去它
            for start_ts in range(episode_len - self.pred_horizon):
                self.indices.append((i, start_ts))
                
        print(f"加载完成! 样本数量: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        episode_idx, start_ts = self.indices[idx]
        
        n_obs = self.normalized_obs[episode_idx]
        n_action = self.normalized_actions[episode_idx]
        
        # 1. 取 Observation
        # 注意：这里如果 obs_horizon > 1，需要处理边界情况(padding)，
        # 但为简化起见，假设 start_ts + obs_horizon 不会越界(因为上面 range 只减了 pred_horizon)
        # 实际严谨代码通常会在上面 range 减去 max(obs_horizon, pred_horizon)
        obs_seq = n_obs[start_ts : start_ts + self.obs_horizon, :]
        
        # 2. 取 Action
        action_seq = n_action[start_ts : start_ts + self.pred_horizon, :]
        
        data = {
            'obs': torch.from_numpy(obs_seq).float(),      # shape: (obs_horizon, obs_dim)
            'action': torch.from_numpy(action_seq).float() # shape: (pred_horizon, 10)
        }
        return data

    # --- 新增：提供一个反归一化+逆变换的辅助函数，供推理时使用 ---
    def get_unnormalized_action(self, n_action_pred):
        """
        推理时使用：
        输入: 模型预测的归一化动作 (B, 10) Tensor or Numpy
        输出: 环境可执行的动作 (B, 7) Numpy
        """
        is_tensor = isinstance(n_action_pred, torch.Tensor)
        if is_tensor:
            n_action_pred = n_action_pred.detach().cpu().numpy()

        # --- 数值安全：SI/ODE 积分可能让 latent 漂出 [-1, 1] 或出现 NaN/Inf ---
        n_action_pred = np.asarray(n_action_pred, dtype=np.float32)
        n_action_pred = np.nan_to_num(n_action_pred, nan=0.0, posinf=0.0, neginf=0.0)
        n_action_pred = np.clip(n_action_pred, -1.0, 1.0)
         
        # 1. 反归一化 (10D -> 10D)
        action_10d = self.action_normalizer.denormalize(n_action_pred)
        
        # 2. 拆解
        pos = action_10d[..., :3]
        rot_6d = action_10d[..., 3:9]
        gripper = action_10d[..., 9:]
        
        # 3. 逆旋转变换 (6D -> 3D Axis-Angle)
        rot_6d_tensor = torch.from_numpy(rot_6d)
        rot_axis = self.rotation_transformer.inverse(rot_6d_tensor).numpy()
        
        # 4. 拼接 (10D -> 7D)
        action_7d = np.concatenate([pos, rot_axis, gripper], axis=-1)
        return action_7d
#----------------------------------------------------------

pred_horizon = 16    # 预测未来 16 步
obs_horizon = 2      # 看过去 2 步
action_horizon = 8   # (这一步在 dataset 里用不到，但在推理时有用)

# 1. 实例化 Dataset
dataset = RobomimicDataset(
    dataset_path=DATASET_PATH, # 来自 core.config，按 TASK 自动派生
    obs_keys=[
        'object', 
        'robot0_eef_pos', 
        'robot0_eef_quat', 
        'robot0_gripper_qpos'
    ],
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)

# 2. 实例化 DataLoader
dataloader = DataLoader(
    dataset, 
    batch_size=64, 
    shuffle=True, 
    num_workers=0, # 本地调试设为 0 比较稳妥，避免多进程报错
    pin_memory=True
)