import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from core.config import *  # noqa: F401,F403  (DATASET_PATH / pred_horizon / obs_horizon, etc.)
# 7D (axis-angle) <-> 10D (rotation_6d) rotation representation conversion
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
        Pass data to auto-compute min/max, or pass known min/max directly.
        """
        if data is not None:
            # Assume data is (N, Dim)
            self.min_val = np.min(data, axis=0)
            self.max_val = np.max(data, axis=0)
        else:
            self.min_val = np.array(min_val)
            self.max_val = np.array(max_val)
            
        # Avoid division by zero (constant dimensions)
        self.scale = self.max_val - self.min_val
        self.scale[self.scale == 0] = 1.0 

    def normalize(self, x):
        # Normalize to [0, 1]
        norm = (x - self.min_val) / self.scale
        # Map to [-1, 1]
        return norm * 2 - 1

    def denormalize(self, x):
        # Map from [-1, 1] back to [0, 1]
        denorm = (x + 1) / 2
        # Restore original range
        return denorm * self.scale + self.min_val
#----------------------------------------------------------
class RobomimicDataset(Dataset):
    def __init__(self, 
                 dataset_path, 
                 obs_keys, 
                 pred_horizon, 
                 obs_horizon, 
                 action_horizon):
        
        # Parameter bookkeeping
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.obs_keys = obs_keys
        
        # --- Initialize rotation transformer ---
        # Converts Axis-Angle (3) -> Rotation 6D (6)
        self.rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep='rotation_6d')

        # 1. Load data into memory
        print(f"Loading dataset: {dataset_path}")
        with h5py.File(dataset_path, 'r') as f:
            demos = f['data']
            self.all_obs = []
            self.all_actions = [] # stores converted 10D actions
            
            for key in tqdm(demos.keys(), desc="Loading"):
                demo = demos[key]
                
                # --- Process observation (unchanged) ---
                obs_list = []
                for k in self.obs_keys:
                    obs_data = demo['obs'][k][:] 
                    obs_list.append(obs_data)
                obs_seq = np.concatenate(obs_list, axis=-1)
                self.all_obs.append(obs_seq)
                
                # --- Key change: process action (7D -> 10D) ---
                raw_actions = demo['actions'][:].astype(np.float32) # (T, 7)
                
                # 1. Split action components
                pos = raw_actions[:, :3]     # (T, 3)
                rot = raw_actions[:, 3:6]    # (T, 3) Axis-Angle
                gripper = raw_actions[:, 6:] # (T, 1)
                
                # 2. Rotation transform (Axis-Angle -> Rotation 6D)
                # Transformer takes Tensor, returns Tensor; convert back to NumPy
                rot_tensor = torch.from_numpy(rot)
                rot_6d_tensor = self.rotation_transformer.forward(rot_tensor)
                rot_6d = rot_6d_tensor.numpy() # (T, 6)
                
                # 3. Concatenate for absolute-position control format (T, 10)
                # 3 pos + 6 rot + 1 gripper = 10 dim
                new_actions = np.concatenate([pos, rot_6d, gripper], axis=-1)
                
                self.all_actions.append(new_actions)

        # 2. Compute statistics and initialize normalizers
        # Concatenate all episodes
        all_obs_concat = np.concatenate(self.all_obs, axis=0)
        all_action_concat = np.concatenate(self.all_actions, axis=0)
        
        # --- Key change: use MinMaxNormalizer ---
        # all_action_concat is already 10D at this point
        self.obs_normalizer = MinMaxNormalizer(data=all_obs_concat)
        self.action_normalizer = MinMaxNormalizer(data=all_action_concat)
        
        # Verify dimensions
        print(f"Original action dim: 7, converted action dim: {all_action_concat.shape[-1]}")
        
        # 3. Preprocess: normalize all data and build index
        self.normalized_obs = []
        self.normalized_actions = []
        self.indices = []
        
        for i in range(len(self.all_obs)):
            # --- Key change: normalize via instance methods ---
            n_obs = self.obs_normalizer.normalize(self.all_obs[i])
            n_action = self.action_normalizer.normalize(self.all_actions[i])
            
            self.normalized_obs.append(n_obs)
            self.normalized_actions.append(n_action)
            
            # Build index (logic unchanged)
            episode_len = n_obs.shape[0]
            # Subtract pred_horizon so remaining length is sufficient
            for start_ts in range(episode_len - self.pred_horizon):
                self.indices.append((i, start_ts))
                
        print(f"Loading complete! Number of samples: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        episode_idx, start_ts = self.indices[idx]
        
        n_obs = self.normalized_obs[episode_idx]
        n_action = self.normalized_actions[episode_idx]
        
        # 1. Fetch observation
        # Note: if obs_horizon > 1, boundary padding may be needed;
        # for simplicity, assume start_ts + obs_horizon stays in bounds (range above only subtracts pred_horizon)
        # A rigorous version would subtract max(obs_horizon, pred_horizon) in the range above
        obs_seq = n_obs[start_ts : start_ts + self.obs_horizon, :]
        
        # 2. Fetch action
        action_seq = n_action[start_ts : start_ts + self.pred_horizon, :]
        
        data = {
            'obs': torch.from_numpy(obs_seq).float(),      # shape: (obs_horizon, obs_dim)
            'action': torch.from_numpy(action_seq).float() # shape: (pred_horizon, 10)
        }
        return data

    # --- Helper for inference: denormalize + inverse rotation transform ---
    def get_unnormalized_action(self, n_action_pred):
        """
        For inference:
        Input: model-predicted normalized action (B, 10) Tensor or NumPy
        Output: environment-executable action (B, 7) NumPy
        """
        is_tensor = isinstance(n_action_pred, torch.Tensor)
        if is_tensor:
            n_action_pred = n_action_pred.detach().cpu().numpy()

        # --- Numerical safety: SI/ODE integration may drift outside [-1, 1] or produce NaN/Inf ---
        n_action_pred = np.asarray(n_action_pred, dtype=np.float32)
        n_action_pred = np.nan_to_num(n_action_pred, nan=0.0, posinf=0.0, neginf=0.0)
        n_action_pred = np.clip(n_action_pred, -1.0, 1.0)
         
        # 1. Denormalize (10D -> 10D)
        action_10d = self.action_normalizer.denormalize(n_action_pred)
        
        # 2. Split components
        pos = action_10d[..., :3]
        rot_6d = action_10d[..., 3:9]
        gripper = action_10d[..., 9:]
        
        # 3. Inverse rotation (6D -> 3D axis-angle)
        rot_6d_tensor = torch.from_numpy(rot_6d)
        rot_axis = self.rotation_transformer.inverse(rot_6d_tensor).numpy()
        
        # 4. Concatenate (10D -> 7D)
        action_7d = np.concatenate([pos, rot_axis, gripper], axis=-1)
        return action_7d
#----------------------------------------------------------

pred_horizon = 16    # predict 16 future steps
obs_horizon = 2      # observe past 2 steps
action_horizon = 8   # unused in dataset; used at inference

# 1. Instantiate Dataset
dataset = RobomimicDataset(
    dataset_path=DATASET_PATH, # from core.config, derived from TASK
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

# 2. Instantiate DataLoader
dataloader = DataLoader(
    dataset, 
    batch_size=64, 
    shuffle=True, 
    num_workers=0, # 0 is safer for local debugging (avoids multiprocessing issues)
    pin_memory=True
)
