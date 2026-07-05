import os
import math
import collections

import numpy as np
import torch
from tqdm.auto import tqdm

from core.config import *  # noqa: F401,F403  (device / obs_horizon / action_horizon / pred_horizon, etc.)
from core.config import CKPT_PATH_SI
from core.robomimic_env import get_env
_env = get_env()
wrapper       = _env.wrapper
CONTROL_DELTA = _env.control_delta
from core.dataset import dataset
from core.flow_policy import (
    si_velocity_net,
    si_denoiser_net,
    gamma_t_si,
    d_gamma_dt_si,
)


def _torch_load_compat(path, map_location):
    """Same helper as train/train_policy.py: compatible with PyTorch 2.6+ weights_only default."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


load_pretrained = True
ckpt_path_si = CKPT_PATH_SI

# Check checkpoint exists
if load_pretrained and os.path.isfile(ckpt_path_si):
    print(f"Found pretrained checkpoint: {ckpt_path_si}")
    checkpoint = _torch_load_compat(ckpt_path_si, map_location=device)

    # 1. Load weights into base networks
    si_velocity_net.load_state_dict(checkpoint['velocity_net'])
    si_denoiser_net.load_state_dict(checkpoint['denoiser_net'])

    # 2. Set up inference models
    # When skipping training, use loaded weights directly as inference models
    # (EMA model interface simulation)
    ema_si_velocity_net = si_velocity_net
    ema_si_denoiser_net = si_denoiser_net

    print('Pretrained weights loaded for Streaming SI Policy. Ready for inference!')

else:
    print(f"Checkpoint {ckpt_path_si} not found. Please run the training cell below.")


import collections
import numpy as np
import torch
import math
from tqdm.auto import tqdm
from scipy.spatial.transform import Rotation as R

# =========================================================
# Helper: compute drift (10D action)
# =========================================================
def get_drift(x, t, global_cond, sigma_infer, eps=1e-6):
    """
    Compute corrected drift b(x, t)
    x: (B, 1, 10)
    """
    if t.ndim == 0: t = t.unsqueeze(0)
    
    # Predict v and eta
    v_pred = ema_si_velocity_net(sample=x, timestep=t, global_cond=global_cond)
    eta_pred = ema_si_denoiser_net(sample=x, timestep=t, global_cond=global_cond)
    
    # gamma and its derivative
    gamma = gamma_t_si(t).view(-1, 1, 1).to(device)
    gamma_dot = d_gamma_dt_si(t).view(-1, 1, 1).to(device)
    
    # Score and drift coefficient
    s_pred = -eta_pred / (gamma + eps)
    score_coeff = 0.5 * (sigma_infer ** 2) - (gamma * gamma_dot)
    
    b = v_pred + score_coeff * s_pred
    return b

# =========================================================
# SSIP inference (auto-adapts control_delta: delta / absolute)
# =========================================================
def run_si_inference(sigma_infer, title):
    # 1. Reset environment
    obs = wrapper.reset() 
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    rewards = []
    # Robust success counting:
    # - sparse-reward robosuite tasks (reward_shaping=False): reward>0 ≈ success
    # - also check success/task_success in info (key varies by version)
    success_steps = 0
    _printed_success_info = False
    
    # 2. Warm start: build initial flow state x_0 from current observation
    # -----------------------------------------------------------
    # [Key]: map current obs accurately to 10D latent action format
    # -----------------------------------------------------------
    
    # Hard-coded (legacy):
    # start_dim = 14 

    
    # 1. Dimension per obs key
    key_dims = []
    for key in wrapper.obs_keys:
        # Peek at one observation for shape
        dummy_obs = wrapper.env.get_observation()[key]
        dim = np.array(dummy_obs).flatten().shape[0]
        key_dims.append(dim)
    
    # 2. Index of 'robot0_eef_pos' in obs keys
    target_key = 'robot0_eef_pos'
    if target_key in wrapper.obs_keys:
        idx = wrapper.obs_keys.index(target_key)
        # start_dim = sum of lengths of preceding keys
        start_dim = sum(key_dims[:idx])
    else:
        raise ValueError(f"{target_key} not found in obs keys!")
        
    print(f"Auto-computed start dim: {start_dim}")
    # eef position in current observation (physical coordinates)
    curr_pos = obs[start_dim : start_dim+3].astype(np.float32) # (x, y, z)

    # Helper: map raw action to action_normalizer space [-1, 1]
    def _raw_to_norm(raw: np.ndarray, min_v: np.ndarray, max_v: np.ndarray) -> np.ndarray:
        raw = raw.astype(np.float32)
        min_v = min_v.astype(np.float32)
        max_v = max_v.astype(np.float32)
        denom = (max_v - min_v + 1e-6)
        norm01 = (raw - min_v) / denom
        norm01 = np.clip(norm01, 0.0, 1.0)
        return norm01 * 2.0 - 1.0

    is_delta = bool(CONTROL_DELTA) if CONTROL_DELTA is not None else True
    if is_delta:
        # delta control: initial latent action is zero increment
        raw_pos = np.zeros(3, dtype=np.float32)
    else:
        # absolute control: initial latent action uses current eef_pos
        raw_pos = curr_pos

    pos_min = dataset.action_normalizer.min_val[:3].astype(np.float32)
    pos_max = dataset.action_normalizer.max_val[:3].astype(np.float32)
    norm_pos = _raw_to_norm(raw_pos, pos_min, pos_max)
    
    # 3. Initialize na (latent state)
    # shape: (1, 1, 10)
    na = torch.zeros((1, 1, 10), device=device, dtype=torch.float32)
    
    # [Fix 1]: set position (zero increment or current eef_pos per control_delta)
    na[0, 0, :3] = torch.from_numpy(norm_pos).to(device)
    
    # [Fix 2]: initialize gripper to "open" (-1.0)
    # Assume dim 10 (index 9) is gripper.
    # In normalized space, -1 usually maps to raw minimum (open)
    na[0, 0, 9] = -1 
    # [Fix 3]: all-zero 6D rotation is degenerate -> NaN in RotationTransformer
    # For delta/absolute, zero/unit rotation maps to the same stable 6D representation
    rot6d_raw = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    rot_min = dataset.action_normalizer.min_val[3:9].astype(np.float32)
    rot_max = dataset.action_normalizer.max_val[3:9].astype(np.float32)
    rot6d_raw = np.clip(rot6d_raw, rot_min, rot_max)
    rot6d_norm = _raw_to_norm(rot6d_raw, rot_min, rot_max)
    na[0, 0, 3:9] = torch.from_numpy(rot6d_norm).to(device, dtype=torch.float32)

    # Numerical safety
    na = torch.nan_to_num(na, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)

    # Flow starting point
    na_from_prev_chunk = na
    
    # =========================================================
    
    done = False
    step_idx = 0
    max_steps = 400 
    dt_val = 1.0 / (pred_horizon - obs_horizon) # flow time step
    
    print(f"Running Inference: {title} (sigma={sigma_infer})")
    
    with torch.no_grad():
        with tqdm(total=max_steps, desc=title) as pbar:
            while not done and step_idx < max_steps:
                # --- A. Prepare observation ---
                obs_seq = np.stack(obs_deque)
                # [Fix]: use dataset.obs_normalizer
                nobs = dataset.obs_normalizer.normalize(obs_seq)
                o_test = torch.from_numpy(nobs).to(device, dtype=torch.float32).flatten().unsqueeze(0)
                
                # --- B. Streaming loop ---
                na = na_from_prev_chunk
                
                for i in range(action_horizon):
                    # 1. Decode current action and execute
                    # [Fix]: dataset.get_unnormalized_action handles denorm + 6D inverse
                    a_real_full = dataset.get_unnormalized_action(na) # returns (1, 1, 7) numpy
                    a_real = a_real_full.squeeze() # (7,)
                    
                    # Environment step
                    obs, reward, done, info = wrapper.step(a_real)

                    # Success check: reward>0 or info success flag
                    r = float(reward) if reward is not None else 0.0
                    info_success = False
                    if isinstance(info, dict):
                        info_success = bool(info.get("success", False) or info.get("task_success", False))
                    if (r > 0.0) or info_success:
                        success_steps += 1
                        if (not _printed_success_info) and isinstance(info, dict):
                            _printed_success_info = True
                            print(f"[success debug] reward={r} info_keys={list(info.keys())}")
                    
                    obs_deque.append(obs)
                    rewards.append(reward)
                    
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    
                    if done or step_idx >= max_steps: break
                    
                    # 2. Integration (Euler step) for next action
                    t_scalar = np.clip(i * dt_val, 1e-3, 1.0 - 1e-3)
                    t = torch.tensor([t_scalar], device=device, dtype=torch.float32)
                    
                    b_drift = get_drift(na, t, o_test, sigma_infer)
                    noise = torch.randn_like(na)
                    diffusion = sigma_infer * math.sqrt(dt_val) * noise
                    
                    na = na + b_drift * dt_val + diffusion
                    # Numerical safety: clamp latent to [-1, 1], clear NaN/Inf
                    na = torch.nan_to_num(na, nan=0.0, posinf=0.0, neginf=0.0)
                    na = na.clamp(-1.0, 1.0)
                
                # Update flow start (streaming)
                na_from_prev_chunk = na
                
                if done: break

    print(f"Inference done! Max reward: {max(rewards) if rewards else 0} | success_steps: {success_steps}")


if __name__ == "__main__":
    # 1. ODE mode (smooth)
    run_si_inference(sigma_infer=0.0, title="Streaming SI (ODE Mode)")

    # 2. SDE mode (exploration)
    run_si_inference(sigma_infer=0.02, title="Streaming SI (SDE Mode)")
