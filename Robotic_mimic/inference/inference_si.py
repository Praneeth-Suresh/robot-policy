import os
import math
import collections

import numpy as np
import torch
from tqdm.auto import tqdm

from core.config import *  # noqa: F401,F403  (device / obs_horizon / action_horizon / pred_horizon 等)
from core.config import CKPT_PATH_SI
from core.pusht_env import get_env
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
    """与 train/train_policy.py 中同名工具一致：兼容 PyTorch 2.6+ 的 weights_only 默认值。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


load_pretrained = True
ckpt_path_si = CKPT_PATH_SI

# 检查文件是否存在
if load_pretrained and os.path.isfile(ckpt_path_si):
    print(f"Found pretrained checkpoint: {ckpt_path_si}")
    checkpoint = _torch_load_compat(ckpt_path_si, map_location=device)

    # 1. 加载权重到基础网络
    si_velocity_net.load_state_dict(checkpoint['velocity_net'])
    si_denoiser_net.load_state_dict(checkpoint['denoiser_net'])

    # 2. 设置推理用的模型
    # 在跳过训练的情况下，我们将直接使用加载的权重作为推理模型
    # (模拟 EMA 模型及其接口)
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
# 辅助函数: 计算 Drift (适配 10维 Action)
# =========================================================
def get_drift(x, t, global_cond, sigma_infer, eps=1e-6):
    """
    计算修正后的漂移项 b(x, t)
    x: (B, 1, 10)
    """
    if t.ndim == 0: t = t.unsqueeze(0)
    
    # 预测 v 和 eta
    v_pred = ema_si_velocity_net(sample=x, timestep=t, global_cond=global_cond)
    eta_pred = ema_si_denoiser_net(sample=x, timestep=t, global_cond=global_cond)
    
    # 获取 gamma 及其导数
    gamma = gamma_t_si(t).view(-1, 1, 1).to(device)
    gamma_dot = d_gamma_dt_si(t).view(-1, 1, 1).to(device)
    
    # 计算 score 和 漂移系数
    s_pred = -eta_pred / (gamma + eps)
    score_coeff = 0.5 * (sigma_infer ** 2) - (gamma * gamma_dot)
    
    b = v_pred + score_coeff * s_pred
    return b

# =========================================================
# 核心修复: SSIP 推理函数（自动适配 control_delta：delta / absolute）
# =========================================================
def run_si_inference(sigma_infer, title):
    # 1. 重置环境
    obs = wrapper.reset() 
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    rewards = []
    # 兼容性更强的 success 统计：
    # - 对 robosuite 的稀疏奖励任务（reward_shaping=False），reward>0 基本等价于成功
    # - 同时保留 info 里的 success/task_success 检测（不同版本 key 不同）
    success_steps = 0
    _printed_success_info = False
    
    # 2. Warm Start: 基于当前观测构建初始流状态 x_0
    # -----------------------------------------------------------
    # [关键]: 必须准确地将当前 obs 转换为 10维的 latent action 格式
    # -----------------------------------------------------------
    
    # 硬编码
    # start_dim = 14 

    
    # 1. 获取各个 key 的维度
    key_dims = []
    for key in wrapper.obs_keys:
        # 临时取一个观测值来看看形状
        dummy_obs = wrapper.env.get_observation()[key]
        dim = np.array(dummy_obs).flatten().shape[0]
        key_dims.append(dim)
    
    # 2. 找到 'robot0_eef_pos' 是第几个 key
    target_key = 'robot0_eef_pos'
    if target_key in wrapper.obs_keys:
        idx = wrapper.obs_keys.index(target_key)
        # 前面所有 key 的长度之和就是 start_dim
        start_dim = sum(key_dims[:idx])
    else:
        raise ValueError(f"Obs keys 里面没有 {target_key}！")
        
    print(f"自动计算的 Start Dim: {start_dim}")
    # 当前观测里的 eef 位置（物理坐标）
    curr_pos = obs[start_dim : start_dim+3].astype(np.float32) # (x, y, z)

    # helper：把 raw action 映射到 action_normalizer 的归一化空间 [-1, 1]
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
        # delta 控制：初始 latent action 应该是 “0 增量”
        raw_pos = np.zeros(3, dtype=np.float32)
    else:
        # absolute 控制：初始 latent action 用当前 eef_pos 更合理
        raw_pos = curr_pos

    pos_min = dataset.action_normalizer.min_val[:3].astype(np.float32)
    pos_max = dataset.action_normalizer.max_val[:3].astype(np.float32)
    norm_pos = _raw_to_norm(raw_pos, pos_min, pos_max)
    
    # 3. 初始化 na (Latent State)
    # 维度: (1, 1, 10)
    na = torch.zeros((1, 1, 10), device=device, dtype=torch.float32)
    
    # [Fix 1]: 覆盖位置（根据 control_delta 自动选择 “0增量” 或 “当前eef_pos”）
    na[0, 0, :3] = torch.from_numpy(norm_pos).to(device)
    
    # [Fix 2 - 关键]: 强制初始化夹爪为 "张开" (-1.0)
    # 假设第 10 维 (索引 9) 是夹爪。
    # 在归一化空间中，-1 通常代表 raw action 的最小值 (Open)
    na[0, 0, 9] = -1 
    # [Fix 3 - 关键]: 6D Rotation 的全 0 是退化的，会在 RotationTransformer 中产生 NaN
    # 对于 delta/absolute 两种控制，“零旋转/单位旋转” 都应对应同一个稳定 6D 表示
    rot6d_raw = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    rot_min = dataset.action_normalizer.min_val[3:9].astype(np.float32)
    rot_max = dataset.action_normalizer.max_val[3:9].astype(np.float32)
    rot6d_raw = np.clip(rot6d_raw, rot_min, rot_max)
    rot6d_norm = _raw_to_norm(rot6d_raw, rot_min, rot_max)
    na[0, 0, 3:9] = torch.from_numpy(rot6d_norm).to(device, dtype=torch.float32)

    # 数值安全
    na = torch.nan_to_num(na, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)

    # 设置为流的起点
    na_from_prev_chunk = na
    
    # =========================================================
    
    done = False
    step_idx = 0
    max_steps = 400 
    dt_val = 1.0 / (pred_horizon - obs_horizon) # 流的时间步长
    
    print(f"Running Inference: {title} (sigma={sigma_infer})")
    
    with torch.no_grad():
        with tqdm(total=max_steps, desc=title) as pbar:
            while not done and step_idx < max_steps:
                # --- A. 准备 Observation ---
                obs_seq = np.stack(obs_deque)
                # [修复点]: 使用 dataset.obs_normalizer
                nobs = dataset.obs_normalizer.normalize(obs_seq)
                o_test = torch.from_numpy(nobs).to(device, dtype=torch.float32).flatten().unsqueeze(0)
                
                # --- B. Streaming Loop ---
                na = na_from_prev_chunk
                
                for i in range(action_horizon):
                    # 1. 解码当前动作并执行
                    # [修复点]: 使用 dataset.get_unnormalized_action 自动处理反归一化+6D转回
                    a_real_full = dataset.get_unnormalized_action(na) # 返回 (1, 1, 7) numpy
                    a_real = a_real_full.squeeze() # (7,)
                    
                    # 环境交互
                    obs, reward, done, info = wrapper.step(a_real)

                    # success 判定（强鲁棒）：reward>0 或 info success 标记
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
                    
                    # 2. 积分 (Euler Step) 计算下一个时间步的动作
                    t_scalar = np.clip(i * dt_val, 1e-3, 1.0 - 1e-3)
                    t = torch.tensor([t_scalar], device=device, dtype=torch.float32)
                    
                    b_drift = get_drift(na, t, o_test, sigma_infer)
                    noise = torch.randn_like(na)
                    diffusion = sigma_infer * math.sqrt(dt_val) * noise
                    
                    na = na + b_drift * dt_val + diffusion
                    # 数值安全：限制 latent 在 [-1, 1]，并清理 NaN/Inf
                    na = torch.nan_to_num(na, nan=0.0, posinf=0.0, neginf=0.0)
                    na = na.clamp(-1.0, 1.0)
                
                # 更新流的起点 (Streaming)
                na_from_prev_chunk = na
                
                if done: break

    print(f"推理结束! 最高得分: {max(rewards) if rewards else 0} | success_steps: {success_steps}")


if __name__ == "__main__":
    # 1. ODE 模式 (平滑)
    run_si_inference(sigma_infer=0.0, title="Streaming SI (ODE Mode)")

    # 2. SDE 模式 (探索)
    run_si_inference(sigma_infer=0.02, title="Streaming SI (SDE Mode)")
