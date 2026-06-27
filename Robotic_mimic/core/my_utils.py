import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- 模块化补全：critic 的 .pth 统一放在 checkpoint/（_MODEL_ROOT 来自 core.config）---
from core.config import _MODEL_ROOT

# 检测设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from typing import Dict, Optional

def _infer_task_name_from_dataset_path(dataset_path: str) -> str:
    p = (dataset_path or "").lower()
    for name in ["lift", "can", "square"]:
        if f"/{name}/" in p:
            return name
    return "unknown"

def _get_obs_key_dims(wrapper) -> Dict[str, int]:
    key_dims: Dict[str, int] = {}
    raw = wrapper.env.get_observation()
    for k in wrapper.obs_keys:
        key_dims[k] = int(np.asarray(raw[k]).reshape(-1).shape[0])
    return key_dims

def _get_eef_pos_start(wrapper, target_key: str = "robot0_eef_pos") -> int:
    if target_key not in wrapper.obs_keys:
        raise ValueError(f"wrapper.obs_keys 里没有 {target_key}，当前 keys={wrapper.obs_keys}")
    key_dims = _get_obs_key_dims(wrapper)
    idx = wrapper.obs_keys.index(target_key)
    return int(sum(key_dims[k] for k in wrapper.obs_keys[:idx]))

def _get_eef_pos_from_obs(obs: np.ndarray, eef_pos_start: int) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    return obs[eef_pos_start : eef_pos_start + 3].astype(np.float32)

def _raw_to_norm(raw: np.ndarray, min_v: np.ndarray, max_v: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    min_v = np.asarray(min_v, dtype=np.float32)
    max_v = np.asarray(max_v, dtype=np.float32)
    denom = (max_v - min_v + 1e-6)
    norm01 = (raw - min_v) / denom
    norm01 = np.clip(norm01, 0.0, 1.0)
    return norm01 * 2.0 - 1.0

def _init_na_from_obs(
    obs: np.ndarray,
    dataset,
    eef_pos_start: int,
    control_delta: Optional[bool],
    device: torch.device,
) -> torch.Tensor:
    """
    初始化 latent action na: (1, 1, action_dim)
    - absolute: 用当前 eef_pos 作为初始 pos
    - delta: 用 0 增量作为初始 pos
    - 旋转/夹爪使用数值稳定初始化（避免 6D 退化导致 NaN）
    """
    action_dim_local = int(dataset[0]["action"].shape[-1])
    na = torch.zeros((1, 1, action_dim_local), device=device, dtype=torch.float32)

    is_delta = bool(control_delta) if control_delta is not None else True
    curr_eef_pos = _get_eef_pos_from_obs(obs, eef_pos_start)  # 物理坐标
    raw_pos = np.zeros(3, dtype=np.float32) if is_delta else curr_eef_pos

    pos_min = dataset.action_normalizer.min_val[:3].astype(np.float32)
    pos_max = dataset.action_normalizer.max_val[:3].astype(np.float32)
    na[0, 0, :3] = torch.from_numpy(_raw_to_norm(raw_pos, pos_min, pos_max)).to(device)

    # gripper: 默认第 9 维是夹爪；用 “张开”(-1) 做稳定起点
    if action_dim_local >= 10:
        na[0, 0, 9] = -1.0

    # 6D rotation: 用单位旋转的稳定 6D 表示，并映射到归一化空间
    if action_dim_local >= 9:
        rot6d_raw = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        rot_min = dataset.action_normalizer.min_val[3:9].astype(np.float32)
        rot_max = dataset.action_normalizer.max_val[3:9].astype(np.float32)
        rot6d_raw = np.clip(rot6d_raw, rot_min, rot_max)
        na[0, 0, 3:9] = torch.from_numpy(_raw_to_norm(rot6d_raw, rot_min, rot_max)).to(device)

    na = torch.nan_to_num(na, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
    return na

def _r_obs_norm_from_phys(dataset, r_phys: float) -> float:
    """
    物理空间半径 r_phys（米）粗略换算到归一化空间半径 r_norm。
    x_norm = (x-min)/(max-min)*2-1 => dx_norm ≈ 2*dx_phys/(max-min)
    用 xyz 的平均尺度做近似（足够用于风险打标阈值）。
    """
    pos_min = dataset.action_normalizer.min_val[:3].astype(np.float32)
    pos_max = dataset.action_normalizer.max_val[:3].astype(np.float32)
    scale = np.mean(np.maximum(pos_max - pos_min, 1e-6))
    return float(2.0 * r_phys / scale)

def _task_cfg(task_name: str) -> Dict[str, float]:
    cfg: Dict[str, float] = dict(
        collision_threshold_phys=0.02,
        warning_dist_phys=0.05,
        obstacle_offset_xy=0.02,
        obstacle_offset_z=0.0,
        obstacle_noise_std_norm=0.10,
        table_z=0.8,
    )
    if task_name == "lift":
        cfg.update(collision_threshold_phys=0.02, warning_dist_phys=0.05)
    elif task_name == "can":
        cfg.update(collision_threshold_phys=0.02, warning_dist_phys=0.05)
    elif task_name == "square":
        cfg.update(collision_threshold_phys=0.02, warning_dist_phys=0.05)

    cfg["task_name"] = task_name
    # critic 的 .pth 也统一放在 checkpoint/ 目录（load/save 一致）
    _critic_name = f"Robust_critic_{task_name}.pth" if task_name != "unknown" else "Robust_critic.pth"
    cfg["critic_save_path"] = os.path.join(_MODEL_ROOT, _critic_name)
    cfg["critic_checkpoint_path"] = cfg["critic_save_path"].replace(".pth", "_latest_checkpoint.pth")
    return cfg
