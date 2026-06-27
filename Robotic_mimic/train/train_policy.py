import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from core.config import *  # noqa: F401,F403  (device / TASK / obs_horizon / pred_horizon 等)
from core.config import _MODEL_ROOT
from core.dataset import dataset, dataloader
from core.flow_policy import (
    si_velocity_net,
    si_denoiser_net,
    optimizer_si,
    lr_scheduler_si,
    ema_si_v,
    ema_si_eta,
    num_epochs_si,
    LinearlyInterpolateTrajectory,
    gamma_t_si,
    d_gamma_dt_si,
)


sigma0 = 0.4  # SFP 初始管状半径
k = 10.0      # SFP 稳定系数


print("Preparing Streaming SI Policy (training block is disabled; load checkpoint for inference)...")

import random
import time

save_dir = _MODEL_ROOT  # = checkpoint/，训练保存/续训/推理加载都落在这里
os.makedirs(save_dir, exist_ok=True)
save_every_epochs = 200

# 想续训就填路径，例如：
# resume_from = os.path.join(save_dir, f"{TASK}_state_latest_si_abs.ckpt")
resume_from = None
# 恢复训练：指定最近的 checkpoint（例如固定为 800 epoch）
# resume_from = os.path.join(save_dir, f"{TASK}_state_800_ep_si_abs.ckpt")
# if (resume_from is not None) and (not os.path.isfile(resume_from)):
#     raise FileNotFoundError(f"[resume] checkpoint not found: {resume_from}")

global_step = 0
train_history = []  # 每个 epoch 记录摘要（loss 曲线/恢复训练用）

def _pack_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["torch_cuda_all"] = None
    return state

def _restore_rng_state(state):
    if not isinstance(state, dict):
        return
    try:
        if state.get("python", None) is not None:
            random.setstate(state["python"])
        if state.get("numpy", None) is not None:
            np.random.set_state(state["numpy"])
        if state.get("torch", None) is not None:
            torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("torch_cuda_all", None) is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    except Exception as e:
        print(f"[warn] failed to restore rng state: {e}")

def save_checkpoint(path, epoch_idx, global_step, history):
    ckpt = {
        # --- 推理兼容键（下面的 load_pretrained 会用到） ---
        "velocity_net": si_velocity_net.state_dict(),
        "denoiser_net": si_denoiser_net.state_dict(),

        # --- 训练恢复所需信息 ---
        "epoch": int(epoch_idx),
        "global_step": int(global_step),
        "optimizer_si": optimizer_si.state_dict(),
        "lr_scheduler_si": lr_scheduler_si.state_dict(),
        "ema_si_v": getattr(ema_si_v, "state_dict", lambda: None)(),
        "ema_si_eta": getattr(ema_si_eta, "state_dict", lambda: None)(),
        "history": history,
        "sigma0": float(sigma0),
        "k": float(k),
        "num_epochs_si": int(num_epochs_si),
        "saved_at": float(time.time()),
        "rng_state": _pack_rng_state(),
    }
    torch.save(ckpt, path)

def _torch_load_compat(path, map_location):
    """
    PyTorch 2.6+ 默认 weights_only=True，会导致包含 optimizer / rng_state 等对象的 ckpt 无法加载。
    这里显式使用 weights_only=False，并对旧版本 torch 做参数兼容。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # older torch 不支持 weights_only 参数
        return torch.load(path, map_location=map_location)

def load_checkpoint(path):
    ckpt = _torch_load_compat(path, map_location=device)
    si_velocity_net.load_state_dict(ckpt["velocity_net"])
    si_denoiser_net.load_state_dict(ckpt["denoiser_net"])

    if "optimizer_si" in ckpt:
        optimizer_si.load_state_dict(ckpt["optimizer_si"])
    if "lr_scheduler_si" in ckpt:
        try:
            lr_scheduler_si.load_state_dict(ckpt["lr_scheduler_si"])
        except Exception as e:
            print(f"[warn] failed to load lr_scheduler state, continue with fresh scheduler. err={e}")

    # diffusers.EMAModel 支持 load_state_dict
    if ckpt.get("ema_si_v", None) is not None:
        try:
            ema_si_v.load_state_dict(ckpt["ema_si_v"])
        except Exception as e:
            print(f"[warn] failed to load ema_si_v state. err={e}")
    if ckpt.get("ema_si_eta", None) is not None:
        try:
            ema_si_eta.load_state_dict(ckpt["ema_si_eta"])
        except Exception as e:
            print(f"[warn] failed to load ema_si_eta state. err={e}")

    if "rng_state" in ckpt:
        _restore_rng_state(ckpt["rng_state"])

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    step = int(ckpt.get("global_step", 0))
    history = ckpt.get("history", [])
    return start_epoch, step, history

start_epoch = 0
if (resume_from is not None) and os.path.isfile(resume_from):
    print(f"[resume] loading checkpoint: {resume_from}")
    start_epoch, global_step, train_history = load_checkpoint(resume_from)
    print(f"[resume] start_epoch={start_epoch} global_step={global_step} history_len={len(train_history)}")

with tqdm(range(start_epoch, num_epochs_si), desc='Epoch', total=num_epochs_si, initial=start_epoch) as tglobal:
    for epoch_idx in tglobal:
        epoch_loss = []
        # 用于记录每个 epoch 的平均分项 loss
        epoch_v_loss = []
        epoch_eta_loss = []
        
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # 1. 数据准备
                nobs = nbatch['obs'].to(device)
                naction = nbatch['action'].to(device)
                
                # 提取当前及未来的动作片段
                ξ = naction[:, obs_horizon-1:, :] 
                B = ξ.shape[0]
                
                # 随机采样时间 t ~ U[0, 1]
                t = torch.rand(B, device=device)
                
                # 2. 计算 Ground Truth 位置和速度 (Linearly Interpolate)
                ξt, dξdt = LinearlyInterpolateTrajectory(ξ, t)
                
                # 3. 构建 Flow Matching (FP) 基础目标 (管状分布)
                # a_t_fp = ξ(t) + σ(t) * ε1
                # 这里的 σ(t) = σ0 * exp(-kt)
                t_expanded = t.view(B, 1)
                sigma_t_fp = sigma0 * torch.exp(-k * t_expanded)
                noise_fp = torch.randn_like(ξt)
                a_t_fp = ξt + sigma_t_fp * noise_fp
                
                # 计算 Velocity Target (指向轨迹的向量场)
                # v_target = dξ/dt - k * (a_t_fp - ξ(t))
                # 注意：这个速度是定义在 a_t_fp 上的
                v_target = dξdt - k * (a_t_fp - ξt)
                
                # 4. 构建 Stochastic Interpolant (SI) 扰动
                # x_t = a_t_fp + γ(t) * z
                # 我们在 SFP 的管子外面再套一层 SI 的气泡
                gamma = gamma_t_si(t_expanded)
                z_noise_si = torch.randn_like(ξt)
                x_t_in_s = a_t_fp + gamma * z_noise_si
                
                # 5. 网络前向传播
                # 输入需要 reshape 成 (B, 1, A) 以适应 UNet 接口
                net_input = x_t_in_s.unsqueeze(1)
                global_cond = nobs.flatten(start_dim=1)
                
                # 预测速度
                v_pred = si_velocity_net(sample=net_input, timestep=t, global_cond=global_cond)
                v_pred = v_pred.squeeze(1)
                
                # 预测噪声 (Score 相关的量)
                eta_pred = si_denoiser_net(sample=net_input, timestep=t, global_cond=global_cond)
                eta_pred = eta_pred.squeeze(1)
                
                # 6. 计算 Loss
                loss_v = nn.functional.mse_loss(v_pred, v_target)
                loss_eta = nn.functional.mse_loss(eta_pred, z_noise_si)
                
                loss = loss_v + loss_eta
                
                # 7. 反向传播
                loss.backward()
                optimizer_si.step()
                optimizer_si.zero_grad()
                lr_scheduler_si.step()
                
                # 更新 EMA
                ema_si_v.step(si_velocity_net.parameters())
                ema_si_eta.step(si_denoiser_net.parameters())
                
                # 记录数据
                loss_val = loss.item()
                v_loss_val = loss_v.item()
                eta_loss_val = loss_eta.item()
                
                epoch_loss.append(loss_val)
                epoch_v_loss.append(v_loss_val)
                epoch_eta_loss.append(eta_loss_val)

                global_step += 1
                
                # [修改点] 这里增加了 s_loss (score loss) 的显示
                tepoch.set_postfix(loss=loss_val, v_loss=v_loss_val, s_loss=eta_loss_val, step=global_step)
                
        # [修改点] Epoch 结束时也可以看平均的分项 Loss
        ep_loss = float(np.mean(epoch_loss)) if len(epoch_loss) else 0.0
        ep_v = float(np.mean(epoch_v_loss)) if len(epoch_v_loss) else 0.0
        ep_s = float(np.mean(epoch_eta_loss)) if len(epoch_eta_loss) else 0.0
        tglobal.set_postfix(
            loss=ep_loss,
            v_loss=ep_v,
            s_loss=ep_s
        )

        train_history.append({
            "epoch": int(epoch_idx),
            "global_step": int(global_step),
            "loss": ep_loss,
            "v_loss": ep_v,
            "s_loss": ep_s,
        })

        # [修改] 保存可续训 checkpoint（同时写一份 latest 方便 resume）
        if (epoch_idx + 1) % int(save_every_epochs) == 0:
            ckpt_path_epoch = os.path.join(save_dir, f"{TASK}_state_{epoch_idx+1}_ep_si_abs.ckpt")
            ckpt_path_latest = os.path.join(save_dir, f"{TASK}_state_latest_si_abs.ckpt")
            save_checkpoint(ckpt_path_epoch, epoch_idx=epoch_idx, global_step=global_step, history=train_history)
            save_checkpoint(ckpt_path_latest, epoch_idx=epoch_idx, global_step=global_step, history=train_history)
            print(f" Saved resumable checkpoint to {ckpt_path_epoch} (and latest)")

# 保存模型
ckpt_path_si = os.path.join(save_dir, f"{TASK}_state_{num_epochs_si}_ep_si_abs.ckpt")
ckpt_path_latest = os.path.join(save_dir, f"{TASK}_state_latest_si_abs.ckpt")
save_checkpoint(ckpt_path_si, epoch_idx=num_epochs_si - 1, global_step=global_step, history=train_history)
save_checkpoint(ckpt_path_latest, epoch_idx=num_epochs_si - 1, global_step=global_step, history=train_history)
print(f"Saved Streaming SI checkpoint to {ckpt_path_si} (and {ckpt_path_latest})")

# 准备推理用的 EMA 模型
ema_si_velocity_net = si_velocity_net
ema_si_denoiser_net = si_denoiser_net
ema_si_v.copy_to(ema_si_velocity_net.parameters())
ema_si_eta.copy_to(ema_si_denoiser_net.parameters())
