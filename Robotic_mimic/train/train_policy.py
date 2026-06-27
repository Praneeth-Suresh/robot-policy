import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from core.config import *  # noqa: F401,F403  (device / TASK / obs_horizon / pred_horizon, etc.)
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


sigma0 = 0.4  # SFP initial tube radius
k = 10.0      # SFP stability coefficient


print("Preparing Streaming SI Policy (training block is disabled; load checkpoint for inference)...")

import random
import time

save_dir = _MODEL_ROOT  # = checkpoint/; training save/resume/inference load all go here
os.makedirs(save_dir, exist_ok=True)
save_every_epochs = 200

# To resume training, set path, e.g.:
# resume_from = os.path.join(save_dir, f"{TASK}_state_latest_si_abs.ckpt")
resume_from = None
# Resume training: point to latest checkpoint (e.g. fixed at epoch 800)
# resume_from = os.path.join(save_dir, f"{TASK}_state_800_ep_si_abs.ckpt")
# if (resume_from is not None) and (not os.path.isfile(resume_from)):
#     raise FileNotFoundError(f"[resume] checkpoint not found: {resume_from}")

global_step = 0
train_history = []  # per-epoch summary (loss curves / resume training)

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
        # --- Inference-compatible keys (used by load_pretrained below) ---
        "velocity_net": si_velocity_net.state_dict(),
        "denoiser_net": si_denoiser_net.state_dict(),

        # --- Training resume metadata ---
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
    PyTorch 2.6+ defaults weights_only=True, which breaks ckpts with optimizer / rng_state.
    Explicitly use weights_only=False; fall back for older torch without that argument.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # older torch does not support weights_only
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

    # diffusers.EMAModel supports load_state_dict
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
        # Per-epoch average component losses
        epoch_v_loss = []
        epoch_eta_loss = []
        
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # 1. Data preparation
                nobs = nbatch['obs'].to(device)
                naction = nbatch['action'].to(device)
                
                # Current and future action segment
                ξ = naction[:, obs_horizon-1:, :] 
                B = ξ.shape[0]
                
                # Sample time t ~ U[0, 1]
                t = torch.rand(B, device=device)
                
                # 2. Ground-truth position and velocity (linear interpolation)
                ξt, dξdt = LinearlyInterpolateTrajectory(ξ, t)
                
                # 3. Flow Matching (FP) base target (tube distribution)
                # a_t_fp = ξ(t) + σ(t) * ε1
                # σ(t) = σ0 * exp(-kt)
                t_expanded = t.view(B, 1)
                sigma_t_fp = sigma0 * torch.exp(-k * t_expanded)
                noise_fp = torch.randn_like(ξt)
                a_t_fp = ξt + sigma_t_fp * noise_fp
                
                # Velocity target (vector field toward trajectory)
                # v_target = dξ/dt - k * (a_t_fp - ξ(t))
                # Note: velocity is defined on a_t_fp
                v_target = dξdt - k * (a_t_fp - ξt)
                
                # 4. Stochastic Interpolant (SI) perturbation
                # x_t = a_t_fp + γ(t) * z
                # SI bubble outside the SFP tube
                gamma = gamma_t_si(t_expanded)
                z_noise_si = torch.randn_like(ξt)
                x_t_in_s = a_t_fp + gamma * z_noise_si
                
                # 5. Network forward pass
                # Reshape input to (B, 1, A) for UNet interface
                net_input = x_t_in_s.unsqueeze(1)
                global_cond = nobs.flatten(start_dim=1)
                
                # Predict velocity
                v_pred = si_velocity_net(sample=net_input, timestep=t, global_cond=global_cond)
                v_pred = v_pred.squeeze(1)
                
                # Predict noise (score-related quantity)
                eta_pred = si_denoiser_net(sample=net_input, timestep=t, global_cond=global_cond)
                eta_pred = eta_pred.squeeze(1)
                
                # 6. Compute loss
                loss_v = nn.functional.mse_loss(v_pred, v_target)
                loss_eta = nn.functional.mse_loss(eta_pred, z_noise_si)
                
                loss = loss_v + loss_eta
                
                # 7. Backpropagation
                loss.backward()
                optimizer_si.step()
                optimizer_si.zero_grad()
                lr_scheduler_si.step()
                
                # Update EMA
                ema_si_v.step(si_velocity_net.parameters())
                ema_si_eta.step(si_denoiser_net.parameters())
                
                # Logging
                loss_val = loss.item()
                v_loss_val = loss_v.item()
                eta_loss_val = loss_eta.item()
                
                epoch_loss.append(loss_val)
                epoch_v_loss.append(v_loss_val)
                epoch_eta_loss.append(eta_loss_val)

                global_step += 1
                
                # Show score loss (s_loss) in progress bar
                tepoch.set_postfix(loss=loss_val, v_loss=v_loss_val, s_loss=eta_loss_val, step=global_step)
                
        # Epoch-end average component losses
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

        # Save resumable checkpoint (also write latest for resume)
        if (epoch_idx + 1) % int(save_every_epochs) == 0:
            ckpt_path_epoch = os.path.join(save_dir, f"{TASK}_state_{epoch_idx+1}_ep_si_abs.ckpt")
            ckpt_path_latest = os.path.join(save_dir, f"{TASK}_state_latest_si_abs.ckpt")
            save_checkpoint(ckpt_path_epoch, epoch_idx=epoch_idx, global_step=global_step, history=train_history)
            save_checkpoint(ckpt_path_latest, epoch_idx=epoch_idx, global_step=global_step, history=train_history)
            print(f" Saved resumable checkpoint to {ckpt_path_epoch} (and latest)")

# Save final model
ckpt_path_si = os.path.join(save_dir, f"{TASK}_state_{num_epochs_si}_ep_si_abs.ckpt")
ckpt_path_latest = os.path.join(save_dir, f"{TASK}_state_latest_si_abs.ckpt")
save_checkpoint(ckpt_path_si, epoch_idx=num_epochs_si - 1, global_step=global_step, history=train_history)
save_checkpoint(ckpt_path_latest, epoch_idx=num_epochs_si - 1, global_step=global_step, history=train_history)
print(f"Saved Streaming SI checkpoint to {ckpt_path_si} (and {ckpt_path_latest})")

# Prepare EMA models for inference
ema_si_velocity_net = si_velocity_net
ema_si_denoiser_net = si_denoiser_net
ema_si_v.copy_to(ema_si_velocity_net.parameters())
ema_si_eta.copy_to(ema_si_denoiser_net.parameters())
