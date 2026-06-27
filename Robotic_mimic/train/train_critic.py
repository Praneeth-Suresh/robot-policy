import os
import math

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional

from core.config import *  # noqa: F401,F403  (device / obs_horizon / pred_horizon, etc.)
from core.pusht_env import get_env
_env = get_env()
wrapper      = _env.wrapper
dataset_path = _env.dataset_path
from core.dataset import dataset, dataloader
from core.flow_policy import si_velocity_net
from core.critic import CollisionPredictionCritic, compute_hybrid_risk_3d
from core.my_utils import (
    _infer_task_name_from_dataset_path,
    _task_cfg,
    _get_eef_pos_start,
    _r_obs_norm_from_phys,
)


def train_robust_critic(
    dataloader,
    si_velocity_net,
    action_dim,
    obs_dim,
    obs_horizon,
    pred_horizon,
    eef_pos_start: Optional[int] = None,
    epochs=150,
    K_samples=12,
    sample_t_max=0.8,
    r_obs_norm: float = 0.07,
    obstacle_noise_std_norm: float = 0.10,
    save_path="Robust_critic.pth",
):
    
    print(f"--- Initialization ---")
    checkpoint_path = save_path.replace(".pth", "_latest_checkpoint.pth")
    best_model_path = save_path.replace(".pth", "_best.pth")
    
    # Initialize Critic (cross-task obs_dim; geometric features from eef_pos in obs by default)
    critic = CollisionPredictionCritic(
        action_dim, obs_dim, obs_horizon,
        hidden_dim=1024, depth=6,
        eef_pos_start=eef_pos_start,
        use_obs_eef_pos=True,
    ).to(device)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    
    start_epoch = 0
    best_loss = float('inf')
    # Match inference-side dt_val (more stable, reusable across tasks)
    dt = 1.0 / max(1, (pred_horizon - obs_horizon))

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        critic.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f"Restarting at Epoch {start_epoch+1}")

    # Drift function using pretrained si_velocity_net
    def get_drift(a, t, cond):
        t_in = torch.clamp(t, 0.02, 0.98).view(-1)
        # Unsqueeze for UNet interface: (B, 1, 10)
        v = si_velocity_net(sample=a.unsqueeze(1), timestep=t_in, global_cond=cond).squeeze(1)
        return v

    for epoch in range(start_epoch, epochs):
        epoch_losses = []
        critic.train()
        
        with tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False) as pbar:
            for batch in pbar:
                obs = batch['obs'].to(device)
                gt_action = batch['action'].to(device) # (B, 16, 10)
                
                B = obs.shape[0]
                global_cond = obs.flatten(start_dim=1)
                
                # ==========================================
                # A. Dynamic virtual obstacle generation (target sampling)
                # ==========================================
                with torch.no_grad():
                    # Random start time t
                    t_start = torch.rand(B, 1, device=device) * sample_t_max
                    
                    # Map t to a future time-step index
                    # obs_horizon-1 is the current last frame; look ahead
                    # Use pred_horizon adaptively (no hard-coded 16)
                    future_step = (t_start * max(1, (pred_horizon - 1))).long()
                    target_idx = torch.clamp(obs_horizon - 1 + future_step, max=gt_action.shape[1]-1)
                    
                    # GT action at that step (B, 1, 10) -> (B, 10)
                    # gather requires matching index dimensions
                    idx_expanded = target_idx.unsqueeze(-1).expand(-1, -1, action_dim)
                    a_base = torch.gather(gt_action, 1, idx_expanded).squeeze(1)

                    # Roll out a short segment for more robust obstacle placement
                    curr_a, curr_t, path = a_base.clone(), t_start.clone().view(-1), [a_base.clone()]
                    warm_roll_steps = min(5, max(1, pred_horizon - 1))
                    for _ in range(warm_roll_steps):
                        v = get_drift(curr_a, curr_t, global_cond)
                        curr_a += v * dt * (curr_t < 0.98).float().unsqueeze(1)
                        curr_t += dt
                        path.append(curr_a.clone())
                    
                    path_tensor = torch.stack(path, dim=0) # (Steps, B, 10)
                    
                    # Random point on rollout path as obstacle center
                    rand_step = torch.randint(0, len(path), (B,))
                    base_obs_full = path_tensor[rand_step, torch.arange(B)] # (B, 10)
                    
                    # Obstacle defined as XYZ only (first 3 dims), with noise
                    base_obs_pos = base_obs_full[:, :3]
                    final_obs_pos = torch.clamp(
                        base_obs_pos + torch.randn_like(base_obs_pos) * float(obstacle_noise_std_norm),
                        -0.95, 0.95
                    ) # (B, 3)

                # ==========================================
                # B. Diverse action perturbations
                # ==========================================
                n_pert = 6
                
                # Gaussian noise in 10D space for diversity
                # a_base: (B, 10)
                noise_small = torch.randn_like(a_base) * 0.1
                noise_med = torch.randn_like(a_base) * 0.3
                noise_large = torch.randn_like(a_base) * 0.6
                
                # Position-only perturbation (first 3 dims)
                noise_pos_only = torch.zeros_like(a_base)
                noise_pos_only[:, :3] = torch.randn(B, 3, device=device) * 0.4
                
                a_list = [
                    a_base,                  # original GT
                    a_base + noise_small,    # small perturbation
                    a_base + noise_med,      # medium perturbation
                    a_base + noise_large,    # large perturbation
                    a_base + noise_pos_only, # position-only perturbation
                    a_base - noise_pos_only  # reverse position perturbation
                ]
                
                # (B * n_pert, 10)
                a_exp = torch.stack(a_list, dim=1).view(-1, action_dim)
                
                # Expand conditioning tensors
                cond_exp = global_cond.repeat_interleave(n_pert, dim=0)
                t_exp = t_start.repeat_interleave(n_pert, dim=0)
                obs_exp = final_obs_pos.repeat_interleave(n_pert, dim=0) # (B*n_pert, 3)

                # ==========================================
                # C. Rollout risk labels (K samples per action)
                # ==========================================
                # For each perturbed action, sample K trajectories (SDE stochasticity)
                curr_a_k = a_exp.repeat_interleave(K_samples, dim=0)
                curr_t_k = t_exp.repeat_interleave(K_samples, dim=0).view(-1)
                curr_cond_k = cond_exp.repeat_interleave(K_samples, dim=0)
                
                traj = [curr_a_k.clone()]
                with torch.no_grad():
                    rollout_steps = min(12, max(1, pred_horizon - 1))
                    for _ in range(rollout_steps): # Rollout to estimate collision risk
                        v = get_drift(curr_a_k, curr_t_k, curr_cond_k)
                        # SDE noise term: 0.05 * sqrt(dt)
                        noise = torch.randn_like(curr_a_k)
                        curr_a_k += (v * dt + 0.05 * math.sqrt(dt) * noise) * (curr_t_k < 0.99).float().unsqueeze(1)
                        curr_t_k += dt
                        traj.append(curr_a_k.clone())
                
                # Compute risk label
                # traj_stack: (Steps, Total_B, 10)
                traj_stack = torch.stack(traj, dim=0)
                
                # Expand obs_exp to match K_samples
                obs_targets = obs_exp.repeat_interleave(K_samples, dim=0)
                
                # 3D hybrid risk function
                target_risk = compute_hybrid_risk_3d(traj_stack, obs_targets, r_obs=float(r_obs_norm))
                
                # Aggregate risk over K trajectories (mean) -> (B * n_pert, 1)
                target_risk = target_risk.view(B * n_pert, K_samples).mean(dim=1, keepdim=True)

                # ==========================================
                # D. Training step
                # ==========================================
                # Critic input: a(10), t(1), obs(3), cond(flat)
                pred_risk = critic(a_exp, t_exp, obs_exp, cond_exp)
                
                loss = loss_fn(pred_risk, target_risk)
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer.step()
                
                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.5f}")

        # E. Checkpoint logic
        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.7f}")

        # Save checkpoint
        ckpt = {
            'epoch': epoch, 'model_state_dict': critic.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 'best_loss': best_loss
        }
        torch.save(ckpt, checkpoint_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(critic.state_dict(), best_model_path)
            print("Successfully saved best model.")

        # if (epoch + 1) % 20 == 0:
        #     torch.save(critic.state_dict(), save_path.replace(".pth", f"_ep{epoch+1}.pth"))

    torch.save(critic.state_dict(), save_path)
    return critic


_ccg_task_name = _infer_task_name_from_dataset_path(dataset_path)
_ccg_cfg = _task_cfg(_ccg_task_name)
_ccg_action_dim = int(dataset[0]["action"].shape[-1])
_ccg_obs_dim = int(dataset[0]["obs"].shape[-1])
_ccg_eef_pos_start = _get_eef_pos_start(wrapper, "robot0_eef_pos")
_ccg_r_obs_norm = _r_obs_norm_from_phys(dataset, float(_ccg_cfg["collision_threshold_phys"]))

trained_critic = train_robust_critic(
    dataloader,
    si_velocity_net, # pretrained policy from prior training
    action_dim=_ccg_action_dim,
    obs_dim=_ccg_obs_dim,
    obs_horizon=obs_horizon,
    pred_horizon=pred_horizon,
    eef_pos_start=_ccg_eef_pos_start,
    r_obs_norm=_ccg_r_obs_norm,
    obstacle_noise_std_norm=float(_ccg_cfg["obstacle_noise_std_norm"]),
    save_path=str(_ccg_cfg["critic_save_path"]),
    epochs=300,
)
