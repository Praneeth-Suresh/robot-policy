import os
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from core.dataset import PushTDataset
from core.models import ConditionalUnet1D
from core.critic import CollisionPredictionCritic, compute_hybrid_risk


PROJECT_ROOT = Path(__file__).resolve().parents[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

obs_horizon = 2
obs_dim = 5
action_dim = 2
pred_horizon = 16
action_horizon = 8


def load_training_inputs():
    dataset_path = PROJECT_ROOT / "data" / "pusht_cchi_v7_replay.zarr.zip"
    ckpt_path_si = PROJECT_ROOT / "checkpoints" / "pusht_state_1000ep_si.ckpt"

    dataset = PushTDataset(
        dataset_path=str(dataset_path),
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    si_velocity_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        updownsample_type="Linear",
        sin_embedding_scale=100,
    ).to(device)

    if os.path.isfile(ckpt_path_si):
        checkpoint = torch.load(ckpt_path_si, map_location=device)
        si_velocity_net.load_state_dict(checkpoint["velocity_net"])
        si_velocity_net.eval()
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path_si}")

    return dataloader, si_velocity_net


def train_robust_critic(dataloader, si_velocity_net, action_dim, obs_dim, obs_horizon, 
                        epochs=200, K_samples=12, sample_t_max=0.8, save_path=PROJECT_ROOT / "checkpoints" / "CCG_V6.pth"):
    
    print(f"--- Initialization ---")
    save_path = Path(save_path)
    checkpoint_path = save_path.with_name(f"{save_path.stem}_latest_checkpoint{save_path.suffix}")
    best_model_path = save_path.with_name(f"{save_path.stem}_best{save_path.suffix}")
    
    critic = CollisionPredictionCritic(action_dim, obs_dim, obs_horizon, hidden_dim=1024, depth=6).to(device)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    
    start_epoch = 0
    best_loss = float('inf')
    dt = 1.0 / 16.0

    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        critic.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f"Restarting at Epoch {start_epoch+1}")

    def get_drift(a, t, cond):
        t_in = torch.clamp(t, 0.02, 0.98).view(-1)
        v = si_velocity_net(a.unsqueeze(1), t_in, cond).squeeze(1)
        return v

    for epoch in range(start_epoch, epochs):
        epoch_losses = []
        critic.train()
        
        with tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False) as pbar:
            for batch in pbar:
                obs = batch['obs'].to(device)
                gt_action = batch['action'].to(device)
                B = obs.shape[0]
                global_cond = obs.flatten(start_dim=1)
                
                with torch.no_grad():
                    t_start = torch.rand(B, 1, device=device) * sample_t_max
                    target_idx = torch.clamp((obs_horizon-1 + (t_start*16).long()), max=gt_action.shape[1]-1)
                    a_base = torch.gather(gt_action, 1, target_idx.unsqueeze(-1).expand(-1,-1,2)).squeeze(1)

                    curr_a, curr_t, path = a_base.clone(), t_start.clone().view(-1), [a_base.clone()]
                    for _ in range(8):
                        v = get_drift(curr_a, curr_t, global_cond)
                        curr_a += v * dt * (curr_t < 0.98).float().unsqueeze(1)
                        curr_t += dt
                        path.append(curr_a.clone())
                    
                    path_tensor = torch.stack(path, dim=0)
                    base_obs = path_tensor[torch.randint(0,9,(B,)), torch.arange(B)]
                    final_obs = torch.clamp(base_obs + torch.randn_like(base_obs)*0.15, -0.95, 0.95)

                n_pert = 6
                perp = torch.stack([-(path_tensor[-1,:,1]-path_tensor[0,:,1]), path_tensor[-1,:,0]-path_tensor[0,:,0]], dim=1)
                a_list = [a_base, a_base+torch.randn_like(a_base)*0.1, a_base+torch.randn_like(a_base)*0.25, 
                          a_base+perp*0.2, a_base-perp*0.2, a_base+torch.randn_like(a_base)*0.5]
                
                a_exp = torch.stack(a_list, dim=1).view(-1, 2)
                cond_exp = global_cond.repeat_interleave(n_pert, dim=0)
                t_exp = t_start.repeat_interleave(n_pert, dim=0)
                obs_exp = final_obs.repeat_interleave(n_pert, dim=0)

                curr_a_k = a_exp.repeat_interleave(K_samples, dim=0)
                curr_t_k = t_exp.repeat_interleave(K_samples, dim=0).view(-1)
                curr_cond_k = cond_exp.repeat_interleave(K_samples, dim=0)
                
                traj = [curr_a_k.clone()]
                with torch.no_grad():
                    for _ in range(12):
                        v = get_drift(curr_a_k, curr_t_k, curr_cond_k)
                        curr_a_k += (v*dt + 0.05*math.sqrt(dt)*torch.randn_like(curr_a_k)) * (curr_t_k < 0.99).float().unsqueeze(1)
                        curr_t_k += dt
                        traj.append(curr_a_k.clone())
                
                target_risk = compute_hybrid_risk(torch.stack(traj, dim=0), obs_exp.repeat_interleave(K_samples, dim=0))
                target_risk = target_risk.view(B * n_pert, K_samples).mean(dim=1, keepdim=True)

                pred_risk = critic(a_exp, t_exp, obs_exp, cond_exp)
                loss = loss_fn(pred_risk, target_risk)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer.step()
                
                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.5f}")

        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.7f}")

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

        if (epoch + 1) % 40 == 0:
            torch.save(critic.state_dict(), save_path.with_name(f"{save_path.stem}_ep{epoch+1}{save_path.suffix}"))

    torch.save(critic.state_dict(), save_path)
    return critic


def main():
    dataloader, si_velocity_net = load_training_inputs()
    trained_critic = train_robust_critic(
        dataloader=dataloader,
        si_velocity_net=si_velocity_net,
        action_dim=action_dim,
        obs_dim=obs_dim,
        obs_horizon=obs_horizon,
        epochs=200,
        K_samples=12,
        sample_t_max=0.8,
        save_path=PROJECT_ROOT / "checkpoints" / "CCG_V6.pth",
    )


if __name__ == "__main__":
    main()
