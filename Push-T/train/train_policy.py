import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from core.dataset import PushTDataset
from core.models import ConditionalUnet1D
from core.flow_policy import LinearlyInterpolateTrajectory, gamma_t_si

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = PROJECT_ROOT / "data" / "pusht_cchi_v7_replay.zarr.zip"

    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    obs_dim = 5
    action_dim = 2

    batch_size = 256
    num_epochs_si = 1000

    dataset = PushTDataset(
        dataset_path=str(dataset_path),
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
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

    si_denoiser_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        updownsample_type="Linear",
        sin_embedding_scale=100,
    ).to(device)

    print("Models initialized for Streaming SI Policy.")

    optimizer_si = torch.optim.AdamW(
        [
            {"params": si_velocity_net.parameters()},
            {"params": si_denoiser_net.parameters()},
        ],
        lr=1e-4,
        weight_decay=1e-6,
    )

    ema_si_v = EMAModel(parameters=si_velocity_net.parameters(), power=0.75)
    ema_si_eta = EMAModel(parameters=si_denoiser_net.parameters(), power=0.75)

    lr_scheduler_si = get_scheduler(
        name="cosine",
        optimizer=optimizer_si,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs_si,
    )

    sigma0 = 0.4
    k = 10.0

    print("Starting training for Streaming SI Policy...")

    with tqdm(range(num_epochs_si), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    nobs = nbatch['obs'].to(device)
                    naction = nbatch['action'].to(device)
                    
                    ξ = naction[:, obs_horizon-1:, :] 
                    B = ξ.shape[0]
                    
                    t = torch.rand(B, device=device)
                    
                    ξt, dξdt = LinearlyInterpolateTrajectory(ξ, t)
                    
                    # a_t_fp = ξ(t) + σ(t) * ε1
                    t_expanded = t.view(B, 1)
                    sigma_t_fp = sigma0 * torch.exp(-k * t_expanded)
                    noise_fp = torch.randn_like(ξt)
                    a_t_fp = ξt + sigma_t_fp * noise_fp
                    
                    # v_target = dξ/dt - k * (a_t_fp - ξ(t))
                    v_target = dξdt - k * (a_t_fp - ξt)
                    
                    # x_t = a_t_fp + γ(t) * z
                    gamma = gamma_t_si(t_expanded)
                    z_noise_si = torch.randn_like(ξt)
                    x_t_in_s = a_t_fp + gamma * z_noise_si
                    
                    net_input = x_t_in_s.unsqueeze(1)
                    global_cond = nobs.flatten(start_dim=1)
                    
                    v_pred = si_velocity_net(sample=net_input, timestep=t, global_cond=global_cond)
                    v_pred = v_pred.squeeze(1)
                    
                    eta_pred = si_denoiser_net(sample=net_input, timestep=t, global_cond=global_cond)
                    eta_pred = eta_pred.squeeze(1)
                    
                    loss_v = nn.functional.mse_loss(v_pred, v_target)
                    loss_eta = nn.functional.mse_loss(eta_pred, z_noise_si)
                    
                    loss = loss_v + loss_eta
                    
                    loss.backward()
                    optimizer_si.step()
                    optimizer_si.zero_grad()
                    lr_scheduler_si.step()
                    
                    ema_si_v.step(si_velocity_net.parameters())
                    ema_si_eta.step(si_denoiser_net.parameters())
                    
                    epoch_loss.append(loss.item())
                    tepoch.set_postfix(loss=loss.item(), v_loss=loss_v.item())
                    
            tglobal.set_postfix(loss=np.mean(epoch_loss))

    ckpt_path_si = PROJECT_ROOT / "checkpoints" / "pusht_state_1000ep_si.ckpt"
    torch.save({
        'velocity_net': si_velocity_net.state_dict(),
        'denoiser_net': si_denoiser_net.state_dict()
    }, ckpt_path_si)
    print(f"Saved Streaming SI model to {ckpt_path_si}")

    ema_si_velocity_net = si_velocity_net
    ema_si_denoiser_net = si_denoiser_net
    ema_si_v.copy_to(ema_si_velocity_net.parameters())
    ema_si_eta.copy_to(ema_si_denoiser_net.parameters())


if __name__ == "__main__":
    main()