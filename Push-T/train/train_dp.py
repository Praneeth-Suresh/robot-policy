import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from core.dataset import PushTDataset
from core.models import ConditionalUnet1D

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_horizon = 2
    obs_dim = 5
    action_dim = 2
    pred_horizon = 16
    action_horizon = 8

    dataset_path = PROJECT_ROOT / "data" / "pusht_cchi_v7_replay.zarr.zip"

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

    # Create PyTorch model for diffusion policy.
    dp_noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        updownsample_type="Conv",
        sin_embedding_scale=1,
    ).to(device)

    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # The choice of beta schedule has a big impact on performance.
        # We found squared cosine works the best.
        beta_schedule='squaredcos_cap_v2',
        # Clip output to [-1,1] to improve stability.
        clip_sample=True,
        # Our network predicts noise instead of denoised action.
        prediction_type='epsilon'
    )

    # Diffusion policy: train.
    num_epochs = 100

    # Exponential Moving Average accelerates training and improves stability.
    # It holds a copy of the model weights.
    ema_dp = EMAModel(
        parameters=dp_noise_pred_net.parameters(),
        power=0.75)

    # Standard ADAM optimizer. EMA parameters are not optimized.
    optimizer = torch.optim.AdamW(
        params=dp_noise_pred_net.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Preserve the original scheduler step count for benchmark compatibility.
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataset) * num_epochs
    )

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # Note that the data is normalized in the dataset.
                    # Device transfer
                    nobs = nbatch['obs'].to(device)  # (B, To, O)
                    naction = nbatch['action'].to(device)  # (B, Tp, A)
                    B = nobs.shape[0]

                    # Observation as FiLM conditioning
                    obs_cond = nobs.flatten(start_dim=1)  # (B, To*O)

                    # Sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)  # (B, Tp, A)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()  # (B,)

                    # Forward diffusion process: Add noise to the clean images
                    # according to the noise magnitude at each diffusion iteration.
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)  # (B, Tp, A)

                    # Predict the noise residual.
                    noise_pred = dp_noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema_dp.step(dp_noise_pred_net.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

    # Weights of the EMA model are used for inference.
    ema_noise_pred_net_dp = dp_noise_pred_net
    ema_dp.copy_to(ema_noise_pred_net_dp.parameters())

    ckpt_path_dp = PROJECT_ROOT / "checkpoints" / "pusht_state_100ep_dp.ckpt"

    torch.save(
        ema_noise_pred_net_dp.state_dict(),
        ckpt_path_dp,
    )

    print(f"Saved diffusion policy checkpoint to {ckpt_path_dp}")


if __name__ == "__main__":
    main()
