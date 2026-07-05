import torch
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from core.config import *  # noqa: F401,F403  (device / action_dim / obs_horizon / pred_horizon, etc.)
from core.models import ConditionalUnet1D
from core.dataset import dataset, dataloader



def LinearlyInterpolateTrajectory(ξ, t):
    """
    Vectorized computation of positions and velocities if each trajectory
    (from a batch of trajectories) at given times for each trajectory, using
    linear interpolation.

    ξ (Tensor, dtype=float, shape=(B, T, A)): batch of action trajectories.
    t (Tensor, dtype=float, shape=(B,)): batch of times in [0, 1].

    Returns:
        ξt   (Tensor, shape=(B, A)): positions at time t
        dξdt (Tensor, shape=(B, A)): velocities at time t
    """
    B, T, A = ξ.shape

    scaled_t = t * (T - 1)
    l = scaled_t.floor().long().clamp(0, T - 2)
    u = (l + 1).clamp(0, T - 1)
    λ = scaled_t - l.float()

    batch_idx = torch.arange(B, device=ξ.device)
    ξl = ξ[batch_idx, l, :]
    ξu = ξ[batch_idx, u, :]

    λ = λ.unsqueeze(-1)
    ξt = ξl + λ * (ξu - ξl)

    dξdt = (ξu - ξl) * (T - 1)

    return ξt, dξdt


def SampleCFMInputsAndTargets(ξt, dξdt, t, k, σ0):
    """
    Sample inputs and targets for the conditional flow matching loss (CFM)
    given positions and velocities at time t.

    This functions performs the following sampling (Eq. 2 and 3 of the paper):
        a ~ N(ξ(t), σ₀² exp(-2kt))
        v = -k (a - ξ(t)) + dξdt(t)
    """
    t = t.unsqueeze(-1)
    sampled_error = σ0 * torch.exp(-k * t) * torch.randn_like(ξt)
    a = ξt + sampled_error
    v = -k * sampled_error + dξdt

    return a, v


EPS = 1e-6


def gamma_t_si(t):
    return 0.1 * torch.sqrt(t * (1.0 - t) + EPS)


def d_gamma_dt_si(t):
    return 0.1 * (1.0 - 2.0 * t) / (2.0 * torch.sqrt(t * (1.0 - t) + EPS))


def build_si_policy(num_epochs_si: int = 1200, lr: float = 1e-4):
    # Dynamically infer obs_dim: dataset[0] returns {'obs': Tensor, 'action': Tensor},
    # shape (obs_horizon, dim); [-1] gives concatenated observation dim (task-adaptive).
    obs_dim = dataset[0]['obs'].shape[-1]
    print(f"✅ Auto-detected obs_dim: {obs_dim}")  # expected: 23

    # Two networks:
    # 1. si_velocity_net: predicts deterministic velocity field (v)
    # 2. si_denoiser_net: predicts noise field (eta/score)
    si_velocity_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        updownsample_type='Linear',
        sin_embedding_scale=100,
    ).to(device)

    si_denoiser_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        updownsample_type='Linear',
        sin_embedding_scale=100,
    ).to(device)

    print("Models initialized for Streaming SI Policy.")

    # Optimizer: one AdamW for both networks
    optimizer_si = torch.optim.AdamW([
        {'params': si_velocity_net.parameters()},
        {'params': si_denoiser_net.parameters()}
    ], lr=lr, weight_decay=1e-6)

    # EMA (Exponential Moving Average) for better inference stability
    ema_si_v = EMAModel(parameters=si_velocity_net.parameters(), power=0.75)
    ema_si_eta = EMAModel(parameters=si_denoiser_net.parameters(), power=0.75)

    lr_scheduler_si = get_scheduler(
        name='cosine',
        optimizer=optimizer_si,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs_si
    )

    return (
        si_velocity_net,
        si_denoiser_net,
        optimizer_si,
        ema_si_v,
        ema_si_eta,
        num_epochs_si,
        lr_scheduler_si,
    )

(
    si_velocity_net,
    si_denoiser_net,
    optimizer_si,
    ema_si_v,
    ema_si_eta,
    num_epochs_si,
    lr_scheduler_si,
) = build_si_policy()
