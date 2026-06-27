import torch
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from core.config import *  # noqa: F401,F403  (device / action_dim / obs_horizon / pred_horizon 等)
from core.models import ConditionalUnet1D
from core.dataset import dataset, dataloader


EPS = 1e-6


def gamma_t_si(t):
    # 使用简单的抛物线/气泡形状，在 t=0 和 t=1 时为 0，中间膨胀
    # 这里的系数 0.1 控制 "SI 气泡" 的大小，相对于 SFP 的管状噪声
    return 0.1 * torch.sqrt(t * (1.0 - t) + EPS)

def d_gamma_dt_si(t):
    # gamma(t) 对 t 的导数
    return 0.1 * (1.0 - 2.0 * t) / (2.0 * torch.sqrt(t * (1.0 - t) + EPS))


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

    # Compute the lower and upper limits of the bins that the time-points lie in.
    scaled_t = t * (T - 1)  # (B,) lies in [0, T-1]
    l = scaled_t.floor().long().clamp(0, T - 2)  # (B,) lower bin limits
    u = (l + 1).clamp(0, T - 1)  # (B,) upper bin limits
    λ = scaled_t - l.float()  # fractional part, lies in [0, 1]

    # Query the values of the upper and lower bin limits.
    batch_idx = torch.arange(B, device=ξ.device)  # (B,)
    ξl = ξ[batch_idx, l, :]  # (B, A)
    ξu = ξ[batch_idx, u, :]  # (B, A)

    # Linearly interpolate between bin limits to get position.
    λ = λ.unsqueeze(-1)  # (B, 1)
    ξt = ξl + λ * (ξu - ξl)  # (B, A)

    # Compute velocity as first-order hold.
    # Note that the time interval between two bins is Δt = 1 / (T-1).
    dξdt = (ξu - ξl) * (T - 1)  # (B, A)

    return ξt, dξdt  # (B, A) and (B, A)

def SampleCFMInputsAndTargets(ξt, dξdt, t, k, σ0):
    """
    Sample inputs and targets for the conditional flow matching loss (CFM)
    given positions and velocities at time t.

    This functions performs the following sampling (Eq. 2 and 3 of the paper):
        a ~ N(ξ(t), σ₀² exp(-2kt))  # (Eq. 3 in the paper)
        v = -k (a - ξ(t)) + dξdt(t)  # (Eq. 2 in the paper)

    Args:
        ξt (Tensor, shape=(B, A)): positions at time t.
        dξdt (Tensor, shape=(B, A)): velocities at time t.
        t (Tensor, shape=(B,)): times in [0, 1].
        k (float): Stabilizing gains of the conditional flow.
        σ0 (float): initial standard deviation of the noise added to the action.

    Returns:
        a (Tensor, shape=(B, A)): noised actions at time t
        v (Tensor, shape=(B, A)): noised action velocity targets at time t
    """
    # error = σ0 * torch.exp(-k*t).unsqueeze(1) * torch.randn_like(xt)
    t = t.unsqueeze(-1)  # (B, 1)
    sampled_error = σ0 * torch.exp(-k * t) * torch.randn_like(ξt)  # (B, A)
    a = ξt + sampled_error  # (B, A) ⟸ Eq. 3 in the paper
    v = -k * sampled_error + dξdt  # (B, A) ⟸ Eq. 2 in the paper

    return a, v  # (B, A) and (B, A)



def build_si_policy(num_epochs_si: int = 1200, lr: float = 1e-4):
    """
    装配 Streaming SI 策略所需的全部对象（网络 / 优化器 / EMA / scheduler）。

    —— 把原先在模块顶层裸奔的实例化全部收进来，运行时才执行，
       消除 “import core.flow_policy 即建模型 / 读数据集 / 占显存” 的副作用。

    Returns (顺序与原顶层导出变量一致):
        si_velocity_net, si_denoiser_net,
        optimizer_si, ema_si_v, ema_si_eta,
        num_epochs_si, lr_scheduler_si
    """
    # 动态获取 obs_dim：dataset[0] 返回 {'obs': Tensor, 'action': Tensor}，
    # shape 是 (obs_horizon, dim)，取 [-1] 得到拼接后的观测维度（换任务自适配）。
    obs_dim = dataset[0]['obs'].shape[-1]
    print(f"✅ 自动检测到的 obs_dim: {obs_dim}")  # 期望是 23

    # 两个网络：
    # 1. si_velocity_net: 预测确定性速度场 (v)
    # 2. si_denoiser_net: 预测噪声场 (eta/score)
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

    # 优化器：一个 AdamW 同时管两个网络
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
# ----------------------------------------------------------
