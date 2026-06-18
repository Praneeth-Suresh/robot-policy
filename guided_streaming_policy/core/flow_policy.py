import torch


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
