"""Conflict-Aware Additive Guidance (CAR) for flow-policy experiments.

The conflict score and smootherstep gate follow the reference implementation
from https://github.com/yuxuehui/CAR-guidance.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor, nn


def smootherstep(value: Tensor) -> Tensor:
    """Return the quintic smootherstep interpolation for values in [0, 1]."""
    value = value.clamp(0.0, 1.0)
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def conflict_gate(
    conflict: Tensor,
    threshold: float = 0.15,
    temperature: float = 0.1,
) -> Tensor:
    """Convert a conflict score into the CAR correction weight."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    normalized = (conflict - (threshold - temperature)) / (2.0 * temperature)
    return smootherstep(normalized)


def pairwise_conflict(
    reward_gradients: Tensor,
    zero_gradient_threshold: float = 1e-2,
) -> Tensor:
    """Compute CAR's all-pairs conflict score.

    ``reward_gradients`` has shape ``(batch, rewards, ...features)``. All
    feature dimensions are flattened, so trajectories can be represented as
    ``(batch, rewards, horizon, action_dim)``. The result has shape
    ``(batch,)`` for vector inputs or ``(batch, horizon)`` for trajectories.
    """
    if reward_gradients.ndim < 3:
        raise ValueError("reward_gradients must have shape (batch, rewards, ...)")
    if reward_gradients.shape[1] < 2:
        return torch.zeros(
            reward_gradients.shape[0],
            *reward_gradients.shape[2:-1],
            device=reward_gradients.device,
            dtype=reward_gradients.dtype,
        )

    gradients = reward_gradients.movedim(1, -2)
    norms = gradients.norm(dim=-1)
    normalized = gradients / norms.unsqueeze(-1).clamp_min(1e-8)
    cosine = torch.einsum("...id,...jd->...ij", normalized, normalized)

    reward_count = reward_gradients.shape[1]
    upper = torch.triu(
        torch.ones(reward_count, reward_count, device=gradients.device, dtype=torch.bool),
        diagonal=1,
    )
    valid = norms >= zero_gradient_threshold
    mask = upper & (valid.unsqueeze(-1) & valid.unsqueeze(-2))
    conflict = ((1.0 - cosine) / 2.0).masked_fill(~mask, 0.0)
    pair_count = mask.sum(dim=(-1, -2)).clamp_min(1)
    return conflict.sum(dim=(-1, -2)) / pair_count


class ConflictAwareAdditiveGuidance(nn.Module):
    """Add a correction to a flow velocity when reward gradients conflict.

    ``base_guidance`` and ``correction`` receive the same arguments and must
    return tensors shaped like ``base_velocity``. ``reward_gradients`` should
    return one gradient per reward with shape ``(batch, rewards, ...)``.
    """

    def __init__(
        self,
        base_guidance: Callable[..., Tensor],
        correction: Callable[..., Tensor],
        reward_gradients: Callable[..., Tensor],
        *,
        threshold: float = 0.15,
        temperature: float = 0.1,
        zero_gradient_threshold: float = 1e-2,
    ) -> None:
        super().__init__()
        self.base_guidance = base_guidance
        self.correction = correction
        self.reward_gradients = reward_gradients
        self.threshold = threshold
        self.temperature = temperature
        self.zero_gradient_threshold = zero_gradient_threshold

    def forward(self, x: Tensor, t: Tensor, base_velocity: Tensor, **kwargs: Any) -> Tensor:
        base = self.base_guidance(x=x, t=t, base_velocity=base_velocity, **kwargs)
        correction = self.correction(x=x, t=t, base_velocity=base_velocity, **kwargs)
        gradients = self.reward_gradients(x=x, t=t, base_velocity=base_velocity, **kwargs)
        conflict = pairwise_conflict(gradients, self.zero_gradient_threshold)
        weight = conflict_gate(conflict, self.threshold, self.temperature)
        while weight.ndim < correction.ndim:
            weight = weight.unsqueeze(-1)
        return base + weight * correction

