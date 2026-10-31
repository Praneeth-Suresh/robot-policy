import unittest
from importlib.util import find_spec

if find_spec("torch") is not None:
    import torch

    from guided_streaming_policy.core.car_guidance import (
        ConflictAwareAdditiveGuidance,
        conflict_gate,
        pairwise_conflict,
    )


@unittest.skipUnless(find_spec("torch") is not None, "CAR tests require the optional Torch dependency")
class CARGuidanceTests(unittest.TestCase):
    def test_aligned_and_opposed_rewards_have_expected_conflict(self) -> None:
        gradients = torch.tensor([[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]]])
        conflict = pairwise_conflict(gradients)
        torch.testing.assert_close(conflict, torch.tensor([0.0, 1.0]))

    def test_gate_is_zero_below_and_one_above_transition(self) -> None:
        gate = conflict_gate(torch.tensor([0.0, 0.15, 0.30]), threshold=0.15, temperature=0.1)
        torch.testing.assert_close(gate, torch.tensor([0.0, 0.5, 1.0]))

    def test_guidance_only_adds_correction_for_conflict(self) -> None:
        def base_guidance(**kwargs):
            return torch.ones_like(kwargs["base_velocity"])

        def correction(**kwargs):
            return torch.full_like(kwargs["base_velocity"], 2.0)

        def reward_gradients(**kwargs):
            return kwargs["x"]

        guidance = ConflictAwareAdditiveGuidance(
            base_guidance, correction, reward_gradients, threshold=0.15, temperature=0.1
        )
        x = torch.tensor([[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]]])
        result = guidance(x=x, t=torch.zeros(2), base_velocity=torch.zeros_like(x))
        torch.testing.assert_close(result[0], torch.ones(2, 2))
        torch.testing.assert_close(result[1], torch.full((2, 2), 3.0))


if __name__ == "__main__":
    unittest.main()
