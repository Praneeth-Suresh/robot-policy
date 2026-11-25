"""The checked-in comparison and gate configs load and name registered policies."""

import unittest
from pathlib import Path

from guided_streaming_policy.lab import schema
from guided_streaming_policy.lab.adapters import find_policy, resolve_policy_config
from guided_streaming_policy.lab.tasks.registry import get_task

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARISON = REPO_ROOT / "lab/comparisons/pusht_guidance_v1.json"
GATE = REPO_ROOT / "lab/gates/pusht_v1.json"


class CheckedInConfigTests(unittest.TestCase):
    def test_comparison_uses_registered_policies_and_valid_overrides(self) -> None:
        config = schema.load_comparison_config(COMPARISON)
        adapter = get_task(config.task)
        for variant in config.variants.values():
            spec = find_policy(adapter, variant.policy)
            resolve_policy_config(spec, variant.overrides)
        self.assertEqual(config.seeds, (0, 1, 2, 3, 4))

    def test_gate_loads_and_requires_every_declared_pair(self) -> None:
        gate = schema.load_gate_config(GATE)
        comparison = schema.load_comparison_config(COMPARISON)
        cells = len(comparison.seeds) * len(comparison.conditions)
        self.assertEqual(gate.min_completed_pairs, cells)
        rule_ids = {rule.id for rule in gate.rules}
        self.assertEqual(rule_ids, {"success_not_worse", "separation_proxy"})


if __name__ == "__main__":
    unittest.main()
