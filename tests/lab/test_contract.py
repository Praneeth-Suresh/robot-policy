"""The task-pack contract: version constant, conditions and import-free inspection."""

import subprocess
import sys
import unittest
from pathlib import Path

from guided_streaming_policy.lab import adapters
from guided_streaming_policy.lab.tasks.base import CONTRACT_VERSION, ConditionSpec
from guided_streaming_policy.lab.tasks.pusht.adapter import PushTTask
from tests.lab.helpers import FakeTask

_ROOT = Path(__file__).resolve().parents[2]
_HEAVY = ("numpy", "torch", "pymunk", "zarr", "gym", "diffusers", "cv2")
_PROBE_SCRIPT = f"""
import sys
from guided_streaming_policy.lab.tasks.pusht.adapter import PushTTask
task = PushTTask()
task.probe(); task.policies(); task.policy_statuses(); task.conditions()
print(",".join(sorted(name for name in {_HEAVY!r} if name in sys.modules)))
"""


class ContractTests(unittest.TestCase):
    def test_version_is_one_and_adapters_declare_it(self) -> None:
        self.assertEqual(CONTRACT_VERSION, 1)
        self.assertEqual(PushTTask().contract_version, CONTRACT_VERSION)
        self.assertEqual(FakeTask.contract_version, CONTRACT_VERSION)

    def test_pusht_declares_static_and_dynamic_with_descriptions(self) -> None:
        specs = PushTTask().conditions()
        self.assertEqual([spec.name for spec in specs], ["static", "dynamic"])
        self.assertTrue(all(isinstance(spec, ConditionSpec) and spec.description for spec in specs))
        self.assertEqual([spec.runnable for spec in specs], [True, False])

    def test_probe_and_policy_listing_import_no_heavy_package(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT], cwd=_ROOT, capture_output=True,
            text=True, check=True,
        )
        self.assertEqual(result.stdout.strip(), "", "probe/policies imported a heavy package")

    def test_unknown_condition_lists_the_supported_names(self) -> None:
        with self.assertRaises(ValueError) as caught:
            adapters.validate_conditions(PushTTask(), ("static", "sideways"))
        message = str(caught.exception)
        self.assertIn("'sideways'", message)
        self.assertIn("static, dynamic", message)

    def test_declared_but_not_runnable_condition_is_refused_only_for_runs(self) -> None:
        adapters.validate_conditions(PushTTask(), ("dynamic",))
        with self.assertRaises(ValueError) as caught:
            adapters.validate_conditions(PushTTask(), ("dynamic",), require_runnable=True)
        self.assertIn("research scripts", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
