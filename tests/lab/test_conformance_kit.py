"""The conformance kit passes a conforming pack and names each way a pack can fail."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from guided_streaming_policy.lab import adapters, schema, testing
from guided_streaming_policy.lab.tasks.base import FixtureSet
from guided_streaming_policy.lab.tasks.pusht.adapter import PushTTask

_FIXTURES = PushTTask().fixtures().root


def _statuses(results: list[testing.CheckResult]) -> dict[str, str]:
    return {result.name: result.status for result in results}


class PushTConformanceTests(testing.TaskContractTests):
    """Push-T is a built-in pack; its trial-backed check runs only on request."""

    run_trials = os.environ.get("GSP_LAB_REAL_RUN") == "1"

    def make_adapter(self) -> PushTTask:
        return PushTTask()


class ConformanceKitTests(unittest.TestCase):
    def test_a_conforming_pack_has_no_failures_and_ten_fixture_cases(self) -> None:
        results = testing.run_conformance(PushTTask(), run_trials=False)
        statuses = _statuses(results)
        self.assertEqual([r.name for r in results if r.status == testing.FAIL], [])
        self.assertEqual(sum(name.startswith("fixture:") for name in statuses),
                         len(testing.FIXTURE_CASES))
        self.assertEqual(len(testing.FIXTURE_CASES), 10)
        self.assertEqual(statuses["reset_fingerprint"], testing.SKIP)
        self.assertEqual(statuses["offline_demo"], testing.PASS)
        self.assertEqual(statuses["no_heavy_imports"], testing.PASS)

    def test_fingerprint_check_skips_with_a_reason_when_no_policy_can_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = PushTTask(Path(temporary))
            result = {r.name: r for r in testing.run_conformance(task)}["reset_fingerprint"]
        self.assertEqual(result.status, testing.SKIP)
        self.assertIn("no runnable policy", result.detail)

    def test_a_wrong_contract_version_fails_naming_both_numbers(self) -> None:
        with patch.object(PushTTask, "contract_version", 9):
            result = {r.name: r for r in testing.run_conformance(PushTTask(), run_trials=False)}
        self.assertEqual(result["contract_version"].status, testing.FAIL)
        self.assertIn("9", result["contract_version"].detail)
        self.assertIn("1", result["contract_version"].detail)

    def test_duplicate_or_empty_conditions_fail(self) -> None:
        from guided_streaming_policy.lab.tasks.base import ConditionSpec

        for specs in ((), (ConditionSpec("a", "x"), ConditionSpec("a", "y")),
                      (ConditionSpec("a", ""),)):
            with self.subTest(specs=specs), patch.object(PushTTask, "conditions",
                                                         return_value=specs):
                result = {r.name: r for r in testing.run_conformance(PushTTask(),
                                                                     run_trials=False)}
            self.assertEqual(result["conditions"].status, testing.FAIL)

    def test_an_adapter_that_imports_a_heavy_package_fails(self) -> None:
        from tests.lab.packs import PackSpec, installed_packs

        source = (
            "import numpy\n"
            "from guided_streaming_policy.lab.tasks.pusht.adapter import PushTTask\n\n\n"
            "class Heavy(PushTTask):\n    name = 'pusht'\n"
        )
        with installed_packs(PackSpec("heavy", source=source)):
            import importlib

            heavy = importlib.import_module("fake_pack_heavy").Heavy()
            result = {r.name: r for r in testing.run_conformance(heavy, run_trials=False)}
        self.assertEqual(result["no_heavy_imports"].status, testing.FAIL)
        self.assertIn("numpy", result["no_heavy_imports"].detail)

    def test_every_fixture_case_failure_is_reported_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixtures"
            shutil.copytree(_FIXTURES, root)
            cases = json.loads((root / "cases.json").read_text())
            del cases["cases"]["leakage"]
            cases["cases"]["corrupt"]["doctor_exit"] = 0
            cases["cases"]["seeded_regression"]["gate_exit"] = 0
            (root / "cases.json").write_text(schema.canonical_json(cases))
            with patch.object(PushTTask, "fixtures", return_value=FixtureSet(
                    root / "demos/clean", root / "demos/corrupt",
                    root / "scenarios/seeded_regression", root)):
                result = _statuses(testing.run_conformance(PushTTask(), run_trials=False))
        self.assertEqual(result["fixture:leakage"], testing.FAIL)
        self.assertEqual(result["fixture:corrupt"], testing.FAIL)
        self.assertEqual(result["fixture:seeded_regression"], testing.FAIL)
        self.assertEqual(result["fixture:clean"], testing.PASS)

    def test_a_pack_without_cases_json_fails_one_named_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixtures = FixtureSet(Path(temporary), Path(temporary), Path(temporary),
                                  Path(temporary))
            with patch.object(PushTTask, "fixtures", return_value=fixtures):
                result = _statuses(testing.run_conformance(PushTTask(), run_trials=False))
        self.assertEqual(result["fixture_cases"], testing.FAIL)

    def test_a_missing_pin_or_paper_fails(self) -> None:
        document = adapters.load_pins(PushTTask().pins_path())
        del document["tasks"]["pusht"]["policies"]["pusht/dp"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pins.json"
            path.write_text(schema.canonical_json(document))
            with patch.object(PushTTask, "pins_path", return_value=path):
                result = {r.name: r for r in testing.run_conformance(PushTTask(),
                                                                     run_trials=False)}
        self.assertEqual(result["policy_pins"].status, testing.FAIL)
        self.assertIn("pusht/dp has no pin", result["policy_pins"].detail)

    def test_the_mixin_base_class_skips_itself(self) -> None:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(testing.TaskContractTests)
        outcome = unittest.TestResult()
        suite.run(outcome)
        self.assertEqual((len(outcome.skipped), len(outcome.failures), len(outcome.errors)),
                         (1, 0, 0))


if __name__ == "__main__":
    unittest.main()
