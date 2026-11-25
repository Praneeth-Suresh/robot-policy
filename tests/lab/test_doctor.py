"""Doctor check locations, provenance limits and deterministic report bytes."""

import importlib.util
import json
import unittest
from pathlib import Path
from typing import Any

from guided_streaming_policy.lab import doctor, schema
from tests.lab.helpers import demo_episode, rollout_episode

__all__: list[str] = []

_GOLDEN = Path(__file__).parents[1] / "fixtures/lab/golden/doctor_corrupt.json"


def corrupt_episode() -> schema.Episode:
    """Build the golden input; run this builder once to explicitly regenerate the golden."""
    episode = demo_episode("corrupt")
    del episode.manifest["dataset"]
    episode.steps[2]["obs"]["agent_pos"] = [1.0]
    episode.steps[3]["obs"]["agent_pos"][0] = float("nan")
    episode.steps[4]["t"] = 0.2
    episode.steps[5]["t"] = 1.0
    return episode


def _report(*episodes: schema.Episode) -> dict[str, Any]:
    return doctor.check_episodes(iter(episodes), input_path="fixture", input_checksum="fixture")


class DoctorTests(unittest.TestCase):
    def findings(self, episode: schema.Episode, check: str) -> list[dict[str, Any]]:
        return [finding for finding in _report(episode)["findings"] if finding["check"] == check]

    def test_schema_missing_required(self) -> None:
        episode = demo_episode("missing")
        del episode.manifest["dataset"]
        del episode.steps[1]["step"]
        del episode.steps[2]["t"]
        del episode.steps[3]["obs"]
        episode.manifest["fields_present"].extend(["action_proposed", "reward", "info"])
        findings = self.findings(episode, "schema.missing_required")
        locations = {(f["episode_id"], f.get("step"), f["field"]) for f in findings}
        for step, field in ((None, "dataset"), (1, "step"), (2, "t"), (3, "obs"),
                            (0, "action_proposed"), (0, "reward"), (0, "info")):
            self.assertIn(("missing", step, field), locations)

    def test_unknown_or_missing_kind_gets_no_deeper_checks(self) -> None:
        for kind in (None, "unknown"):
            episode = corrupt_episode()
            episode.manifest.pop("kind")
            if kind is not None:
                episode.manifest["kind"] = kind
            findings = _report(episode)["findings"]
            self.assertEqual([(f["check"], f["field"]) for f in findings
                              if f["episode_id"] == "corrupt"],
                             [("schema.missing_required", "kind")])

    def test_each_missing_manifest_field_remains_diagnosable(self) -> None:
        for kind, builder in (("demonstration", demo_episode), ("rollout", rollout_episode)):
            for field in schema.required_manifest_fields(kind):
                with self.subTest(kind=kind, field=field):
                    episode = builder("missing", split="train")
                    del episode.manifest[field]
                    report = _report(episode, demo_episode("other", split="val"))
                    self.assertTrue(any(f["check"] == "schema.missing_required" and
                                        f["field"] == field for f in report["findings"]
                                        if "field" in f))

    def test_absent_uncollected_telemetry_is_clean(self) -> None:
        episode = demo_episode("uncollected", split="train")
        episode.manifest["fields_present"] = []
        for step in episode.steps:
            del step["obs"]
            del step["action_executed"]
        self.assertEqual(_report(episode)["findings"], [])

    def test_shape_mismatch(self) -> None:
        episode = demo_episode("shape")
        episode.steps[2]["obs"]["agent_pos"] = [[1.0, 2.0]]
        self.assertEqual([(f["episode_id"], f["step"], f["field"]) for f in
                          self.findings(episode, "shape.mismatch")],
                         [("shape", 2, "obs.agent_pos")])

    def test_shape_missing_undeclared_ragged_and_first_step_reference(self) -> None:
        episode = demo_episode("shape")
        del episode.steps[1]["action_executed"]["target"]
        episode.steps[2]["obs"]["block_pos"] = [[1.0], [2.0, 3.0]]
        episode.steps[0]["obs"]["extra"] = [0.0]
        episode.steps[3]["obs"]["extra"] = [0.0, 1.0]
        findings = self.findings(episode, "shape.mismatch")
        self.assertTrue(any(f["step"] == 1 and f["field"] == "action_executed.target"
                            for f in findings))
        self.assertTrue(any(f["step"] == 2 and f["field"] == "obs.block_pos" for f in findings))
        extra = [f for f in findings if f["step"] == 3 and f["field"] == "obs.extra"]
        self.assertEqual(len(extra), 2)

    def test_value_non_finite(self) -> None:
        episode = demo_episode("nan")
        episode.steps[3]["obs"]["agent_pos"][0] = float("nan")
        self.assertEqual([(f["episode_id"], f["step"], f["field"]) for f in
                          self.findings(episode, "value.non_finite")],
                         [("nan", 3, "obs.agent_pos")])

    def test_non_finite_all_numeric_groups_and_boolean_exclusion(self) -> None:
        episode = demo_episode("numeric")
        episode.steps[0]["t"] = float("inf")
        episode.steps[1]["action_executed"]["target"] = [float("-inf"), 0.0]
        episode.steps[2]["action_proposed"] = {"target": [[float("nan")]]}
        episode.steps[3]["reward"] = float("nan")
        episode.steps[4]["reward"] = True
        self.assertEqual([f["field"] for f in self.findings(episode, "value.non_finite")],
                         ["t", "action_executed.target", "action_proposed.target", "reward"])
        self.assertEqual(self.findings(episode, "time.non_monotonic"), [])

    def test_time_non_monotonic(self) -> None:
        episode = demo_episode("time")
        episode.steps[4]["t"] = 0.2
        self.assertEqual([(f["episode_id"], f["step"], f["field"]) for f in
                          self.findings(episode, "time.non_monotonic")], [("time", 4, "t")])

    def test_equal_times_fail_and_non_finite_times_skip_adjacent_comparisons(self) -> None:
        episode = demo_episode("time")
        episode.steps[2]["t"] = episode.steps[1]["t"]
        episode.steps[3]["t"] = 0.2
        episode.steps[4]["t"] = float("nan")
        self.assertEqual([f["step"] for f in self.findings(episode, "time.non_monotonic")], [2])
        self.assertEqual(self.findings(episode, "time.gap"), [])

    def test_time_gap(self) -> None:
        episode = demo_episode("gap")
        episode.steps[5]["t"] = 1.0
        self.assertEqual([(f["episode_id"], f["step"], f["field"]) for f in
                          self.findings(episode, "time.gap")], [("gap", 5, "t")])

    def test_split_leakage(self) -> None:
        report = _report(demo_episode("b", split="val"), demo_episode("a", split="train"))
        findings = [f for f in report["findings"] if f["check"] == "split.leakage"]
        self.assertEqual([f["episode_id"] for f in findings], ["a", "b"])
        self.assertIn("b", findings[0]["message"])
        self.assertIn("val", findings[0]["message"])
        self.assertEqual(doctor.exit_code(report), 1)

    def test_split_duplicate_episode(self) -> None:
        for split in (None, "train"):
            report = _report(demo_episode("b", split=split), demo_episode("a", split=split))
            findings = [f for f in report["findings"] if f["check"] == "split.duplicate_episode"]
            self.assertEqual([f["episode_id"] for f in findings], ["b"])
            self.assertIn("a", findings[0]["message"])
            self.assertEqual(findings[0]["severity"], "warning")

    def test_leakage_does_not_infer_a_split_for_unlabeled_matches(self) -> None:
        a, b = demo_episode("train", split="train"), demo_episode("val", split="val")
        unlabeled = demo_episode("unlabeled")
        findings = _report(a, b, unlabeled)["findings"]
        self.assertEqual([(f["check"], f["episode_id"]) for f in findings],
                         [("split.leakage", "train"), ("split.leakage", "val")])
        self.assertEqual(schema.canonical_json(_report(a, b, unlabeled)),
                         schema.canonical_json(_report(unlabeled, b, a)))

    def test_split_not_checked(self) -> None:
        report = _report(demo_episode("a"), demo_episode("b", offset=1.0))
        self.assertEqual([(f["check"], f["severity"], f["episode_id"])
                          for f in report["findings"]],
                         [("split.not_checked", "info", "(dataset)")])
        self.assertIn("no leakage claim", report["findings"][0]["message"])
        self.assertEqual(doctor.exit_code(report), 0)

    def test_fingerprints_round_numbers_and_exclude_non_finite_episodes(self) -> None:
        a, b = demo_episode("a", split="train"), demo_episode("b", split="val")
        b.steps[0]["obs"]["agent_pos"][0] += 0.00001
        self.assertEqual(sum(f["check"] == "split.leakage" for f in _report(a, b)["findings"]), 2)
        a.steps[3]["reward"] = float("nan")
        self.assertFalse(any(f["check"].startswith("split.") for f in _report(a, b)["findings"]))

    def test_outcome_inconsistent(self) -> None:
        episode = rollout_episode("rollout", outcome={"length": 99, "terminated": True,
                                                     "reason": "crashed", "native_success": True})
        findings = self.findings(episode, "outcome.inconsistent")
        self.assertEqual({f["field"] for f in findings},
                         {"outcome.length", "outcome.terminated", "outcome.native_success"})
        self.assertTrue(all(f["episode_id"] == "rollout" and "step" not in f for f in findings))

    def test_clean_and_diverging_trajectories_from_same_start(self) -> None:
        a = demo_episode("a", split="train")
        b = demo_episode("b", split="val")
        b.steps[1]["action_executed"]["target"][0] += 1.0
        report = _report(a, b, rollout_episode("rollout", offset=2.0, split="val"))
        self.assertEqual(report["findings"], [])
        self.assertEqual(report["summary"]["checks_run"], sorted(doctor.CHECK_IDS))
        self.assertEqual(doctor.exit_code(report), 0)

    def test_checks_skipped_reasons_and_mixed_eligibility(self) -> None:
        episode = demo_episode("timestamps")
        episode.manifest["timebase"] = {"mode": "timestamps", "unit": "s"}
        expected = {"time.gap": "timebase has no nominal dt",
                    "split.leakage": "no split provenance in the input",
                    "outcome.inconsistent": "input has no rollout episodes"}
        self.assertEqual({v["check"]: v["reason"] for v in
                          _report(episode)["summary"]["checks_skipped"]}, expected)
        episode.manifest["timebase"] = {"mode": "fixed_dt", "unit": "s"}
        self.assertIn({"check": "time.gap", "reason": expected["time.gap"]},
                      _report(episode)["summary"]["checks_skipped"])
        report = _report(episode, demo_episode("fixed", offset=2.0, split="val"),
                         rollout_episode("rollout", offset=3.0))
        self.assertEqual(report["summary"]["checks_skipped"], [])

    def test_determinism_and_golden(self) -> None:
        report = _report(corrupt_episode())
        actual = schema.canonical_json(report).encode()
        self.assertEqual(actual, schema.canonical_json(_report(corrupt_episode())).encode())
        self.assertTrue(_GOLDEN.is_file(),
                        "Create doctor_corrupt.json by running corrupt_episode() and "
                        "schema.canonical_json(_report(corrupt_episode())) once.")
        self.assertEqual(actual, _GOLDEN.read_bytes())

    @unittest.skipUnless(importlib.util.find_spec("jsonschema"), "jsonschema is not installed")
    def test_report_validates_against_committed_schema(self) -> None:
        import jsonschema

        path = Path(__file__).parents[2] / "schemas/doctor_report.schema.json"
        jsonschema.validate(_report(corrupt_episode()), json.loads(path.read_text()))

    def test_render_summary_names_findings(self) -> None:
        summary = doctor.render_summary(_report(corrupt_episode()))
        self.assertIn("schema.missing_required", summary)
        self.assertIn("obs.agent_pos", summary)
        self.assertIn("corrupt", summary)

    def test_rollout_run_has_no_split_findings(self) -> None:
        for split in (None, "val"):
            with self.subTest(split=split):
                report = _report(rollout_episode("a"), rollout_episode("b", split=split))
                self.assertEqual(report["findings"], [])
                skipped = {item["check"]: item["reason"]
                           for item in report["summary"]["checks_skipped"]}
                for check in ("split.leakage", "split.duplicate_episode", "split.not_checked"):
                    self.assertEqual(skipped[check], "input has no demonstration episodes")
                    self.assertNotIn(check, report["summary"]["checks_run"])

    def test_rollouts_do_not_supply_demo_split_provenance_or_duplicates(self) -> None:
        report = _report(demo_episode("demo"), rollout_episode("rollout", split="val"))
        self.assertEqual([f["check"] for f in report["findings"]], ["split.not_checked"])
