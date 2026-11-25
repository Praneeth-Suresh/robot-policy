"""Expected matrix pairing, provenance compatibility, and deterministic deltas."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from guided_streaming_policy.lab import pairing, schema
from tests.lab.helpers import (
    comparison_config,
    comparison_document,
    geometry_rollout,
    rollout_run,
)

__all__ = ["basic_comparison"]

_GOLDEN = Path(__file__).parents[1] / "fixtures/lab/golden/compare_basic.json"


def basic_comparison(root: Path) -> dict[str, Any]:
    """Build the golden using fixed relative run/output locations."""
    config = comparison_config(root / "comparison.json")
    baseline = rollout_run(root / "runs/baseline", "baseline", config,
                           [geometry_rollout("a", seed=0),
                            geometry_rollout("b", seed=1, outcome_kind="truncated")])
    candidate = rollout_run(root / "runs/candidate", "candidate", config,
                            [geometry_rollout("a", seed=0, outcome_kind="crashed"),
                             geometry_rollout("b", seed=1)])
    return pairing.build_comparison(baseline, candidate, config, out_dir=root / "compare")


class PairingTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = comparison_config(self.root / "comparison.json", seeds=(0,))
        self.baseline = self.root / "baseline"
        self.candidate = self.root / "candidate"
        self.out = self.root / "out"

    def runs(self, baseline: list[schema.Episode] | None = None,
             candidate: list[schema.Episode] | None = None) -> None:
        rollout_run(self.baseline, "baseline", self.config,
                    [geometry_rollout("a")] if baseline is None else baseline)
        rollout_run(self.candidate, "candidate", self.config,
                    [geometry_rollout("a")] if candidate is None else candidate)

    def compare(self) -> dict[str, Any]:
        return pairing.build_comparison(self.baseline, self.candidate, self.config,
                                        out_dir=self.out)

    def edit(self, path: Path, **changes: Any) -> None:
        document = json.loads(path.read_text())
        document.update(changes)
        path.write_text(schema.canonical_json(document))

    def assert_status(self, status: str, reason: str | None = None) -> dict[str, Any]:
        document = self.compare()
        self.assertEqual(document["pairs"][0]["status"], status)
        self.assertEqual(sum(document["coverage"][key] for key in
                             ("completed", "incompatible", "missing")),
                         document["coverage"]["expected"])
        if reason:
            self.assertIn(reason, document["pairs"][0]["reasons"])
        self.assertEqual(pairing.compare_exit_code(document), 0 if document["valid"] else 2)
        return document

    def test_completed_distinct_trajectories_same_start(self) -> None:
        candidate = geometry_rollout("different")
        candidate.steps[1]["action_executed"]["target"][0] += 10.0
        self.runs(candidate=[candidate])
        document = self.assert_status("completed")
        self.assertTrue(document["valid"])
        self.assertEqual(document["pairs"][0]["reasons"], [])
        a, b = (document["pairs"][0][key] for key in ("baseline", "candidate"))
        self.assertNotEqual(a["steps_sha256"], b["steps_sha256"])
        self.assertEqual(a["path"], "../baseline/a")
        self.assertEqual(a["manifest_sha256"],
                         schema.sha256_file(self.baseline / "a/manifest.json"))

    def test_completed_truncated_and_crashed(self) -> None:
        self.runs([geometry_rollout("a", outcome_kind="truncated")],
                  [geometry_rollout("a", outcome_kind="crashed")])
        document = self.assert_status("completed")
        self.assertEqual(document["deltas"]["native_success"]["n"], 1)
        self.assertEqual(document["deltas"]["native_success"]["ties"], 1)
        self.assertEqual(document["deltas"]["native_success"]["candidate_successes"], 0)

    def test_incompatible_different_fingerprints(self) -> None:
        self.runs()
        self.edit(self.candidate / "a/manifest.json", reset_state_fingerprint="other")
        self.assert_status("incompatible", "reset state fingerprints differ")

    def test_incompatible_null_fingerprint(self) -> None:
        self.runs()
        self.edit(self.candidate / "a/manifest.json", reset_state_fingerprint=None)
        self.assert_status("incompatible", "reset state not recorded")

    def test_incompatible_simulator_version(self) -> None:
        self.runs()
        self.edit(self.candidate / "a/manifest.json", simulator={"name": "fixture", "version": "2"})
        self.assert_status("incompatible", "simulator versions differ from comparison")

    def test_incompatible_same_wrong_simulator_version_or_different_name(self) -> None:
        self.runs()
        for path in (self.baseline, self.candidate):
            self.edit(path / "a/manifest.json", simulator={"name": "fixture", "version": "2"})
        self.assert_status("incompatible", "simulator versions differ from comparison")
        self.edit(self.candidate / "a/manifest.json", simulator={"name": "other", "version": "2"})
        self.assert_status("incompatible", "simulator identities differ")

    def test_duplicate_key(self) -> None:
        self.runs([geometry_rollout("a"), geometry_rollout("duplicate")])
        self.assert_status("incompatible", "duplicate episode for key in baseline run")

    def test_missing_either_side(self) -> None:
        self.runs(baseline=[])
        self.assert_status("missing", "missing episode in baseline run")
        schema.write_episode(schema.read_episode(self.candidate / "a"), self.baseline)
        for path in (self.candidate / "a").iterdir():
            path.unlink()
        (self.candidate / "a").rmdir()
        self.assert_status("missing", "missing episode in candidate run")

    def test_every_cell_once_in_config_order_and_extra_episodes(self) -> None:
        document = comparison_document((2, 0))
        document["matrix"]["conditions"] = ["c2", "c1"]
        self.config = comparison_config(self.root / "comparison.json", document=document)
        self.runs([geometry_rollout("a"), geometry_rollout("extra", seed=99)], [])
        result = self.compare()
        self.assertEqual([(p["key"]["condition"], p["key"]["seed"]) for p in result["pairs"]],
                         [("c2", 2), ("c2", 0), ("c1", 2), ("c1", 0)])
        self.assertEqual(result["coverage"], {"expected": 4, "completed": 0, "incompatible": 0,
                                              "missing": 4, "extra_episodes": 1})
        delta = result["deltas"]["native_success"]
        self.assertEqual(tuple(delta[key] for key in
                               ("n", "baseline_rate", "candidate_rate", "delta")),
                         (0, None, None, None))

    def test_run_level_identity_mismatches(self) -> None:
        self.runs()
        path = self.baseline / "run_manifest.json"
        original = path.read_text()
        for change, reason in (
            ({"comparison": {"name": self.config.name, "sha256": "wrong"}},
             "baseline run comparison hash differs"),
            ({"task": "other"}, "baseline run task differs"),
            ({"variant": "candidate"}, "baseline run variant differs"),
            ({"policy": {"name": "other"}}, "baseline run policy name differs"),
        ):
            with self.subTest(change=change):
                path.write_text(original)
                self.edit(path, **change)
                result = self.compare()
                self.assertFalse(result["valid"])
                self.assertIn(reason, result["invalid_reasons"])

    def test_unintended_config_difference_and_intended_override(self) -> None:
        self.runs()
        self.assertTrue(self.compare()["valid"])
        path = self.candidate / "resolved_config.json"
        resolved = json.loads(path.read_text())
        resolved["settings"]["other"] = 2
        self.edit(path, **resolved)
        self.edit(self.candidate / "a/manifest.json", resolved_config={
            "path": "resolved_config.json", "sha256": schema.sha256_json(resolved)})
        result = self.compare()
        self.assertIn("unintended configuration difference: settings.other",
                      result["invalid_reasons"])
        self.assertFalse(result["valid"])

    def test_policy_intended_difference_covers_entry_point_and_all_settings(self) -> None:
        document = comparison_document((0,))
        document["intended_differences"] = ["policy"]
        document["variants"]["candidate"]["policy"] = "toy/b"
        self.config = comparison_config(self.root / "comparison.json", document=document)
        rollout_run(self.baseline, "baseline", self.config, [geometry_rollout("a")])
        rollout_run(self.candidate, "candidate", self.config, [geometry_rollout("a")],
                    resolved_config={"policy": "toy/b", "entry_point": "other:run",
                                     "settings": {"different": 7}})
        self.assertTrue(self.assert_status("completed")["valid"])

    def test_episode_config_hash_mismatch_invalidates_run_and_pair(self) -> None:
        self.runs()
        self.edit(self.candidate / "a/manifest.json", resolved_config={
            "path": "resolved_config.json", "sha256": "wrong"})
        result = self.assert_status("incompatible",
                                    "candidate episode resolved config hash differs")
        self.assertFalse(result["valid"])
        self.assertIn("candidate episode a resolved config hash differs", result["invalid_reasons"])

    def test_deltas_wins_losses_ties_and_null_is_unsuccessful(self) -> None:
        self.config = comparison_config(self.root / "comparison.json", seeds=(0, 1, 2, 3))
        self.runs([geometry_rollout(str(i), seed=i, outcome_kind=kind) for i, kind in enumerate(
            ("success", "native_fail", "success", "crashed"))],
            [geometry_rollout(str(i), seed=i, outcome_kind=kind) for i, kind in enumerate(
                ("crashed", "success", "success", "native_fail"))])
        delta = self.compare()["deltas"]["native_success"]
        self.assertEqual({k: v for k, v in delta.items() if k != "note"}, {
            "n": 4, "baseline_successes": 2, "candidate_successes": 2, "baseline_rate": 0.5,
            "candidate_rate": 0.5, "delta": 0.0, "wins": 1, "losses": 1, "ties": 2})
        self.assertIn("null", delta["note"])
        self.assertIn("not successful", delta["note"])

    def test_deltas_exclude_incompatible_and_missing_cells(self) -> None:
        self.config = comparison_config(self.root / "comparison.json", seeds=(0, 1, 2))
        self.runs([geometry_rollout("a", outcome_kind="native_fail"),
                   geometry_rollout("b", seed=1)],
                  [geometry_rollout("a"), geometry_rollout("b", seed=1),
                   geometry_rollout("c", seed=2)])
        self.edit(self.candidate / "b/manifest.json", reset_state_fingerprint="different")
        document = self.compare()
        self.assertEqual([p["status"] for p in document["pairs"]],
                         ["completed", "incompatible", "missing"])
        delta = document["deltas"]["native_success"]
        self.assertEqual((delta["n"], delta["baseline_rate"], delta["candidate_rate"],
                          delta["delta"], delta["wins"], delta["losses"]), (1, 0, 1, 1, 1, 0))
        self.assertFalse(document["valid"])

    def test_extra_episodes_in_both_runs_are_ignored(self) -> None:
        self.runs([geometry_rollout("a"), geometry_rollout("extra", seed=98)],
                  [geometry_rollout("a"), geometry_rollout("extra", seed=99)])
        document = self.assert_status("completed")
        self.assertTrue(document["valid"])
        self.assertEqual(document["invalid_reasons"], [])
        self.assertEqual(document["coverage"]["extra_episodes"], 2)

    def test_unreadable_run_and_resolved_config(self) -> None:
        with self.assertRaises(schema.TraceReadError):
            self.compare()
        self.runs()
        for contents in ("{", "[]"):
            (self.candidate / "resolved_config.json").write_text(contents)
            with self.assertRaises(schema.TraceReadError):
                self.compare()

    def test_determinism_and_golden(self) -> None:
        result = basic_comparison(self.root)
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        config = schema.load_comparison_config(self.root / "comparison.json")
        again = pairing.build_comparison(self.root / "runs/baseline", self.root / "runs/candidate",
                                         config, out_dir=self.root / "compare")
        self.assertEqual(schema.canonical_json(result), schema.canonical_json(again))
        self.assertEqual(schema.canonical_json(result).encode(), _GOLDEN.read_bytes())
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        self.assertFalse((self.root / "compare").exists())

    @unittest.skipUnless(importlib.util.find_spec("jsonschema"), "jsonschema is optional")
    def test_document_validates_against_committed_schema(self) -> None:
        import jsonschema

        result = basic_comparison(self.root)
        definition = json.loads((Path(__file__).parents[2] /
                                 "schemas/compare_result.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(definition)
        jsonschema.validate(result, definition)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate({**result, "unexpected": True}, definition)

    def test_summary_includes_pair_reasons_and_delta(self) -> None:
        self.runs(baseline=[])
        summary = pairing.render_summary(self.compare())
        for text in (self.config.name, self.config.sha256[:12], "missing episode in baseline run",
                     "native success", "valid", "expected=1"):
            self.assertIn(text, summary)
