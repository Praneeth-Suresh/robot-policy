"""Gate evidence, regression replay anchors, coverage, and immutable trace checks."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from guided_streaming_policy.lab import gate, pairing, schema
from tests.lab.helpers import gate_comparison, gate_config, gate_document, gate_episodes

__all__ = ["regression_gate"]

_GOLDEN = Path(__file__).parents[1] / "fixtures/lab/golden/gate_regression.json"


def regression_gate(root: Path) -> dict[str, Any]:
    """Build a separation regression with hand-known first violation and native success."""
    episodes = gate_episodes()
    episodes[1].steps[2]["action_executed"]["target"] = [10.0, 0.0]
    document = gate_comparison(root, candidate=episodes)
    return gate.evaluate_gate(
        document, gate_config(root / "gate-config.json"), base_dir=root / "compare"
    )


class GateTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / "compare"
        self.config_document = gate_document()

    def evaluate(self, document: dict[str, Any]) -> dict[str, Any]:
        config = gate_config(self.root / "gate-config.json", self.config_document)
        return gate.evaluate_gate(document, config, base_dir=self.base)

    def assert_exit(self, document: dict[str, Any], code: int, verdict: str) -> None:
        self.assertEqual(gate.gate_exit_code(document), code, document["blocked_reasons"])
        self.assertEqual(document["verdict"], verdict)
        self.assertEqual(
            gate.render_verdict(document).splitlines()[0],
            f"Policy Trace Lab gate: {verdict} (exit {code})",
        )

    def test_seeded_constraint_regression_preserves_native_success(self) -> None:
        candidates = gate_episodes()
        candidates[1].steps[2]["action_executed"]["target"] = [10.0, 0.0]
        compare = gate_comparison(self.root, candidate=candidates)
        before = copy.deepcopy(compare)
        result = self.evaluate(compare)
        self.assert_exit(result, 1, "REGRESSION")
        self.assertEqual(result["rules"][1]["id"], "separation_proxy")
        self.assertEqual(result["rules"][1]["verdict"], "fail")
        self.assertEqual(result["rules"][1]["signal"], "action_executed.target")
        regression = result["regressions"][0]
        self.assertEqual(regression["key"], {"task": "toy", "condition": "c1", "seed": 1})
        self.assertEqual(regression["category"], "constraint_fail")
        self.assertEqual(regression["categories"], ["constraint_fail"])
        self.assertEqual(regression["rule_ids"], ["separation_proxy"])
        self.assertEqual(regression["first_violation"], {"step": 2, "time": 0.2})
        self.assertEqual(
            (regression["replay_step"], regression["replay_basis"]), (2, "first_violation")
        )
        self.assertTrue(compare["pairs"][1]["baseline"]["native_success"])
        self.assertTrue(compare["pairs"][1]["candidate"]["native_success"])
        self.assertEqual(compare, before)
        self.assertIn("FAIL separation_proxy", gate.render_verdict(result))
        self.assertIn("first violation: c1 seed 1 step 2 t=0.2", gate.render_verdict(result))

    def test_different_valid_trajectories_pass_and_separation_pair_statistics(self) -> None:
        candidates = gate_episodes()
        for episode, distance in zip(candidates, (60.0, 40.0, 50.0), strict=True):
            for step in episode.steps:
                step["action_executed"]["target"] = [distance, 0.0]
        result = self.evaluate(gate_comparison(self.root, candidate=candidates))
        self.assert_exit(result, 0, "PASS")
        self.assertEqual(result["regressions"], [])
        rule = result["rules"][1]
        self.assertEqual(
            (rule["baseline_value"], rule["candidate_value"], rule["delta"]), (50.0, 40.0, -10.0)
        )
        self.assertEqual(
            (rule["wins"], rule["losses"], rule["ties"], rule["samples"]), (1, 1, 1, 3)
        )
        self.assertEqual(rule["coverage"], 1.0)
        self.assertIn("gate config: toy-gate", gate.render_verdict(result))

    def test_success_delta_failure_and_exact_threshold_boundary(self) -> None:
        candidates = gate_episodes()
        candidates[0].manifest["outcome"]["native_success"] = False
        compare = gate_comparison(self.root, candidate=candidates)
        result = self.evaluate(compare)
        self.assert_exit(result, 1, "REGRESSION")
        self.assertEqual(result["rules"][0]["delta"], 2 / 3 - 1)
        self.assertEqual(result["rules"][0]["verdict"], "fail")
        self.assertEqual(result["regressions"][0]["rule_ids"], ["success_not_worse"])
        self.config_document["rules"][0]["threshold"] = 2 / 3 - 1
        self.assert_exit(self.evaluate(compare), 0, "PASS")

    def test_required_missing_signal_low_coverage_and_units_are_unknown(self) -> None:
        for name in ("missing", "coverage", "units"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                candidates = gate_episodes()
                if name == "missing":
                    candidates[0].manifest["fields_present"].remove("action_executed")
                elif name == "coverage":
                    del candidates[0].steps[0]["action_executed"]["target"]
                else:
                    candidates[0].manifest["geometry"]["obstacles"][0]["units"] = "m"
                result = gate.evaluate_gate(
                    gate_comparison(root, candidate=candidates),
                    gate_config(root / "gate-config.json"),
                    base_dir=root / "compare",
                )
                self.assert_exit(result, 2, "BLOCKED")
                rule = result["rules"][1]
                self.assertEqual(rule["verdict"], "unknown")
                self.assertEqual(rule["signal"], "action_executed.target")
                text = {
                    "missing": "action_executed.target",
                    "coverage": "coverage",
                    "units": "units mismatch",
                }[name]
                self.assertIn(text, rule["reason"])
                self.assertIn(
                    "insufficient evidence: separation_proxy", gate.render_verdict(result)
                )

    def test_fail_precedes_unknown_across_candidate_episodes(self) -> None:
        candidates = gate_episodes()
        candidates[0].manifest["fields_present"].remove("action_executed")
        candidates[1].steps[0]["action_executed"]["target"] = [0, 0]
        self.assert_exit(
            self.evaluate(gate_comparison(self.root, candidate=candidates)), 1, "REGRESSION"
        )

    def test_baseline_unknown_does_not_block_candidate_verdict(self) -> None:
        baseline = gate_episodes()
        baseline[0].manifest["fields_present"].remove("action_executed")
        result = self.evaluate(gate_comparison(self.root, baseline=baseline))
        self.assert_exit(result, 0, "PASS")
        self.assertEqual(result["rules"][1]["samples"], 2)

    def test_incompatible_pairs_limits_and_minimum(self) -> None:
        candidates = gate_episodes()
        candidates[0].manifest["reset_state_fingerprint"] = "different"
        compare = gate_comparison(self.root, candidate=candidates)
        self.config_document["min_completed_pairs"] = 2
        result = self.evaluate(compare)
        self.assert_exit(result, 2, "BLOCKED")
        self.assertIn("incompatible pairs 1 exceed maximum 0", result["blocked_reasons"])
        self.config_document["max_incompatible_pairs"] = 1
        self.assert_exit(self.evaluate(compare), 0, "PASS")
        self.config_document["min_completed_pairs"] = 3
        result = self.evaluate(compare)
        self.assert_exit(result, 2, "BLOCKED")
        self.assertIn("completed pairs 2 below minimum 3", result["blocked_reasons"])

    def test_missing_pairs_and_zero_completed_pairs(self) -> None:
        compare = gate_comparison(self.root, candidate=[])
        result = self.evaluate(compare)
        self.assert_exit(result, 2, "BLOCKED")
        self.assertIn("missing pairs 3 exceed maximum 0", result["blocked_reasons"])
        self.assertEqual(result["rules"][0]["verdict"], "unknown")
        self.assertIsNone(result["rules"][0]["delta"])

    def test_native_failure_categories_and_episode_end(self) -> None:
        candidates = gate_episodes()
        for episode, kind in zip(candidates, ("native_fail", "truncated", "crashed"), strict=True):
            episode.manifest["outcome"].update(
                native_success=False, reason=kind, truncated=kind == "truncated"
            )
        compare = gate_comparison(self.root, candidate=candidates)
        self.assertEqual(compare["coverage"]["completed"], 3)
        result = self.evaluate(compare)
        self.assert_exit(result, 1, "REGRESSION")
        self.assertEqual(
            [r["category"] for r in result["regressions"]], ["native_fail", "truncated", "crashed"]
        )
        for regression in result["regressions"]:
            self.assertIsNone(regression["first_violation"])
            self.assertEqual(regression["replay_step"], len(candidates[0].steps) - 1)
            self.assertEqual(regression["replay_basis"], "episode_end")

    def test_multiple_categories_first_time_sort_and_tolerance(self) -> None:
        candidates = gate_episodes()
        candidates[0].manifest["outcome"].update(reason="crashed", native_success=None)
        candidates[0].steps[3]["action_executed"]["target"] = [0, 0]
        candidates[2].steps[1]["action_executed"]["target"] = [0, 0]
        result = self.evaluate(gate_comparison(self.root, candidate=candidates))
        self.assertEqual([r["key"]["seed"] for r in result["regressions"]], [2, 0])
        self.assertEqual(result["regressions"][1]["categories"], ["crashed", "constraint_fail"])
        self.assertEqual(result["regressions"][1]["category"], "crashed")
        self.assertEqual(
            result["regressions"][1]["rule_ids"], ["success_not_worse", "separation_proxy"]
        )

    def test_tolerance_applies_to_constraint_and_pair_wins(self) -> None:
        candidates = gate_episodes()
        candidates[0].steps[0]["action_executed"]["target"] = [49, 0]
        self.config_document["rules"][1].update(threshold=50, tolerance=1)
        result = self.evaluate(gate_comparison(self.root, candidate=candidates))
        self.assert_exit(result, 0, "PASS")
        self.assertEqual(
            (result["rules"][1]["wins"], result["rules"][1]["losses"], result["rules"][1]["ties"]),
            (0, 0, 3),
        )

    def test_optional_failure_and_unknown_do_not_change_verdict(self) -> None:
        candidates = gate_episodes()
        candidates[0].steps[0]["action_executed"]["target"] = [0, 0]
        compare = gate_comparison(self.root, candidate=candidates)
        self.config_document["rules"][1]["required"] = False
        result = self.evaluate(compare)
        self.assert_exit(result, 0, "PASS")
        self.assertEqual(result["rules"][1]["verdict"], "fail")
        self.config_document["rules"][1]["signal"] = "obs.uncollected"
        self.assert_exit(self.evaluate(compare), 0, "PASS")

    def test_hash_tampering_and_unreadable_hashes_block(self) -> None:
        compare = gate_comparison(self.root)
        self.assertEqual(pairing.verify_trace_hashes(compare, self.base), [])
        path = self.root / "candidate/0/steps.jsonl"
        path.write_text(path.read_text() + "\n")
        result = self.evaluate(compare)
        self.assert_exit(result, 2, "BLOCKED")
        self.assertIn("hash", " ".join(result["blocked_reasons"]))
        path.unlink()
        self.assert_exit(self.evaluate(compare), 2, "BLOCKED")

    def test_invalid_comparison_shapes_and_invalid_reasons_block(self) -> None:
        compare = gate_comparison(self.root)
        invalid = [
            {},
            {**compare, "valid": False},
            {**compare, "schema_version": "0.2"},
            {**compare, "pairs": [None]},
            {**compare, "coverage": {"completed": True}},
            {**compare, "invalid_reasons": ["wrong comparison hash"]},
        ]
        for document in invalid:
            with self.subTest(document=document):
                self.assert_exit(self.evaluate(document), 2, "BLOCKED")

    def test_load_compare_read_errors_and_version(self) -> None:
        path = self.root / "compare.json"
        with self.assertRaises(schema.TraceReadError):
            pairing.load_comparison_result(path)
        for text in ("{", "[]", '{"schema_version":"0.2"}'):
            path.write_text(text)
            with self.assertRaises(schema.TraceReadError):
                pairing.load_comparison_result(path)
        compare = gate_comparison(self.root)
        path.write_text(schema.canonical_json(compare))
        self.assertEqual(pairing.load_comparison_result(path), compare)

    def test_determinism_and_golden(self) -> None:
        result = regression_gate(self.root)
        again = gate.evaluate_gate(
            pairing.build_comparison(
                self.root / "baseline",
                self.root / "candidate",
                schema.load_comparison_config(self.root / "comparison.json"),
                out_dir=self.base,
            ),
            schema.load_gate_config(self.root / "gate-config.json"),
            base_dir=self.base,
        )
        self.assertEqual(schema.canonical_json(result), schema.canonical_json(again))
        self.assertEqual(schema.canonical_json(result).encode(), _GOLDEN.read_bytes())

    @unittest.skipUnless(importlib.util.find_spec("jsonschema"), "jsonschema is optional")
    def test_result_validates_against_generated_schema(self) -> None:
        import jsonschema

        definition = json.loads(
            (Path(__file__).parents[2] / "schemas/gate_result.schema.json").read_text()
        )
        jsonschema.Draft202012Validator.check_schema(definition)
        jsonschema.validate(regression_gate(self.root), definition)
        jsonschema.validate(self.evaluate({}), definition)

    def test_missing_pair_within_limit_and_enough_completed_continues(self) -> None:
        compare = gate_comparison(self.root, candidate=gate_episodes()[:2])
        self.config_document.update(min_completed_pairs=2, max_missing_pairs=1)
        result = self.evaluate(compare)
        self.assert_exit(result, 0, "PASS")
        self.assertEqual(result["coverage"]["missing"], 1)
        self.assertEqual(result["rules"][0]["samples"], 2)

    def test_baseline_constraint_failure_is_reported_without_affecting_verdict(self) -> None:
        baseline = gate_episodes()
        baseline[0].steps[0]["action_executed"]["target"] = [0, 0]
        result = self.evaluate(gate_comparison(self.root, baseline=baseline))
        self.assert_exit(result, 0, "PASS")
        self.assertEqual(result["regressions"], [])
        self.assertEqual(result["rules"][1]["baseline_value"], 0)
        self.assertEqual(result["rules"][1]["wins"], 1)

    def test_empty_crash_replay_anchor_and_unknown_precedence(self) -> None:
        candidates = gate_episodes()
        original = candidates[0]
        original.manifest["outcome"].update(native_success=None, reason="crashed", length=0)
        candidates[0] = schema.Episode(original.manifest, ())
        compare = gate_comparison(self.root, candidate=candidates)
        result = self.evaluate(compare)
        self.assert_exit(result, 2, "BLOCKED")
        self.assertEqual(result["rules"][0]["verdict"], "fail")
        self.assertEqual(result["rules"][1]["verdict"], "unknown")
        self.config_document["rules"][1]["required"] = False
        result = self.evaluate(compare)
        self.assert_exit(result, 1, "REGRESSION")
        self.assertEqual(result["regressions"][0]["category"], "crashed")
        self.assertEqual(result["regressions"][0]["replay_step"], 0)

    def test_malformed_coverage_and_tampered_manifest_are_blocked(self) -> None:
        compare = gate_comparison(self.root)
        for coverage in (None, [], {"completed": -1}):
            with self.subTest(coverage=coverage):
                self.assert_exit(self.evaluate({**compare, "coverage": coverage}), 2, "BLOCKED")
        (self.root / "candidate/0/manifest.json").write_text("{}")
        result = self.evaluate(compare)
        self.assert_exit(result, 2, "BLOCKED")
        self.assertIn("hash mismatch", " ".join(result["blocked_reasons"]))

    @unittest.skipUnless(importlib.util.find_spec("jsonschema"), "jsonschema is optional")
    def test_result_schema_rejects_wrong_codes_and_metric_specific_fields(self) -> None:
        import jsonschema

        result = regression_gate(self.root)
        invalid = copy.deepcopy(result)
        invalid["rules"][0]["tolerance"] = 0
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema.gate_result_json_schema())
        invalid = copy.deepcopy(result)
        invalid["exit_code"] = 0
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema.gate_result_json_schema())
