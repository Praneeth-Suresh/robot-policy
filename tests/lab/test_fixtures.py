"""Committed synthetic fixtures are deterministic, bounded and all exercised."""

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from guided_streaming_policy.cli import main
from guided_streaming_policy.lab import schema
from guided_streaming_policy.lab.tasks.pusht.adapter import PushTTask
from tests.lab.fixture_builders import FIRST_VIOLATION_STEP, build_fixtures

__all__: list[str] = []

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _ROOT / "guided_streaming_policy/lab/tasks/pusht/fixtures"
MAX_FIXTURE_BYTES = 1_500_000
_DEMOS = {"clean": 0, "no_split": 0, "corrupt": 1, "leakage": 1}
_SCENARIOS = {"seeded_regression": (0, 1), "two_valid_trajectories": (0, 0),
              "missing_signal": (0, 2), "mismatched_reset": (0, 2),
              "truncated": (0, 1), "missing_pairs": (0, 2)}


def _invoke(args: list[str]) -> tuple[int, str]:
    output, error = io.StringIO(), io.StringIO()
    with redirect_stdout(output), redirect_stderr(error):
        code = main(["lab", *args])
    return code, error.getvalue()


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


class FixtureTests(unittest.TestCase):
    def test_drift_size_and_every_fixture_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary) / "fixtures"
            build_fixtures(generated)
            self.assertEqual(_tree(generated), _tree(_FIXTURES))
            with self.assertRaises(FileExistsError):
                build_fixtures(generated)
        self.assertLess(sum(map(len, _tree(_FIXTURES).values())), MAX_FIXTURE_BYTES)
        self.assertEqual({p.name for p in (_FIXTURES / "demos").iterdir()}, set(_DEMOS))
        self.assertEqual({p.name for p in (_FIXTURES / "scenarios").iterdir()}, set(_SCENARIOS))
        fixtures = PushTTask().fixtures()
        self.assertEqual(fixtures.demo_clean, _FIXTURES / "demos/clean")
        self.assertEqual(fixtures.demo_corrupt, _FIXTURES / "demos/corrupt")
        self.assertEqual(fixtures.scenario, _FIXTURES / "scenarios/seeded_regression")

    def test_every_fixture_file_is_covered_by_the_package_data_patterns(self) -> None:
        pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for pattern in ("fixtures/**/*.json", "fixtures/**/*.jsonl", "pins.json"):
            self.assertIn(f'"{pattern}"', pyproject)
        suffixes = {path.suffix for path in _FIXTURES.rglob("*") if path.is_file()}
        self.assertLessEqual(suffixes, {".json", ".jsonl"})

    def test_manifests_and_machine_configs_validate(self) -> None:
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema is not installed")
        names = {"manifest.json": "episode_manifest", "run_manifest.json": "run_manifest",
                 "comparison.json": "comparison_config", "gate.json": "gate_config"}
        paths = [p for p in _FIXTURES.rglob("*.json") if p.name in names]
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.relative_to(_FIXTURES)):
                document = json.loads(path.read_text())
                spec = json.loads((_ROOT / f"schemas/{names[path.name]}.schema.json").read_text())
                jsonschema.validate(document, spec)
                if path.name.endswith("manifest.json"):
                    self.assertIs(document["synthetic"], True)
                    self.assertEqual(document["provenance"]["created_at"], "2026-01-01T00:00:00Z")

    def test_episode_contract_and_safe_trajectories(self) -> None:
        adapter = PushTTask()
        for path in _FIXTURES.rglob("manifest.json"):
            episode = schema.read_episode(path.parent)
            manifest = episode.manifest
            with self.subTest(path=path.relative_to(_FIXTURES)):
                self.assertIs(manifest["synthetic"], True)
                self.assertEqual(manifest["observation_schema"], adapter.observation_schema())
                self.assertEqual(manifest["action_schema"], adapter.action_schema())
                self.assertEqual(manifest["timebase"], adapter.timebase())
                self.assertEqual(len(episode.steps), 40)
                if manifest["kind"] != "rollout":
                    continue
                resolved = path.parent.parent / manifest["resolved_config"]["path"]
                self.assertEqual(manifest["resolved_config"]["sha256"],
                                 schema.sha256_json(json.loads(resolved.read_text())))
                self.assertEqual(manifest["policy"]["name"], "pusht/naive")
                self.assertEqual(manifest["simulator"], {"name": "pymunk", "version": "7.2.0"})
                obstacle = manifest["geometry"]["obstacles"][0]
                self.assertEqual(obstacle["position"], [186.65, 146.5])
                self.assertEqual(obstacle["radius"], 15.36)
                for step in episode.steps:
                    if "action_executed" not in manifest["fields_present"]:
                        self.assertNotIn("action_executed", step)
                        continue
                    distance = math.dist(step["action_executed"]["target"], obstacle["position"])
                    violating = ("seeded_regression/candidate" in path.as_posix()
                                 and manifest["seed"] == 2
                                 and step["step"] == FIRST_VIOLATION_STEP)
                    if violating:
                        self.assertLess(distance, 20.0)
                    else:
                        self.assertGreaterEqual(distance, 60.0)

    def test_doctor_cases(self) -> None:
        for name, expected in _DEMOS.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                code, error = _invoke(["doctor", str(_FIXTURES / "demos" / name),
                                       "--out", str(root)])
                self.assertEqual(code, expected, error)
                findings = json.loads((root / "doctor.json").read_text())["findings"]
                self._assert_doctor(name, findings)

    def _assert_doctor(self, name: str, findings: list[dict[str, Any]]) -> None:
        if name == "clean":
            self.assertEqual(findings, [])
        elif name == "no_split":
            self.assertEqual([f["check"] for f in findings], ["split.not_checked"])
        elif name == "corrupt":
            self.assertEqual({(f["check"], f["episode_id"], f.get("step"), f.get("field"))
                              for f in findings}, {
                ("value.non_finite", "nan", 3, "obs.agent_pos"),
                ("shape.mismatch", "shape", 2, "action_executed.target"),
                ("time.non_monotonic", "time", 4, "t"),
            })
        else:
            self.assertEqual({(f["check"], f["episode_id"]) for f in findings},
                             {("split.leakage", "train"), ("split.leakage", "val")})
            self.assertEqual(len(findings), 2)

    def test_scenario_cases(self) -> None:
        for name, (compare_code, gate_code) in _SCENARIOS.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                scenario, out = _FIXTURES / "scenarios" / name, Path(temporary)
                code, error = _invoke(["compare", str(scenario / "baseline"),
                                      str(scenario / "candidate"), "--comparison",
                                      str(scenario / "comparison.json"), "--out", str(out)])
                self.assertEqual(code, compare_code, error)
                code, error = _invoke(["gate", str(out / "compare.json"), "--gate",
                                      str(scenario / "gate.json"), "--out", str(out)])
                self.assertEqual(code, gate_code, error)
                comparison = json.loads((out / "compare.json").read_text())
                result = json.loads((out / "gate.json").read_text())
                self._assert_scenario(name, comparison, result)

    def test_incomplete_scenarios_evaluate_within_gate_limits(self) -> None:
        for name, kind in (("missing_pairs", "missing"), ("mismatched_reset", "incompatible")):
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                scenario, out = _FIXTURES / "scenarios" / name, Path(temporary)
                code, error = _invoke(["compare", str(scenario / "baseline"),
                                      str(scenario / "candidate"), "--comparison",
                                      str(scenario / "comparison.json"), "--out", str(out)])
                self.assertEqual(code, 0, error)
                comparison = json.loads((out / "compare.json").read_text())
                self.assertTrue(comparison["valid"])
                self.assertEqual(comparison["invalid_reasons"], [])
                self.assertEqual(comparison["coverage"][kind], 2)
                config = json.loads((scenario / "gate.json").read_text())
                config.update(min_completed_pairs=2, **{f"max_{kind}_pairs": 2})
                path = out / "gate-config.json"
                path.write_text(schema.canonical_json(config))
                code, error = _invoke(["gate", str(out / "compare.json"), "--gate",
                                      str(path), "--out", str(out)])
                self.assertEqual(code, 0, error)
                result = json.loads((out / "gate.json").read_text())
                self.assertEqual(result["blocked_reasons"], [])
                self.assertEqual(result["coverage"]["completed"], 2)
                self.assertTrue(all(rule["verdict"] == "pass" for rule in result["rules"]))

    def _assert_scenario(
        self, name: str, comparison: dict[str, Any], result: dict[str, Any],
    ) -> None:
        pairs = {p["key"]["seed"]: p for p in comparison["pairs"]}
        rules = {r["id"]: r for r in result["rules"]}
        regressions = result["regressions"]
        if name == "seeded_regression":
            self.assertEqual(rules["success_not_worse"]["verdict"], "pass")
            self.assertEqual(rules["separation_proxy"]["verdict"], "fail")
            self.assertEqual(len(regressions), 1)
            self.assertEqual(regressions[0]["key"],
                             {"task": "pusht", "condition": "static", "seed": 2})
            self.assertEqual(regressions[0]["category"], "constraint_fail")
            self.assertEqual(regressions[0]["first_violation"]["step"], FIRST_VIOLATION_STEP)
            self.assertIs(pairs[2]["candidate"]["native_success"], True)
        elif name == "two_valid_trajectories":
            self.assertEqual(regressions, [])
            self.assertTrue(all(p["baseline"]["steps_sha256"] != p["candidate"]["steps_sha256"]
                                for p in pairs.values()))
        elif name == "missing_signal":
            self.assertEqual(rules["separation_proxy"]["verdict"], "unknown")
            self.assertIn("not collected", rules["separation_proxy"]["reason"])
        elif name == "truncated":
            self.assertEqual(pairs[1]["status"], "completed")
            self.assertEqual([r["category"] for r in regressions], ["truncated"])
        else:
            seeds, status, reason = (([1, 2], "incompatible", "reset state fingerprints differ")
                                     if name == "mismatched_reset" else
                                     ([2, 3], "missing", "missing episode in candidate run"))
            self.assertEqual([s for s, p in pairs.items() if p["status"] == status], seeds)
            for seed in seeds:
                self.assertIn(reason, pairs[seed]["reasons"])
