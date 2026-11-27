"""A one-line registry addition exercises the entire lab with a one-dimensional task."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

from guided_streaming_policy.cli import main
from guided_streaming_policy.lab import adapters, exporter, schema
from guided_streaming_policy.lab.tasks import registry
from guided_streaming_policy.lab.tasks.base import (
    CONTRACT_VERSION,
    ConditionSpec,
    FixtureSet,
    PolicySpec,
    PolicyStatus,
    ProbeResult,
    TaskAdapter,
    TrialRecord,
    TrialRequest,
    TunableSpec,
)

__all__ = ["ToyTask"]

ROOT = Path(__file__).resolve().parents[2]
DT_SECONDS = 0.05
CREATED_AT = "2026-01-01T00:00:00Z"
STEP_COUNT = 6
VIOLATION_STEP = 3
VIOLATION_SEED = 7
OBSTACLE_METERS = 0.8
OBSTACLE_RADIUS_METERS = 0.1
THRESHOLD_METERS = 0.15
START_METERS = 0.1
STEP_METERS = 0.025
VIOLATING_POSITION_METERS = 0.75
TIME_DECIMALS = 6


class ToyTask:
    """Rail positions in metres; all inputs, pins and fixtures live in a temporary root."""

    name = "toy"
    contract_version = CONTRACT_VERSION
    root = Path("/unused-toy-root")

    def observation_schema(self) -> dict[str, Any]:
        return {"fields": {"x": {"shape": [1], "dtype": "float64", "units": "m", "frame": "rail"}}}

    def action_schema(self) -> dict[str, Any]:
        return {"convention": "absolute", "fields": {"u": self.observation_schema()["fields"]["x"]}}

    def timebase(self) -> dict[str, Any]:
        return {"unit": "s", "mode": "fixed_dt", "dt": DT_SECONDS}

    def conditions(self) -> tuple[ConditionSpec, ...]:
        return (ConditionSpec("rail", "A cart on a one-dimensional rail."),)

    def policies(self) -> tuple[PolicySpec, ...]:
        return tuple(
            PolicySpec(
                name,
                "Rail controller",
                "Synthetic contract test",
                "toy:run",
                ("toy.weights",),
                (TunableSpec("gain", 1.0, "Rail gain"),),
            )
            for name in ("toy/a", "toy/b")
        )

    def policy_statuses(self) -> tuple[PolicyStatus, ...]:
        return tuple(PolicyStatus(spec, "runnable", None) for spec in self.policies())

    def probe(self) -> ProbeResult:
        return ProbeResult("rail-sim", "1.0", {}, {})

    def pins_path(self) -> Path:
        return self.root / "pins.json"

    def current_pins(self) -> dict[str, Any]:
        return {
            "policies": {
                spec.name: {
                    "checkpoints": {"toy.weights": "synthetic-weights"},
                    "config_sha256": adapters.config_sha256(
                        adapters.resolve_policy_config(spec, {})
                    ),
                }
                for spec in self.policies()
            }
        }

    def fixtures(self) -> FixtureSet:
        return FixtureSet(self.root / "clean", self.root / "corrupt", self.root / "scenario")

    def run_trial(self, request: TrialRequest) -> TrialRecord:
        gain = float(request.settings["gain"])
        positions = [START_METERS + step * STEP_METERS * gain for step in range(STEP_COUNT)]
        if request.policy == "toy/b" and request.seed == VIOLATION_SEED:
            positions[VIOLATION_STEP] = VIOLATING_POSITION_METERS
        steps = tuple(
            {
                "step": step,
                "t": round(step * DT_SECONDS, TIME_DECIMALS),
                "obs": {"x": [position]},
                "action_executed": {"u": [position]},
            }
            for step, position in enumerate(positions)
        )
        geometry = {
            "obstacles": [
                {
                    "id": "virtual",
                    "kind": "virtual_point",
                    "position": [OBSTACLE_METERS],
                    "radius": OBSTACLE_RADIUS_METERS,
                    "units": "m",
                    "frame": "rail",
                    "source": "synthetic",
                }
            ]
        }
        return TrialRecord(
            steps,
            schema.sha256_json([request.seed, START_METERS]),
            True,
            True,
            False,
            "complete",
            {},
            geometry,
            ("obs", "action_executed", "geometry", "reset_state_fingerprint"),
        )

    def read_demonstrations(self, path: Path) -> Iterator[schema.Episode]:
        """Read native lists into episodes; malformed telemetry is retained for Doctor."""
        for index, values in enumerate(json.loads(path.read_text())):
            episode_id = f"toy-demo-{index}"
            manifest = {
                "schema_version": "0.1",
                "kind": "demonstration",
                "task": self.name,
                "episode_id": episode_id,
                "source_episode_id": episode_id,
                "dataset": {"name": "toy-lists", "revision": "1", "checksum": "synthetic"},
                "split": None,
                "split_source": "none",
                "synthetic": True,
                "observation_schema": self.observation_schema(),
                "action_schema": self.action_schema(),
                "timebase": self.timebase(),
                "fields_present": ["obs", "action_executed"],
                "provenance": schema.make_provenance("toy fixture", CREATED_AT),
            }
            steps = tuple(
                {
                    "step": step,
                    "t": round(step * DT_SECONDS, TIME_DECIMALS),
                    "obs": {"x": [value]},
                    "action_executed": {"u": [START_METERS]},
                }
                for step, value in enumerate(values)
            )
            yield schema.Episode(manifest, steps)


def populate(adapter: ToyTask) -> None:
    """Produce native demos, serialized demos and full runs through the real exporter."""
    adapter.pins_path().write_text(
        schema.canonical_json({"tasks": {"toy": adapter.current_pins()}})
    )
    for name in ("clean", "corrupt"):
        values = [START_METERS + step * STEP_METERS for step in range(STEP_COUNT)]
        if name == "corrupt":
            values[VIOLATION_STEP] = float("nan")
        native = adapter.root / f"{name}.json"
        native.write_text(json.dumps([values]) + "\n")
        for episode in adapter.read_demonstrations(native):
            schema.write_episode(episode, adapter.root / name)
    scenario = adapter.fixtures().scenario
    scenario.mkdir()
    comparison = {
        "schema_version": "0.1",
        "name": "toy-pair",
        "task": "toy",
        "simulator_version": "1.0",
        "matrix": {"conditions": ["rail"], "seeds": [3, VIOLATION_SEED]},
        "variants": {
            "baseline": {"policy": "toy/a"},
            "candidate": {"policy": "toy/b", "overrides": {"gain": 2.0}},
        },
        "intended_differences": ["policy", "overrides.gain"],
    }
    gate = {
        "schema_version": "0.1",
        "name": "toy-gate",
        "min_completed_pairs": 2,
        "max_missing_pairs": 0,
        "max_incompatible_pairs": 0,
        "rules": [
            {
                "id": "rail_separation",
                "metric": "obstacle_separation_min",
                "signal": "obs.x",
                "units": "m",
                "threshold": THRESHOLD_METERS,
                "tolerance": 0.0,
                "min_coverage": 1.0,
                "required": True,
            }
        ],
    }
    for name, document in (("comparison", comparison), ("gate", gate)):
        (scenario / f"{name}.json").write_text(schema.canonical_json(document))
    for variant in ("baseline", "candidate"):
        exporter.export_variant(
            adapter,
            scenario / "comparison.json",
            variant,
            scenario / variant,
            command="toy fixture export",
            created_at=CREATED_AT,
        )


class ToyTaskContractTests(unittest.TestCase):
    def invoke(self, expected: int, *arguments: str) -> str:
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            actual = main(["lab", *arguments])
        self.assertEqual(
            actual, expected, f"{arguments}:\n{stdout.getvalue()}\n{stderr.getvalue()}"
        )
        return stdout.getvalue()

    def test_offline_pipeline(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(ToyTask, "root", Path(temporary)),
        ):
            # This is the sole registration/wiring change required by a new task.
            registry.register("toy", ToyTask)
            try:
                adapter: TaskAdapter = ToyTask()
                populate(ToyTask())
                self.exercise_pipeline(Path(temporary), adapter)
            finally:
                registry.unregister("toy")

    def exercise_pipeline(self, root: Path, adapter: TaskAdapter) -> None:
        policies = self.invoke(0, "policies", "--task", "toy")
        self.assertIn("toy/a", policies)
        self.assertIn("toy/b", policies)
        scenario = adapter.fixtures().scenario
        for name, code in (("clean", 0), ("corrupt", 1)):
            self.invoke(
                code,
                "doctor",
                str(root / f"{name}.json"),
                "--task",
                "toy",
                "--out",
                str(root / "doctor" / name),
            )
        corrupt = json.loads((root / "doctor/corrupt/doctor.json").read_text())
        self.assertTrue(
            any(
                item["check"] == "value.non_finite" and item["step"] == VIOLATION_STEP
                for item in corrupt["findings"]
            )
        )
        for variant in ("baseline", "candidate"):
            self.invoke(
                0,
                "run",
                "--comparison",
                str(scenario / "comparison.json"),
                "--variant",
                variant,
                "--out",
                str(root / variant),
            )
        self.invoke(
            0, "doctor", str(root / "candidate"), "--task", "toy", "--out", str(root / "doctor/run")
        )
        self.invoke(
            0,
            "compare",
            str(root / "baseline"),
            str(root / "candidate"),
            "--comparison",
            str(scenario / "comparison.json"),
            "--out",
            str(root / "compare"),
        )
        self.invoke(
            1,
            "gate",
            str(root / "compare/compare.json"),
            "--gate",
            str(scenario / "gate.json"),
            "--out",
            str(root / "gate"),
        )
        result = json.loads((root / "gate/gate.json").read_text())
        self.assertEqual((result["verdict"], result["exit_code"]), ("REGRESSION", 1))
        self.assertEqual(
            (result["rules"][0]["id"], result["rules"][0]["units"], result["rules"][0]["verdict"]),
            ("rail_separation", "m", "fail"),
        )
        self.assertEqual(len(result["regressions"]), 1)
        regression = result["regressions"][0]
        self.assertEqual(
            regression["key"], {"condition": "rail", "seed": VIOLATION_SEED, "task": "toy"}
        )
        self.assertEqual(regression["first_violation"]["step"], VIOLATION_STEP)
        self.invoke(
            0,
            "report",
            str(root / "compare/compare.json"),
            "--gate",
            str(root / "gate/gate.json"),
            "--out",
            str(root / "report"),
        )
        self.assert_report(root / "report/report.html")
        self.invoke(1, "demo", "--task", "toy", "--out", str(root / "demo"))
        demo_result = json.loads((root / "demo/gate/gate.json").read_text())
        self.assertEqual(demo_result["regressions"][0]["first_violation"]["step"], VIOLATION_STEP)
        self.validate_documents(root, scenario)

    def assert_report(self, path: Path) -> None:
        page = path.read_text()
        payload = page.split('<script type="application/json" id="trace-data">', 1)[1].split(
            "</script>", 1
        )[0]
        data = json.loads(payload)
        self.assertEqual(len(data["pairs"]), 2)
        pair = data["pairs"][0]
        self.assertEqual(pair["baseline"]["observation_schema"]["fields"]["x"]["shape"], [1])
        self.assertEqual(pair["separation"]["signal"], "obs.x")
        self.assertEqual(pair["separation"]["units"], "m")
        separation = page.split("function separation(run)")[1].split("function actionSize")[0]
        self.assertNotRegex(separation, r"point\(value\)|value\[[01]\]|length === 2")
        self.assertIn('id="signal-plots"', page)
        self.assertIn("Separation proxy", page)
        self.assertIn("REGRESSION", page)

    def validate_documents(self, root: Path, scenario: Path) -> None:
        import importlib.util

        if importlib.util.find_spec("jsonschema") is None:
            return
        import jsonschema

        examples = {
            "comparison_config.schema.json": scenario / "comparison.json",
            "gate_config.schema.json": scenario / "gate.json",
            "run_manifest.schema.json": root / "candidate/run_manifest.json",
            "episode_manifest.schema.json": root / "candidate/rail__seed7/manifest.json",
            "doctor_report.schema.json": root / "doctor/clean/doctor.json",
            "compare_result.schema.json": root / "compare/compare.json",
            "gate_result.schema.json": root / "gate/gate.json",
        }
        for filename, path in examples.items():
            with self.subTest(schema=filename):
                definition = json.loads((ROOT / "schemas" / filename).read_text())
                jsonschema.validate(json.loads(path.read_text()), definition)
        definition = json.loads((ROOT / "schemas/episode_step.schema.json").read_text())
        for line in (root / "candidate/rail__seed7/steps.jsonl").read_text().splitlines():
            jsonschema.validate(json.loads(line), definition)

    def test_pipeline_imports_no_optional_runtime_packages(self) -> None:
        script = """
import importlib.abc
import sys
import unittest
class BlockRuntimePackages(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'torch', 'zarr', 'pymunk'}:
            return importlib.util.spec_from_loader(fullname, self)
        return None
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        raise AssertionError('Toy pipeline imported optional package: ' + module.__name__)
sys.meta_path.insert(0, BlockRuntimePackages())
suite = unittest.defaultTestLoader.loadTestsFromName(
    'tests.lab.test_toy_task_contract.ToyTaskContractTests.test_offline_pipeline')
result = unittest.TextTestRunner().run(suite)
raise SystemExit(not result.wasSuccessful())
"""
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
