"""Export matrix cells with provenance, pin validation and crash retention."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from guided_streaming_policy.lab import adapters, doctor, exporter, schema
from guided_streaming_policy.lab.tasks.base import TrialRecord, TrialRequest
from tests.lab.helpers import FakeTask, comparison_document, demo_episode

__all__ = ["hook"]
_CREATED_AT = "2026-01-01T00:00:00Z"


def hook(request: TrialRequest) -> TrialRecord:
    """Importable standalone trial hook using passed-through overrides."""
    episode = demo_episode("hook", offset=float(request.settings["custom"]))
    return TrialRecord(episode.steps, None, False, False, True, "max_steps", {}, None,
                       ("obs", "action_executed"))


class ExporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = FakeTask(self.root)
        self.config = self.root / "comparison.json"
        self.document = comparison_document()
        self.logs: list[str] = []

    def export(self, name: str = "run", variant: str = "baseline", **kwargs: Any) -> Path:
        self.config.write_text(schema.canonical_json(self.document))
        return exporter.export_variant(self.task, self.config, variant, self.root / name,
                                       command="test", created_at=_CREATED_AT,
                                       log=self.logs.append, **kwargs)

    def test_full_export_validates_and_doctor_has_zero_errors(self) -> None:
        out = self.export()
        run = schema.read_run_manifest(out)
        self.assertEqual(run["run_id"], "toy-comparison__baseline")
        self.assertEqual(run["comparison"]["sha256"], schema.sha256_file(self.config))
        self.assertEqual([cell["seed"] for cell in run["episodes"]], [0, 1])
        episodes = list(adapters.read_traces(out))
        report = doctor.check_episodes(episodes, input_path="fixture", input_checksum="fixture")
        self.assertEqual(report["summary"]["counts"]["error"], 0, report["findings"])
        self.assertEqual(len(self.logs), 2)
        if importlib.util.find_spec("jsonschema") is not None:
            import jsonschema
            jsonschema.validate(run, schema.run_manifest_json_schema())
            for episode in episodes:
                jsonschema.validate(episode.manifest, schema.manifest_json_schema())
                for step in episode.steps:
                    jsonschema.validate(step, schema.step_json_schema())
        manifest = episodes[0].manifest
        self.assertEqual(manifest["resolved_config"]["sha256"],
                         schema.sha256_json(json.loads((out / "resolved_config.json").read_text())))
        self.assertEqual(manifest["policy"]["checkpoint_id"], "toy.ckpt")
        self.assertEqual(manifest["policy"]["checkpoint_sha256"],
                         schema.sha256_json({"toy.ckpt": "fixture-digest"}))

    def test_matrix_order_is_preserved_without_extra_cells(self) -> None:
        self.document["matrix"] = {"conditions": ["c2", "c1"], "seeds": [7, 2]}
        with patch.object(self.task, "conditions", return_value=("c1", "c2")):
            with patch.object(self.task, "run_trial", wraps=self.task.run_trial) as run:
                out = self.export()
        expected = [("c2", 7), ("c2", 2), ("c1", 7), ("c1", 2)]
        self.assertEqual([(call.args[0].condition, call.args[0].seed)
                          for call in run.call_args_list], expected)
        cells = schema.read_run_manifest(out)["episodes"]
        self.assertEqual([(cell["condition"], cell["seed"]) for cell in cells], expected)

    def test_variant_resolved_configs_only_differ_in_real_setting(self) -> None:
        baseline = json.loads((self.export() / "resolved_config.json").read_text())
        candidate = json.loads((self.export("candidate", "candidate") /
                                "resolved_config.json").read_text())
        self.assertEqual(baseline.pop("settings"), {"gain": 1.0})
        self.assertEqual(candidate.pop("settings"), {"gain": 2.0})
        self.assertEqual(baseline, candidate)
        self.assertNotIn("variant", baseline)

    def test_partial_crash_is_written_and_next_seed_runs(self) -> None:
        self.task.crash_seeds = (0,)
        out = self.export()
        failed = schema.read_episode(out / "c1__seed0")
        outcome = failed.manifest["outcome"]
        self.assertEqual(outcome["reason"], "crashed")
        self.assertIn("fixture crash", outcome["detail"])
        self.assertIsNone(outcome["native_success"])
        self.assertFalse(outcome["terminated"])
        self.assertFalse(outcome["truncated"])
        self.assertEqual(outcome["length"], len(failed.steps))
        self.assertGreater(len(failed.steps), 0)
        next_episode = schema.read_episode(out / "c1__seed1")
        self.assertTrue(next_episode.manifest["outcome"]["native_success"])
        self.assertIn("crashed", self.logs[0])

    def test_arbitrary_hook_exception_records_empty_crash(self) -> None:
        def crash(_request: TrialRequest) -> TrialRecord:
            raise RuntimeError("unexpected failure")
        out = self.export(runner=crash)
        episode = schema.read_episode(out / "c1__seed0")
        self.assertEqual(episode.steps, ())
        self.assertEqual(episode.manifest["fields_present"], [])
        self.assertEqual(episode.manifest["outcome"]["detail"], "unexpected failure")
        self.assertEqual(len(schema.read_run_manifest(out)["episodes"]), 2)

    def test_setup_validation(self) -> None:
        cases = (
            ("task", "wrong", "task"),
            ("simulator_version", "wrong", "simulator"),
            ("matrix", {"conditions": ["wrong"], "seeds": [0]}, "condition"),
            ("variants", {"baseline": {"policy": "unknown"},
                          "candidate": {"policy": "toy/b"}}, "toy/a.*toy/b"),
        )
        original = copy.deepcopy(self.document)
        for key, value, message in cases:
            with self.subTest(key=key):
                self.document = copy.deepcopy(original)
                self.document[key] = value
                self.document["intended_differences"] = []
                with self.assertRaisesRegex(exporter.SetupError, message):
                    self.export()
                self.assertFalse((self.root / "run").exists())

    def test_status_overrides_pins_output_and_variant_validation(self) -> None:
        self.task.unavailable_reason = "missing fixture checkpoint"
        with self.assertRaisesRegex(exporter.SetupError, "missing fixture checkpoint"):
            self.export()
        self.task.unavailable_reason = None
        self.document["variants"]["baseline"]["overrides"] = {"unknown": 1}
        with self.assertRaisesRegex(exporter.SetupError, "Unknown tunable"):
            self.export()
        self.document = comparison_document()
        self.task.changed_pin = True
        with self.assertRaisesRegex(exporter.SetupError, "pin changed:"):
            self.export()
        self.task.changed_pin = False
        with self.assertRaisesRegex(exporter.SetupError, "variant"):
            self.export(variant="wrong")
        self.export()
        with self.assertRaisesRegex(exporter.SetupError, "empty|overwrite"):
            self.export()

    def test_import_path_hook_and_failed_import(self) -> None:
        self.document["variants"]["baseline"] = {
            "policy": "tests.lab.test_exporter:hook", "overrides": {"custom": 3.0}}
        out = self.export()
        episode = schema.read_episode(out / "c1__seed0")
        self.assertEqual(episode.steps[0]["obs"]["agent_pos"], [3.0, 4.0])
        self.assertEqual(episode.manifest["policy"]["checkpoint_id"], "unregistered")
        self.document["variants"]["baseline"]["policy"] = "absent_module:hook"
        with self.assertRaisesRegex(exporter.SetupError, "absent_module"):
            self.export("bad")

    def test_noncallable_hook_and_malformed_pins_are_setup_errors(self) -> None:
        self.document["variants"]["baseline"]["policy"] = "tests.lab.test_exporter:_CREATED_AT"
        with self.assertRaisesRegex(exporter.SetupError, "not callable"):
            self.export()
        self.document = comparison_document()
        self.task.pins_path().write_text("invalid JSON")
        with self.assertRaises(exporter.SetupError):
            self.export()
        self.assertFalse((self.root / "run").exists())

    def test_output_file_is_rejected_and_preserved(self) -> None:
        output = self.root / "run"
        output.write_bytes(b"earlier output")
        with self.assertRaisesRegex(exporter.SetupError, "overwrite"):
            self.export()
        self.assertEqual(output.read_bytes(), b"earlier output")

    def test_invalid_config_and_default_path(self) -> None:
        with self.assertRaises(exporter.SetupError):
            exporter.default_out_dir(self.config, "baseline")
        self.config.write_text(schema.canonical_json(self.document))
        self.assertEqual(exporter.default_out_dir(self.config, "candidate"),
                         Path("out/runs/toy-comparison/candidate"))

    def test_repeatable_bytes_and_empty_output_directory(self) -> None:
        first = self.export()
        (self.root / "second").mkdir()
        second = self.export("second")
        def contents(path: Path) -> dict[Path, bytes]:
            return {file.relative_to(path): file.read_bytes()
                    for file in path.rglob("*") if file.is_file()}
        self.assertEqual(contents(first), contents(second))

    def test_interrupt_is_propagated(self) -> None:
        with patch.object(self.task, "run_trial", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.export()
