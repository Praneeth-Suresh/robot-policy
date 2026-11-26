"""Schema contracts, generated-artifact drift, and trace IO behavior."""

import copy
import hashlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from guided_streaming_policy.lab import schema

__all__ = ["minimal_rollout_manifest", "minimal_demonstration_manifest"]

ROOT = Path(__file__).resolve().parents[2]
HAS_JSONSCHEMA = importlib.util.find_spec("jsonschema") is not None
STEP = {"step": 0, "t": 0.0, "obs": {"position": [1.0, 2.0]},
        "action_executed": {"position": [3.0, 4.0]}}
COMMON_FIELDS = (
    "schema_version", "kind", "episode_id", "source_episode_id", "task", "split",
    "split_source", "timebase", "observation_schema", "action_schema", "fields_present",
    "provenance",
)
ROLLOUT_FIELDS = (
    "condition", "seed", "reset_state_fingerprint", "policy", "resolved_config", "simulator",
    "outcome",
)


def minimal_demonstration_manifest() -> dict[str, Any]:
    """Return a complete demonstration with explicit absent split metadata."""
    field = {"shape": [2], "dtype": "float64", "units": "m", "frame": "world"}
    return {
        "schema_version": "0.1", "kind": "demonstration", "episode_id": "demo-0",
        "source_episode_id": "source-0", "dataset": {"name": "fixture", "revision": "1",
        "checksum": "dataset-digest"}, "task": "fixture-task", "split": None,
        "split_source": "none", "timebase": {"unit": "s", "mode": "fixed_dt", "dt": 0.1},
        "observation_schema": {"fields": {"position": copy.deepcopy(field)}},
        "action_schema": {"convention": "absolute", "fields": {"position": field}},
        "fields_present": ["obs", "action_executed"],
        "provenance": {"tool_version": "0.1", "git_commit": "fixture", "command": "test",
                       "created_at": "2026-01-01T00:00:00Z"},
    }


def minimal_rollout_manifest() -> dict[str, Any]:
    """Return a rollout with all required policy and outcome provenance."""
    manifest = minimal_demonstration_manifest()
    del manifest["dataset"]
    manifest.update({
        "kind": "rollout", "episode_id": "rollout-0", "condition": "static", "seed": 0,
        "reset_state_fingerprint": None,
        "policy": {"name": "fixture", "checkpoint_id": "fixture", "checkpoint_sha256": "abc"},
        "resolved_config": {"path": "config.json", "sha256": "abc"},
        "simulator": {"name": "fixture", "version": "1"},
        "outcome": {"native_success": None, "terminated": False, "truncated": True,
                    "reason": "step cap", "length": 1},
    })
    return manifest


def _run_manifest() -> dict[str, Any]:
    rollout = minimal_rollout_manifest()
    return {"schema_version": "0.1", "run_id": "run-0", "task": rollout["task"],
            "policy": rollout["policy"], "simulator": rollout["simulator"],
            "provenance": rollout["provenance"],
            "episodes": [{"episode_id": "rollout-0", "condition": "static", "seed": 0}]}


def _doctor_report() -> dict[str, Any]:
    return {"schema_version": "0.1", "input": {"path": "fixture", "checksum": "abc"},
            "summary": {"counts": {"error": 0, "warning": 0, "info": 1},
                        "checks_run": ["fixture"], "checks_skipped": []},
            "findings": [{"check": "fixture", "severity": "info", "episode_id": "demo-0",
                          "message": "Not collected", "hint": "Collect telemetry"}]}


class SchemaTests(unittest.TestCase):
    def test_provenance_with_explicit_time_outside_git(self) -> None:
        created_at = "2026-01-01T00:00:00Z"
        with tempfile.TemporaryDirectory() as temporary:
            provenance = schema.make_provenance("gsp lab doctor fixture", created_at,
                                                root=Path(temporary))
        self.assertEqual(set(provenance), {"tool_version", "git_commit", "command", "created_at"})
        self.assertEqual(provenance["created_at"], created_at)
        self.assertEqual(provenance["command"], "gsp lab doctor fixture")
        self.assertEqual(provenance["git_commit"], "unknown")
        self.assertIsInstance(provenance["tool_version"], str)

    def test_generated_files_match_builders(self) -> None:
        expected_names = {"episode_manifest.schema.json", "episode_step.schema.json",
                          "run_manifest.schema.json", "doctor_report.schema.json",
                          "comparison_config.schema.json", "compare_result.schema.json",
                          "gate_config.schema.json", "gate_result.schema.json"}
        self.assertEqual(set(schema.generated_schemas()), expected_names)
        for name, document in schema.generated_schemas().items():
            with self.subTest(name=name):
                expected = schema.canonical_json(document).encode("utf-8")
                self.assertEqual((ROOT / "schemas" / name).read_bytes(), expected)

    def test_required_fields(self) -> None:
        self.assertEqual(set(schema.required_manifest_fields("demonstration")),
                         set(COMMON_FIELDS + ("dataset",)))
        self.assertEqual(set(schema.required_manifest_fields("rollout")),
                         set(COMMON_FIELDS + ROLLOUT_FIELDS))
        with self.assertRaises(ValueError):
            schema.required_manifest_fields("unknown")

    @unittest.skipUnless(HAS_JSONSCHEMA, "jsonschema is optional")
    def test_valid_documents(self) -> None:
        import jsonschema

        documents = [(minimal_demonstration_manifest(), schema.manifest_json_schema()),
                     (minimal_rollout_manifest(), schema.manifest_json_schema()),
                     (STEP, schema.step_json_schema()),
                     (_run_manifest(), schema.run_manifest_json_schema()),
                     (_doctor_report(), schema.doctor_report_json_schema())]
        for document, definition in documents:
            with self.subTest(schema=definition["$id"], kind=document.get("kind")):
                jsonschema.Draft202012Validator.check_schema(definition)
                jsonschema.Draft202012Validator(definition).validate(document)

    @unittest.skipUnless(HAS_JSONSCHEMA, "jsonschema is optional")
    def test_invalid_documents(self) -> None:
        import jsonschema

        missing = minimal_rollout_manifest()
        del missing["outcome"]
        unknown = minimal_rollout_manifest()
        unknown["unknown"] = True
        version = minimal_rollout_manifest()
        version["schema_version"] = "0.2"
        timebase = minimal_rollout_manifest()
        del timebase["timebase"]["dt"]
        missing_dataset = minimal_demonstration_manifest()
        del missing_dataset["dataset"]
        for manifest in (missing, unknown, version, timebase, missing_dataset):
            with self.subTest(manifest=manifest), self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(manifest, schema.manifest_json_schema())
        report = _doctor_report()
        report["findings"][0]["severity"] = "critical"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(report, schema.doctor_report_json_schema())
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate({**STEP, "unknown": True}, schema.step_json_schema())

    @unittest.skipUnless(HAS_JSONSCHEMA, "jsonschema is optional")
    def test_extended_outcome_and_geometry(self) -> None:
        import jsonschema

        manifest = minimal_rollout_manifest()
        manifest["outcome"].update({"detail": "trial crashed", "native": {
            "reward": 0.5, "extra": {"values": [1, None, True]},
        }})
        manifest["geometry"] = {"obstacles": [], "block": {"pose": [1, 2, 3]},
                                "arena": {"width": 512, "metadata": {"frame": "pixels"}}}
        jsonschema.validate(manifest, schema.manifest_json_schema())
        for key, value in (("detail", 1), ("native", []), ("unknown", True)):
            invalid = copy.deepcopy(manifest)
            invalid["outcome"][key] = value
            with self.subTest(key=key), self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(invalid, schema.manifest_json_schema())
        for key in ("block", "arena"):
            invalid = copy.deepcopy(manifest)
            invalid["geometry"][key] = []
            with self.subTest(key=key), self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(invalid, schema.manifest_json_schema())

    @unittest.skipUnless(HAS_JSONSCHEMA, "jsonschema is optional")
    def test_optional_telemetry_and_metadata(self) -> None:
        import jsonschema

        manifest = minimal_rollout_manifest()
        manifest.update({"synthetic": True, "media": [{"kind": "video", "uri": "movie.mp4"}],
                         "geometry": {"obstacles": [{"id": "a", "kind": "circle",
                         "position": [0, 1], "radius": 1, "units": "m", "frame": "world",
                         "source": "fixture"}], "goal": {"pose": [2, 3]}}})
        manifest["timebase"] = {"unit": "s", "mode": "timestamps"}
        manifest["fields_present"] = []
        jsonschema.validate(manifest, schema.manifest_json_schema())
        jsonschema.validate({"step": 0, "t": 0}, schema.step_json_schema())
        jsonschema.validate({**STEP, "info": {"contacts": 0}}, schema.step_json_schema())


class TraceIOTests(unittest.TestCase):
    def test_episode_round_trip_and_no_overwrite(self) -> None:
        episode = schema.Episode(minimal_rollout_manifest(), (copy.deepcopy(STEP),))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = schema.write_episode(episode, root)
            self.assertEqual(path, root / "rollout-0")
            self.assertEqual(schema.read_episode(path), episode)
            before = {child.name: child.read_bytes() for child in path.iterdir()}
            with self.assertRaises(FileExistsError):
                schema.write_episode(episode, root)
            self.assertEqual({child.name: child.read_bytes() for child in path.iterdir()}, before)
            self.assertEqual((path / "steps.jsonl").read_text(),
                             json.dumps(STEP, sort_keys=True, separators=(",", ":")) + "\n")

    def test_nan_step_survives_round_trip(self) -> None:
        step = copy.deepcopy(STEP)
        step["obs"]["position"][0] = float("nan")
        with tempfile.TemporaryDirectory() as temporary:
            path = schema.write_episode(schema.Episode(minimal_rollout_manifest(), (step,)),
                                        Path(temporary))
            self.assertTrue(math.isnan(schema.read_episode(path).steps[0]["obs"]["position"][0]))

    def test_unreadable_and_invalid_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing"
            with self.assertRaises(schema.TraceReadError):
                schema.read_episode(path)
            path.mkdir()
            for manifest, steps in (("{", ""), ("{}", "{"), ("[]", ""), ("{}", "[]\n")):
                (path / "manifest.json").write_text(manifest)
                (path / "steps.jsonl").write_text(steps)
                with self.subTest(manifest=manifest, steps=steps):
                    with self.assertRaises(schema.TraceReadError):
                        schema.read_episode(path)

    def test_episode_id_stays_within_run_directory(self) -> None:
        for episode_id in ("..", "../escape", "/absolute", ""):
            manifest = minimal_rollout_manifest()
            manifest["episode_id"] = episode_id
            with tempfile.TemporaryDirectory() as temporary, self.subTest(episode_id=episode_id):
                with self.assertRaises(ValueError):
                    schema.write_episode(schema.Episode(manifest, ()), Path(temporary))

    def test_run_manifest_and_sorted_episode_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _run_manifest()
            schema.write_run_manifest(manifest, root)
            self.assertEqual(schema.read_run_manifest(root), manifest)
            for name in ("z", "a"):
                episode = minimal_rollout_manifest()
                episode["episode_id"] = name
                schema.write_episode(schema.Episode(episode, ()), root)
            (root / "ignored").mkdir()
            self.assertEqual(schema.list_episode_dirs(root), [root / "a", root / "z"])
            (root / "run_manifest.json").write_text("[")
            with self.assertRaises(schema.TraceReadError):
                schema.read_run_manifest(root)
            (root / "run_manifest.json").unlink()
            with self.assertRaises(schema.TraceReadError):
                schema.read_run_manifest(root)

    def test_canonical_json_and_hashes(self) -> None:
        expected = '{\n  "a": "é",\n  "b": 1\n}\n'
        self.assertEqual(schema.canonical_json({"b": 1, "a": "é"}), expected)
        self.assertEqual(schema.canonical_json({"a": "é", "b": 1}), expected)
        with self.assertRaises(ValueError):
            schema.canonical_json({"x": float("nan")})
        compact = '{"a":"é","b":1}'.encode()
        digest = hashlib.sha256(compact).hexdigest()
        self.assertEqual(schema.sha256_bytes(compact), digest)
        self.assertEqual(schema.sha256_json({"b": 1, "a": "é"}), digest)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data"
            path.write_bytes(compact)
            self.assertEqual(schema.sha256_file(path), digest)

    def test_schema_generation_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "schemas"
            self.assertEqual(schema.main([str(path)]), 0)
            self.assertEqual(sorted(item.name for item in path.iterdir()),
                             sorted(schema.generated_schemas()))
            self.assertEqual(schema.write_generated_schemas(path),
                             [path / name for name in sorted(schema.generated_schemas())])

    def test_schema_command_exit_codes(self) -> None:
        with redirect_stderr(io.StringIO()):
            self.assertEqual(schema.main([]), 3)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(schema.main(["--help"]), 0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "file"
            path.write_text("existing")
            with redirect_stderr(io.StringIO()):
                self.assertEqual(schema.main([str(path)]), 2)
            self.assertEqual(path.read_text(), "existing")
