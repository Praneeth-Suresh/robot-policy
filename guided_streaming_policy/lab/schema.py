"""Trace contract and canonical IO, independent of task and third-party code.

Regenerate committed schemas with:
    .venv/bin/python -m guided_streaming_policy.lab.schema schemas

Telemetry keys may be absent when not collected; shape consistency with the
manifest is checked by Doctor. Geometry goals and step info are open objects.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA_VERSION", "KIND_DEMONSTRATION", "KIND_ROLLOUT", "SEVERITIES",
    "MANIFEST_FILENAME", "STEPS_FILENAME", "RUN_MANIFEST_FILENAME", "FIELDS_PRESENT_VALUES",
    "required_manifest_fields", "Episode", "TraceReadError", "manifest_json_schema",
    "step_json_schema", "run_manifest_json_schema", "doctor_report_json_schema",
    "generated_schemas", "write_generated_schemas", "canonical_json", "sha256_bytes",
    "sha256_file", "sha256_json", "write_episode", "read_episode", "list_episode_dirs",
    "write_run_manifest", "read_run_manifest", "make_provenance", "main",
]

SCHEMA_VERSION = "0.1"
KIND_DEMONSTRATION = "demonstration"
KIND_ROLLOUT = "rollout"
SEVERITIES = ("error", "warning", "info")
MANIFEST_FILENAME = "manifest.json"
STEPS_FILENAME = "steps.jsonl"
RUN_MANIFEST_FILENAME = "run_manifest.json"
FIELDS_PRESENT_VALUES = (
    "obs", "action_executed", "action_proposed", "reward", "info",
    "reset_state_fingerprint", "geometry",
)
_COMMON_FIELDS = (
    "schema_version", "kind", "episode_id", "source_episode_id", "task", "split",
    "split_source", "timebase", "observation_schema", "action_schema", "fields_present",
    "provenance",
)
_ROLLOUT_FIELDS = (
    "condition", "seed", "reset_state_fingerprint", "policy", "resolved_config",
    "simulator", "outcome",
)
_JSON_INDENT_SPACES = 2
_HASH_CHUNK_BYTES = 65536
_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
_EXIT_INVALID_INPUT = 2
_EXIT_USAGE_OR_INTERNAL = 3
_GIT_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class Episode:
    """A manifest and its ordered steps; no telemetry is synthesized."""

    manifest: dict[str, Any]
    steps: tuple[dict[str, Any], ...]


class TraceReadError(Exception):
    """A trace is missing, unreadable, or contains invalid JSON objects."""


def make_provenance(
    command: str, created_at: str | None = None, *, root: Path | None = None,
) -> dict[str, str]:
    """Record tool version, Git commit, command and UTC creation time to seconds."""
    try:
        tool_version = importlib.metadata.version("guided-streaming-policy")
    except importlib.metadata.PackageNotFoundError:
        tool_version = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2] if root is None else root,
            capture_output=True, text=True, check=True, timeout=_GIT_TIMEOUT_SECONDS,
        )
        git_commit = result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        git_commit = "unknown"
    return {
        "tool_version": tool_version, "git_commit": git_commit, "command": command,
        "created_at": created_at if created_at is not None else
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def required_manifest_fields(kind: str) -> tuple[str, ...]:
    """Return common plus kind-specific mandatory manifest keys."""
    if kind == KIND_DEMONSTRATION:
        return _COMMON_FIELDS + ("dataset",)
    if kind == KIND_ROLLOUT:
        return _COMMON_FIELDS + _ROLLOUT_FIELDS
    raise ValueError(f"Unknown episode kind: {kind!r}")


def _str() -> dict[str, Any]:
    return {"type": "string"}


def _obj(props: dict[str, Any], required: Sequence[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props,
            "required": list(props if required is None else required),
            "additionalProperties": False}


def _strings(*names: str) -> dict[str, Any]:
    return _obj({name: _str() for name in names})


def _array(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


def _document(filename: str, body: dict[str, Any]) -> dict[str, Any]:
    return {"$schema": _SCHEMA_DIALECT, "$id": filename, **body}


def _provenance() -> dict[str, Any]:
    return _strings("tool_version", "git_commit", "command", "created_at")


def _policy() -> dict[str, Any]:
    return _strings("name", "checkpoint_id", "checkpoint_sha256")


def _timebase() -> dict[str, Any]:
    result = _obj({"unit": _str(), "mode": {"enum": ["fixed_dt", "timestamps"]},
                   "dt": {"type": "number", "exclusiveMinimum": 0}}, ("unit", "mode"))
    result["allOf"] = [{"if": {"properties": {"mode": {"const": "fixed_dt"}}},
                        "then": {"required": ["dt"]}}]
    return result


def _field_schema(action: bool = False) -> dict[str, Any]:
    field = _obj({"shape": _array({"type": "integer", "minimum": 0}),
                  "dtype": _str(), "units": _str(), "frame": _str()})
    props: dict[str, Any] = {"fields": {"type": "object", "additionalProperties": field}}
    if action:
        props["convention"] = {"enum": ["absolute", "delta"]}
    return _obj(props)


def _outcome() -> dict[str, Any]:
    return _obj({"native_success": {"type": ["boolean", "null"]},
                 "terminated": {"type": "boolean"}, "truncated": {"type": "boolean"},
                 "reason": _str(), "length": {"type": "integer", "minimum": 0}})


def _geometry() -> dict[str, Any]:
    obstacle = _obj({"id": _str(), "kind": _str(), "position": _array({"type": "number"}),
                     "radius": {"type": "number", "minimum": 0}, "units": _str(),
                     "frame": _str(), "source": _str()})
    return _obj({"obstacles": _array(obstacle), "goal": {"type": "object"}}, ("obstacles",))


def manifest_json_schema() -> dict[str, Any]:
    """Build the episode manifest schema with conditional kind requirements."""
    props = {
        "schema_version": {"const": SCHEMA_VERSION},
        "kind": {"enum": [KIND_DEMONSTRATION, KIND_ROLLOUT]},
        "episode_id": _str(), "source_episode_id": _str(), "task": _str(),
        "split": {"type": ["string", "null"]}, "split_source": _str(),
        "dataset": _strings("name", "revision", "checksum"), "timebase": _timebase(),
        "observation_schema": _field_schema(), "action_schema": _field_schema(action=True),
        "fields_present": {**_array({"enum": list(FIELDS_PRESENT_VALUES)}), "uniqueItems": True},
        "provenance": _provenance(), "media": _array(_strings("kind", "uri")),
        "condition": _str(), "seed": {"type": "integer"},
        "reset_state_fingerprint": {"type": ["string", "null"]}, "policy": _policy(),
        "resolved_config": _strings("path", "sha256"), "simulator": _strings("name", "version"),
        "outcome": _outcome(), "geometry": _geometry(),
        "synthetic": {"type": "boolean", "default": False},
    }
    body = _obj(props, _COMMON_FIELDS)
    body["allOf"] = [
        {"if": {"properties": {"kind": {"const": kind}}, "required": ["kind"]},
         "then": {"required": list(required_manifest_fields(kind))}}
        for kind in (KIND_DEMONSTRATION, KIND_ROLLOUT)
    ]
    return _document("episode_manifest.schema.json", body)


def step_json_schema() -> dict[str, Any]:
    """Build a step schema; numeric arrays recurse, uncollected keys may be absent."""
    values = {"type": "object", "additionalProperties": {"$ref": "#/$defs/value"}}
    body = _obj({"step": {"type": "integer", "minimum": 0}, "t": {"type": "number"},
                 "obs": values, "action_executed": values, "action_proposed": values,
                 "reward": {"type": "number"}, "info": {"type": "object"}}, ("step", "t"))
    body["$defs"] = {"value": {"anyOf": [{"type": "number"},
                              _array({"$ref": "#/$defs/value"})]}}
    return _document("episode_step.schema.json", body)


def run_manifest_json_schema() -> dict[str, Any]:
    """Build a run index with optional comparison digest and variant label."""
    episode = _obj({"episode_id": _str(), "condition": _str(), "seed": {"type": "integer"}})
    props = {"schema_version": {"const": SCHEMA_VERSION}, "run_id": _str(), "task": _str(),
             "comparison": _strings("name", "sha256"), "variant": _str(), "policy": _policy(),
             "simulator": _strings("name", "version"), "provenance": _provenance(),
             "episodes": _array(episode)}
    required = ("schema_version", "run_id", "task", "policy", "simulator", "provenance", "episodes")
    return _document("run_manifest.schema.json", _obj(props, required))


def doctor_report_json_schema() -> dict[str, Any]:
    """Build Doctor's counts, executed/skipped checks, and ordered findings contract."""
    finding = _obj({"check": _str(), "severity": {"enum": list(SEVERITIES)},
                    "episode_id": _str(), "step": {"type": "integer", "minimum": 0},
                    "field": _str(), "message": _str(), "hint": _str()},
                   ("check", "severity", "episode_id", "message", "hint"))
    summary = _obj({"counts": _obj({key: {"type": "integer", "minimum": 0} for key in SEVERITIES}),
                    "checks_run": _array(_str()),
                    "checks_skipped": _array(_strings("check", "reason"))})
    return _document("doctor_report.schema.json", _obj({
        "schema_version": {"const": SCHEMA_VERSION}, "input": _strings("path", "checksum"),
        "summary": summary, "findings": _array(finding),
    }))


def generated_schemas() -> dict[str, dict[str, Any]]:
    """Map committed schema filenames to fresh schema documents."""
    return {"episode_manifest.schema.json": manifest_json_schema(),
            "episode_step.schema.json": step_json_schema(),
            "run_manifest.schema.json": run_manifest_json_schema(),
            "doctor_report.schema.json": doctor_report_json_schema()}


def canonical_json(obj: Any) -> str:
    """Serialize stable, readable UTF-8 JSON text, rejecting nonfinite numbers."""
    return json.dumps(obj, sort_keys=True, indent=_JSON_INDENT_SPACES, ensure_ascii=False,
                      allow_nan=False) + "\n"


def _compact_json(obj: Any, *, allow_nan: bool = False) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=allow_nan)


def sha256_bytes(data: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash file bytes without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    """Hash compact sorted-key UTF-8 JSON, without a trailing newline."""
    return sha256_bytes(_compact_json(obj).encode("utf-8"))


def write_generated_schemas(directory: Path) -> list[Path]:
    """Write canonical schema files in filename order and return their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, document in sorted(generated_schemas().items()):
        path = directory / name
        path.write_text(canonical_json(document), encoding="utf-8")
        paths.append(path)
    return paths


def write_episode(episode: Episode, directory: Path) -> Path:
    """Create a new episode directory; preserve corrupt step numbers for Doctor."""
    episode_id = episode.manifest["episode_id"]
    if (not isinstance(episode_id, str) or not episode_id or episode_id in (".", "..")
            or Path(episode_id).name != episode_id):
        raise ValueError("episode_id must be a single directory name")
    manifest_text = canonical_json(episode.manifest)
    target = directory / episode_id
    target.mkdir(parents=True, exist_ok=False)
    (target / MANIFEST_FILENAME).write_text(manifest_text, encoding="utf-8")
    with (target / STEPS_FILENAME).open("w", encoding="utf-8") as stream:
        for step in episode.steps:
            stream.write(_compact_json(step, allow_nan=True) + "\n")
    return target


def _json_object(text: str) -> dict[str, Any]:
    value: Any = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def read_episode(path: Path) -> Episode:
    """Read JSON objects without schema validation; retain NaN for diagnostics."""
    try:
        manifest = _json_object((path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        with (path / STEPS_FILENAME).open(encoding="utf-8") as stream:
            steps = tuple(_json_object(line) for line in stream)
        return Episode(manifest, steps)
    except (OSError, ValueError) as error:
        raise TraceReadError(f"Cannot read episode {path}: {error}") from error


def list_episode_dirs(run_dir: Path) -> list[Path]:
    """List episode directories containing a manifest, sorted by directory name."""
    return sorted((path for path in run_dir.iterdir()
                   if path.is_dir() and (path / MANIFEST_FILENAME).is_file()), key=lambda p: p.name)


def write_run_manifest(manifest: dict[str, Any], run_dir: Path) -> Path:
    """Write the run's index as canonical JSON."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RUN_MANIFEST_FILENAME
    path.write_text(canonical_json(manifest), encoding="utf-8")
    return path


def read_run_manifest(run_dir: Path) -> dict[str, Any]:
    """Read a run index, translating unreadable or invalid objects to TraceReadError."""
    path = run_dir / RUN_MANIFEST_FILENAME
    try:
        return _json_object(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TraceReadError(f"Cannot read run manifest {path}: {error}") from error


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate schemas in the directory supplied on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    try:
        args = parser.parse_args(argv)
        write_generated_schemas(args.directory)
    except SystemExit as error:
        return 0 if error.code == 0 else _EXIT_USAGE_OR_INTERNAL
    except OSError as error:
        sys.stderr.write(f"Cannot write schemas: {error}\n")
        return _EXIT_INVALID_INPUT
    except Exception as error:
        sys.stderr.write(f"Schema generation failed: {error}\n")
        return _EXIT_USAGE_OR_INTERNAL
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
