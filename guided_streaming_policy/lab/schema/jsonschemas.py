"""JSON Schema builders and deterministic committed-artifact generation.

Regenerate with: .venv/bin/python -m guided_streaming_policy.lab.schema schemas
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import comparison_config_json_schema
from .traces import (
    _COMMON_FIELDS,
    FIELDS_PRESENT_VALUES,
    KIND_DEMONSTRATION,
    KIND_ROLLOUT,
    SCHEMA_VERSION,
    SEVERITIES,
    canonical_json,
    required_manifest_fields,
)

__all__ = [
    "manifest_json_schema",
    "step_json_schema",
    "run_manifest_json_schema",
    "doctor_report_json_schema",
    "compare_result_json_schema",
    "generated_schemas",
    "write_generated_schemas",
    "main",
]

_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
_EXIT_INVALID_INPUT = 2
_EXIT_USAGE_OR_INTERNAL = 3


def _str() -> dict[str, Any]:
    return {"type": "string"}


def _obj(props: dict[str, Any], required: Sequence[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": list(props if required is None else required),
        "additionalProperties": False,
    }


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
    result = _obj(
        {
            "unit": _str(),
            "mode": {"enum": ["fixed_dt", "timestamps"]},
            "dt": {"type": "number", "exclusiveMinimum": 0},
        },
        ("unit", "mode"),
    )
    result["allOf"] = [
        {"if": {"properties": {"mode": {"const": "fixed_dt"}}}, "then": {"required": ["dt"]}}
    ]
    return result


def _field_schema(action: bool = False) -> dict[str, Any]:
    field = _obj(
        {
            "shape": _array({"type": "integer", "minimum": 0}),
            "dtype": _str(),
            "units": _str(),
            "frame": _str(),
        }
    )
    props: dict[str, Any] = {"fields": {"type": "object", "additionalProperties": field}}
    if action:
        props["convention"] = {"enum": ["absolute", "delta"]}
    return _obj(props)


def _outcome() -> dict[str, Any]:
    return _obj(
        {
            "native_success": {"type": ["boolean", "null"]},
            "terminated": {"type": "boolean"},
            "truncated": {"type": "boolean"},
            "reason": _str(),
            "length": {"type": "integer", "minimum": 0},
            "detail": _str(),
            "native": {"type": "object"},
        },
        ("native_success", "terminated", "truncated", "reason", "length"),
    )


def _geometry() -> dict[str, Any]:
    obstacle = _obj(
        {
            "id": _str(),
            "kind": _str(),
            "position": _array({"type": "number"}),
            "radius": {"type": "number", "minimum": 0},
            "units": _str(),
            "frame": _str(),
            "source": _str(),
        }
    )
    return _obj(
        {
            "obstacles": _array(obstacle),
            "goal": {"type": "object"},
            "block": {"type": "object"},
            "arena": {"type": "object"},
        },
        ("obstacles",),
    )


def manifest_json_schema() -> dict[str, Any]:
    """Build the episode manifest schema with conditional kind requirements."""
    props = {
        "schema_version": {"const": SCHEMA_VERSION},
        "kind": {"enum": [KIND_DEMONSTRATION, KIND_ROLLOUT]},
        "episode_id": _str(),
        "source_episode_id": _str(),
        "task": _str(),
        "split": {"type": ["string", "null"]},
        "split_source": _str(),
        "dataset": _strings("name", "revision", "checksum"),
        "timebase": _timebase(),
        "observation_schema": _field_schema(),
        "action_schema": _field_schema(action=True),
        "fields_present": {**_array({"enum": list(FIELDS_PRESENT_VALUES)}), "uniqueItems": True},
        "provenance": _provenance(),
        "media": _array(_strings("kind", "uri")),
        "condition": _str(),
        "seed": {"type": "integer"},
        "reset_state_fingerprint": {"type": ["string", "null"]},
        "policy": _policy(),
        "resolved_config": _strings("path", "sha256"),
        "simulator": _strings("name", "version"),
        "outcome": _outcome(),
        "geometry": _geometry(),
        "synthetic": {"type": "boolean", "default": False},
    }
    body = _obj(props, _COMMON_FIELDS)
    body["allOf"] = [
        {
            "if": {"properties": {"kind": {"const": kind}}, "required": ["kind"]},
            "then": {"required": list(required_manifest_fields(kind))},
        }
        for kind in (KIND_DEMONSTRATION, KIND_ROLLOUT)
    ]
    return _document("episode_manifest.schema.json", body)


def step_json_schema() -> dict[str, Any]:
    """Build a step schema; numeric arrays recurse, uncollected keys may be absent."""
    values = {"type": "object", "additionalProperties": {"$ref": "#/$defs/value"}}
    body = _obj(
        {
            "step": {"type": "integer", "minimum": 0},
            "t": {"type": "number"},
            "obs": values,
            "action_executed": values,
            "action_proposed": values,
            "reward": {"type": "number"},
            "info": {"type": "object"},
        },
        ("step", "t"),
    )
    body["$defs"] = {"value": {"anyOf": [{"type": "number"}, _array({"$ref": "#/$defs/value"})]}}
    return _document("episode_step.schema.json", body)


def run_manifest_json_schema() -> dict[str, Any]:
    """Build a run index with optional comparison digest and variant label."""
    episode = _obj({"episode_id": _str(), "condition": _str(), "seed": {"type": "integer"}})
    props = {
        "schema_version": {"const": SCHEMA_VERSION},
        "run_id": _str(),
        "task": _str(),
        "comparison": _strings("name", "sha256"),
        "variant": _str(),
        "policy": _policy(),
        "simulator": _strings("name", "version"),
        "provenance": _provenance(),
        "episodes": _array(episode),
    }
    required = ("schema_version", "run_id", "task", "policy", "simulator", "provenance", "episodes")
    return _document("run_manifest.schema.json", _obj(props, required))


def doctor_report_json_schema() -> dict[str, Any]:
    """Build Doctor's counts, executed/skipped checks, and ordered findings contract."""
    finding = _obj(
        {
            "check": _str(),
            "severity": {"enum": list(SEVERITIES)},
            "episode_id": _str(),
            "step": {"type": "integer", "minimum": 0},
            "field": _str(),
            "message": _str(),
            "hint": _str(),
        },
        ("check", "severity", "episode_id", "message", "hint"),
    )
    summary = _obj(
        {
            "counts": _obj({key: {"type": "integer", "minimum": 0} for key in SEVERITIES}),
            "checks_run": _array(_str()),
            "checks_skipped": _array(_strings("check", "reason")),
        }
    )
    return _document(
        "doctor_report.schema.json",
        _obj(
            {
                "schema_version": {"const": SCHEMA_VERSION},
                "input": _strings("path", "checksum"),
                "summary": summary,
                "findings": _array(finding),
            }
        ),
    )



def _compare_entry() -> dict[str, Any]:
    return _obj({
        **{key: _str() for key in ("episode_id", "path", "manifest_sha256", "steps_sha256")},
        "native_success": {"type": ["boolean", "null"]},
        "terminated": {"type": "boolean"}, "truncated": {"type": "boolean"},
        "reason": _str(), "length": {"type": "integer", "minimum": 0},
        "reset_state_fingerprint": {"type": ["string", "null"]},
    })


def _compare_delta() -> dict[str, Any]:
    counts = ("n", "baseline_successes", "candidate_successes", "wins", "losses", "ties")
    return _obj({
        **{key: {"type": "integer", "minimum": 0} for key in counts},
        **{key: {"type": ["number", "null"], "minimum": 0, "maximum": 1}
           for key in ("baseline_rate", "candidate_rate")},
        "delta": {"type": ["number", "null"], "minimum": -1, "maximum": 1},
        "note": _str(),
    })


def compare_result_json_schema() -> dict[str, Any]:
    """Build the closed comparison result contract, including partial matrix coverage."""
    entry = {"anyOf": [_compare_entry(), {"type": "null"}]}
    pair = _obj({
        "key": _obj({"task": _str(), "condition": _str(), "seed": {"type": "integer"}}),
        "status": {"enum": ["completed", "incompatible", "missing"]},
        "reasons": _array(_str()), "baseline": entry, "candidate": entry,
    })
    run = _obj({"path": _str(), "run_id": _str(), "policy": _policy(),
                "resolved_config_sha256": _str()})
    coverage_keys = ("expected", "completed", "incompatible", "missing", "extra_episodes")
    return _document("compare_result.schema.json", _obj({
        "schema_version": {"const": SCHEMA_VERSION}, "valid": {"type": "boolean"},
        "invalid_reasons": _array(_str()),
        "comparison": _strings("name", "sha256", "task", "simulator_version"),
        "runs": _obj({"baseline": run, "candidate": run}),
        "coverage": _obj({key: {"type": "integer", "minimum": 0} for key in coverage_keys}),
        "pairs": _array(pair), "deltas": _obj({"native_success": _compare_delta()}),
    }))


def generated_schemas() -> dict[str, dict[str, Any]]:
    """Map committed schema filenames to fresh schema documents."""
    return {
        "episode_manifest.schema.json": manifest_json_schema(),
        "episode_step.schema.json": step_json_schema(),
        "run_manifest.schema.json": run_manifest_json_schema(),
        "doctor_report.schema.json": doctor_report_json_schema(),
        "comparison_config.schema.json": comparison_config_json_schema(),
        "compare_result.schema.json": compare_result_json_schema(),
    }


def write_generated_schemas(directory: Path) -> list[Path]:
    """Write canonical schema files in filename order and return their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, document in sorted(generated_schemas().items()):
        path = directory / name
        path.write_text(canonical_json(document), encoding="utf-8")
        paths.append(path)
    return paths


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
