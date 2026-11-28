"""Deterministic pairing of recorded runs against an explicit comparison matrix.

Resolved config paths are policy, entry_point and settings.<key>. An intended
policy change covers all of them; overrides.<key> covers settings.<key> only.
No run or episode files are written, and no missing telemetry is reconstructed.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from guided_streaming_policy.lab import schema

__all__ = ["build_comparison", "compare_exit_code", "render_summary",
           "load_comparison_result", "verify_trace_hashes", "comparison_errors"]

_VARIANTS = ("baseline", "candidate")
_HASH_PREFIX_CHARACTERS = 12
_EXIT_OK = 0
_EXIT_INVALID_INPUT = 2
_Key = tuple[str, str, int]
_RecordedEpisode = tuple[Path, schema.Episode]


@dataclass(frozen=True)
class _Run:
    directory: Path
    manifest: dict[str, Any]
    resolved: dict[str, Any]
    resolved_sha256: str
    episodes: dict[_Key, list[_RecordedEpisode]]
    invalid_reasons: list[str]


def _relative(path: Path, out_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), out_dir.resolve())).as_posix()


def _read_resolved(directory: Path) -> dict[str, Any]:
    path = directory / "resolved_config.json"
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value["settings"], dict):
            raise ValueError("expected a resolved config object with settings")
        if not all(isinstance(value[key], str) for key in ("policy", "entry_point")):
            raise ValueError("expected policy and entry_point strings")
        schema.sha256_json(value)
        return value
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise schema.TraceReadError(f"Cannot read resolved config {path}: {error}") from error


def _run_reasons(
    manifest: dict[str, Any], variant: str, config: schema.ComparisonConfig,
) -> list[str]:
    checks = (
        (manifest.get("comparison", {}).get("sha256") != config.sha256, "comparison hash"),
        (manifest["task"] != config.task, "task"),
        (manifest.get("variant") != variant, "variant"),
        (manifest["policy"]["name"] != config.variants[variant].policy, "policy name"),
    )
    return [f"{variant} run {field} differs" for differs, field in checks if differs]


def _read_run(directory: Path, variant: str, config: schema.ComparisonConfig) -> _Run:
    manifest = schema.read_run_manifest(directory)
    resolved = _read_resolved(directory)
    digest = schema.sha256_json(resolved)
    reasons = _run_reasons(manifest, variant, config)
    episodes: dict[_Key, list[_RecordedEpisode]] = {}
    for path in schema.list_episode_dirs(directory):
        episode = schema.read_episode(path)
        metadata = episode.manifest
        key = (metadata["task"], metadata["condition"], metadata["seed"])
        episodes.setdefault(key, []).append((path, episode))
        if metadata["resolved_config"]["sha256"] != digest:
            reasons.append(
                f"{variant} episode {metadata['episode_id']} resolved config hash differs")
    return _Run(directory, manifest, resolved, digest, episodes, reasons)


def _flatten(resolved: dict[str, Any]) -> dict[str, Any]:
    return {**{key: value for key, value in resolved.items() if key != "settings"},
            **{f"settings.{key}": value for key, value in resolved["settings"].items()}}


def _intended(path: str, differences: tuple[str, ...]) -> bool:
    if "policy" in differences and (path in ("policy", "entry_point")
                                    or path.startswith("settings.")):
        return True
    override = f"overrides.{path.removeprefix('settings.')}"
    return path.startswith("settings.") and override in differences


def _configuration_reasons(
    baseline: _Run, candidate: _Run, config: schema.ComparisonConfig,
) -> list[str]:
    first, second = _flatten(baseline.resolved), _flatten(candidate.resolved)
    reasons = []
    for path in sorted(first.keys() | second.keys()):
        differs = (path not in first or path not in second
                   or schema.canonical_json(first[path]) != schema.canonical_json(second[path]))
        if differs and not _intended(path, config.intended_differences):
            reasons.append(f"unintended configuration difference: {path}")
    return reasons


def _entry(recorded: _RecordedEpisode | None, out_dir: Path) -> dict[str, Any] | None:
    if recorded is None:
        return None
    directory, episode = recorded
    manifest = episode.manifest
    return {"episode_id": manifest["episode_id"], "path": _relative(directory, out_dir),
            "manifest_sha256": schema.sha256_file(directory / schema.MANIFEST_FILENAME),
            "steps_sha256": schema.sha256_file(directory / schema.STEPS_FILENAME),
            **{key: manifest["outcome"][key] for key in (
                "native_success", "terminated", "truncated", "reason", "length")},
            "reset_state_fingerprint": manifest["reset_state_fingerprint"]}


def _compatibility_reasons(
    baseline: schema.Episode, candidate: schema.Episode, config: schema.ComparisonConfig,
) -> list[str]:
    first, second = baseline.manifest, candidate.manifest
    reasons = []
    fingerprints = (first["reset_state_fingerprint"], second["reset_state_fingerprint"])
    if None in fingerprints:
        reasons.append("reset state not recorded")
    elif fingerprints[0] != fingerprints[1]:
        reasons.append("reset state fingerprints differ")
    identities = [(m["simulator"]["name"], m["simulator"]["version"]) for m in (first, second)]
    if identities[0] != identities[1]:
        reasons.append("simulator identities differ")
    if any(version != config.simulator_version for _, version in identities):
        reasons.append("simulator versions differ from comparison")
    return reasons


def _pair(
    key: _Key, runs: dict[str, _Run], config: schema.ComparisonConfig, out_dir: Path,
) -> dict[str, Any]:
    matches = {variant: run.episodes.get(key, []) for variant, run in runs.items()}
    recorded = {variant: items[0] if items else None for variant, items in matches.items()}
    reasons = [f"missing episode in {variant} run" for variant in _VARIANTS
               if not matches[variant]]
    status = "missing"
    if not reasons:
        reasons.extend(f"duplicate episode for key in {variant} run" for variant in _VARIANTS
                       if len(matches[variant]) > 1)
        reasons.extend(_compatibility_reasons(matches["baseline"][0][1],
                                              matches["candidate"][0][1], config))
        for variant in _VARIANTS:
            if any(episode.manifest["resolved_config"]["sha256"] != runs[variant].resolved_sha256
                   for _, episode in matches[variant]):
                reasons.append(f"{variant} episode resolved config hash differs")
        status = "incompatible" if reasons else "completed"
    return {"key": {"task": key[0], "condition": key[1], "seed": key[2]}, "status": status,
            "reasons": reasons, **{variant: _entry(recorded[variant], out_dir)
                                    for variant in _VARIANTS}}


def _delta(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = [(pair["baseline"]["native_success"] is True,
                 pair["candidate"]["native_success"] is True)
                for pair in pairs if pair["status"] == "completed"]
    count = len(outcomes)
    baseline = sum(first for first, _ in outcomes)
    candidate = sum(second for _, second in outcomes)
    first_rate = baseline / count if count else None
    second_rate = candidate / count if count else None
    return {"n": count, "baseline_successes": baseline, "candidate_successes": candidate,
            "baseline_rate": first_rate, "candidate_rate": second_rate,
            "delta": second_rate - first_rate if second_rate is not None
            and first_rate is not None else None,
            "wins": sum(second and not first for first, second in outcomes),
            "losses": sum(first and not second for first, second in outcomes),
            "ties": sum(first == second for first, second in outcomes),
            "note": "native_success null (including crashed trials) counts as not successful."}


def _document(
    runs: dict[str, _Run], config: schema.ComparisonConfig, out_dir: Path,
) -> dict[str, Any]:
    keys = [(config.task, condition, seed)
            for condition in config.conditions for seed in config.seeds]
    pairs = [_pair(key, runs, config, out_dir) for key in keys]
    coverage = {status: sum(pair["status"] == status for pair in pairs)
                for status in ("completed", "incompatible", "missing")}
    coverage.update(expected=len(keys), extra_episodes=sum(
        len(episodes) for run in runs.values() for key, episodes in run.episodes.items()
        if key not in set(keys)))
    reasons = [reason for run in runs.values() for reason in run.invalid_reasons]
    reasons.extend(_configuration_reasons(runs["baseline"], runs["candidate"], config))
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "valid": not reasons,
        "invalid_reasons": reasons,
        "comparison": {"name": config.name, "sha256": config.sha256, "task": config.task,
                       "simulator_version": config.simulator_version},
        "runs": {variant: {"path": _relative(run.directory, out_dir),
                           "run_id": run.manifest["run_id"], "policy": run.manifest["policy"],
                           "resolved_config_sha256": run.resolved_sha256}
                 for variant, run in runs.items()},
        "coverage": coverage, "pairs": pairs, "deltas": {"native_success": _delta(pairs)},
    }


def build_comparison(
    baseline_dir: Path, candidate_dir: Path, config: schema.ComparisonConfig, *, out_dir: Path,
) -> dict[str, Any]:
    """Read both runs and return a stable comparison, retaining incomplete/failed cells.

    Invalid run identities, unintended configuration changes and episode hash
    mismatches are recorded as invalid reasons. Missing or incompatible cells
    also make the document invalid. Unreadable or incomplete JSON raises TraceReadError.
    """
    try:
        runs = {variant: _read_run(path, variant, config) for variant, path in
                zip(_VARIANTS, (baseline_dir, candidate_dir), strict=True)}
        return _document(runs, config, out_dir)
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as error:
        raise schema.TraceReadError(f"Cannot compare run inputs: {error}") from error


def compare_exit_code(document: dict[str, Any]) -> int:
    """Return 0 for valid comparisons and 2 for invalid or incomplete comparisons."""
    return _EXIT_OK if document["valid"] else _EXIT_INVALID_INPUT


def render_summary(document: dict[str, Any]) -> str:
    """Render identity, coverage, pair reasons, native success delta and validity."""
    comparison = document["comparison"]
    lines = [f"Policy Trace Lab comparison: {comparison['name']} "
             f"({comparison['sha256'][:_HASH_PREFIX_CHARACTERS]})"]
    lines.append("coverage: " + ", ".join(f"{key}={document['coverage'][key]}" for key in
                 ("expected", "completed", "incompatible", "missing", "extra_episodes")))
    for pair in document["pairs"]:
        for reason in pair["reasons"]:
            key = pair["key"]
            lines.append(f"{pair['status']} [{key['condition']} seed={key['seed']}]: {reason}")
    delta = document["deltas"]["native_success"]
    lines.append(f"native success: baseline={delta['baseline_rate']} "
                 f"candidate={delta['candidate_rate']} delta={delta['delta']} (n={delta['n']})")
    lines.append(f"valid: {str(document['valid']).lower()}")
    return "\n".join(lines)


def load_comparison_result(path: Path) -> dict[str, Any]:
    """Read comparison JSON, rejecting unreadable files and unsupported versions."""
    try:
        document: Any = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(document, dict)
                or document.get("schema_version") != schema.SCHEMA_VERSION):
            raise ValueError(f"schema_version: expected {schema.SCHEMA_VERSION!r}")
        schema.canonical_json(document)
        return document
    except (OSError, UnicodeError, ValueError) as error:
        raise schema.TraceReadError(f"Cannot read comparison {path}: {error}") from error


def verify_trace_hashes(document: dict[str, Any], base_dir: Path) -> list[str]:
    """Verify manifest and step file digests for both sides of completed pairs."""
    messages = []
    for pair in document["pairs"]:
        if pair["status"] != "completed":
            continue
        for variant in _VARIANTS:
            entry = pair[variant]
            for filename, key in ((schema.MANIFEST_FILENAME, "manifest_sha256"),
                                  (schema.STEPS_FILENAME, "steps_sha256")):
                path = base_dir / entry["path"] / filename
                try:
                    if schema.sha256_file(path) != entry[key]:
                        messages.append(f"trace hash mismatch: {variant} {path}")
                except OSError as error:
                    messages.append(f"trace hash unreadable: {variant} {path}: {error}")
    return messages


def _shape_type(value: Any, kind: str) -> bool:
    types: dict[str, type[Any] | tuple[type[Any], ...]] = {
        "object": dict, "array": list, "string": str, "integer": int,
        "number": (int, float), "boolean": bool, "null": type(None),
    }
    return isinstance(value, types[kind]) and not (
        kind in ("integer", "number") and isinstance(value, bool)
    )


def _shape(value: Any, spec: dict[str, Any], path: str) -> None:
    """Validate the comparison schema's closed structural subset without dependencies."""
    if "anyOf" in spec:
        if not any(_matches(value, option) for option in spec["anyOf"]):
            raise ValueError(f"{path}: invalid comparison entry")
        return
    kinds = spec.get("type", [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    if kinds and not any(_shape_type(value, kind) for kind in kinds):
        raise ValueError(f"{path}: expected {kinds}")
    _literal_shape(value, spec, path)
    if isinstance(value, dict):
        _object_shape(value, spec, path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _shape(item, spec["items"], f"{path}[{index}]")
    elif type(value) in (int, float):
        _number_shape(value, spec, path)


def _literal_shape(value: Any, spec: dict[str, Any], path: str) -> None:
    if "const" in spec and value != spec["const"]:
        raise ValueError(f"{path}: unsupported value {value!r}")
    if "enum" in spec and value not in spec["enum"]:
        raise ValueError(f"{path}: unsupported value {value!r}")


def _matches(value: Any, spec: dict[str, Any]) -> bool:
    try:
        _shape(value, spec, "entry")
        return True
    except ValueError:
        return False


def _object_shape(value: dict[str, Any], spec: dict[str, Any], path: str) -> None:
    missing = set(spec["required"]) - value.keys()
    unknown = value.keys() - spec["properties"].keys()
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {sorted(missing)}; unknown keys {sorted(unknown)}")
    for key, item in value.items():
        _shape(item, spec["properties"][key], f"{path}.{key}")


def _number_shape(value: float, spec: dict[str, Any], path: str) -> None:
    if not math.isfinite(value) or not spec.get("minimum", -math.inf) <= value <= spec.get(
        "maximum", math.inf
    ):
        raise ValueError(f"{path}: invalid comparison number")


def comparison_errors(document: dict[str, Any]) -> list[str]:
    """Return structural and consistency problems of a recorded comparison document."""
    try:
        _shape(document, schema.compare_result_json_schema(), "comparison")
        counts = document["coverage"]
        for status in ("completed", "incompatible", "missing"):
            if counts[status] != sum(p["status"] == status for p in document["pairs"]):
                raise ValueError(f"coverage.{status}: does not match pairs")
        if counts["expected"] != len(document["pairs"]):
            raise ValueError("coverage.expected: does not match pairs")
        if any(
            p[side] is None
            for p in document["pairs"]
            if p["status"] == "completed"
            for side in _VARIANTS
        ):
            raise ValueError("completed pair has an absent episode")
        if document["deltas"]["native_success"]["n"] != counts["completed"]:
            raise ValueError("deltas.native_success.n: does not match completed pairs")
        if not document["valid"]:
            return document["invalid_reasons"] or (
                ["comparison is not valid"] if counts["completed"] == counts["expected"] else [])
        return list(document["invalid_reasons"])
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        return [f"invalid comparison: {error}"]
