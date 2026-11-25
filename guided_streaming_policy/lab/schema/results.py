"""Gate configurations and result contracts, using only the standard library.

Success rules pass when candidate_rate - baseline_rate >= threshold over completed
pairs; a negative threshold tolerates that drop, and zero pairs means unknown.
Separation rules judge candidate episodes: any failure wins over unknown, then
unknown wins over pass. Baselines are measured for reporting only. Threshold and
tolerance share obstacle units; documentary rule units must match every obstacle.
Coverage below min_coverage yields unknown. Raw config bytes determine sha256.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConfigError, _closed_object, _keys, _list, _object, _string
from .traces import SCHEMA_VERSION, sha256_bytes

__all__ = [
    "GateRule",
    "GateConfig",
    "load_gate_config",
    "gate_config_json_schema",
    "gate_result_json_schema",
]

_METRICS = ("native_success_rate", "obstacle_separation_min")
_GATE_KEYS = (
    "schema_version",
    "name",
    "min_completed_pairs",
    "max_missing_pairs",
    "max_incompatible_pairs",
    "rules",
)
_RULE_KEYS = ("id", "metric", "units", "threshold", "required")
_SUCCESS_KEYS = ("compare",)
_SEPARATION_KEYS = ("signal", "tolerance", "min_coverage")
_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
_MINIMUM_PAIRS = 1
_MINIMUM_COUNT = 0
_FULL_COVERAGE = 1.0
_SIGNAL_PATTERN = r"^[^.\s]+\.[^.\s]+$"
_SUCCESS_DESCRIPTION = (
    "Pass iff candidate_rate - baseline_rate >= threshold over completed pairs. "
    "A negative threshold tolerates that drop; zero completed pairs means unknown."
)
_SEPARATION_DESCRIPTION = (
    "Candidate verdict: fail if any episode fails, else unknown if any is unknown, else pass. "
    "Distances below threshold - tolerance fail; insufficient min_coverage means unknown. "
    "Units must equal every obstacle's units or the episode is unknown (units mismatch). "
    "Baseline measurements and paired minimum deltas do not affect verdicts."
)


@dataclass(frozen=True)
class GateRule:
    """A validated metric rule; metric-specific fields are absent as None."""

    id: str
    metric: str
    units: str
    threshold: float
    required: bool
    compare: str | None = None
    signal: str | None = None
    tolerance: float | None = None
    min_coverage: float | None = None


@dataclass(frozen=True)
class GateConfig:
    """Explicit evidence limits, immutable rules, and digest of exact file bytes."""

    name: str
    min_completed_pairs: int
    max_missing_pairs: int
    max_incompatible_pairs: int
    rules: tuple[GateRule, ...]
    sha256: str


def _count(value: Any, path: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{path}: expected integer >= {minimum} (not bool)")
    return value


def _finite(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: expected a finite number (not bool)")
    try:
        result = float(value)
    except OverflowError as error:
        raise ConfigError(f"{path}: expected a finite number") from error
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ConfigError(
            f"{path}: expected finite number" + (f" >= {minimum}" if minimum is not None else "")
        )
    return result


def _rule(value: Any, path: str) -> GateRule:
    rule = _object(value, path)
    metric = rule.get("metric")
    if metric not in _METRICS:
        raise ConfigError(f"{path}.metric: supported metrics: {', '.join(_METRICS)}")
    _keys(rule, path, _RULE_KEYS + (_SUCCESS_KEYS if metric == _METRICS[0] else _SEPARATION_KEYS))
    identifier = _string(rule["id"], f"{path}.id")
    if not isinstance(rule["units"], str):
        raise ConfigError(f"{path}.units: expected a string")
    if type(rule["required"]) is not bool:
        raise ConfigError(f"{path}.required: expected a bool")
    if metric == _METRICS[0]:
        if rule["compare"] != "candidate_minus_baseline":
            raise ConfigError(f"{path}.compare: expected candidate_minus_baseline")
        return GateRule(
            identifier,
            metric,
            rule["units"],
            _finite(rule["threshold"], f"{path}.threshold"),
            rule["required"],
            compare=rule["compare"],
        )
    signal = rule["signal"]
    if not isinstance(signal, str) or re.fullmatch(_SIGNAL_PATTERN, signal) is None:
        raise ConfigError(f"{path}.signal: expected <group>.<field>")
    coverage = _finite(rule["min_coverage"], f"{path}.min_coverage")
    if not _MINIMUM_COUNT < coverage <= _FULL_COVERAGE:
        raise ConfigError(f"{path}.min_coverage: expected (0, 1]")
    return GateRule(
        identifier,
        metric,
        rule["units"],
        _finite(rule["threshold"], f"{path}.threshold", minimum=_MINIMUM_COUNT),
        rule["required"],
        signal=signal,
        tolerance=_finite(rule["tolerance"], f"{path}.tolerance", minimum=_MINIMUM_COUNT),
        min_coverage=coverage,
    )


def _rules(value: Any) -> tuple[GateRule, ...]:
    result: list[GateRule] = []
    identifiers = set()
    for index, item in enumerate(_list(value, "rules", nonempty=True)):
        rule = _rule(item, f"rules[{index}]")
        if rule.id in identifiers:
            raise ConfigError(f"rules[{index}].id: duplicate rule id {rule.id!r}")
        identifiers.add(rule.id)
        result.append(rule)
    return tuple(result)


def load_gate_config(path: Path) -> GateConfig:
    """Load strict JSON, raising ConfigError with the path of each invalid key."""
    try:
        raw = path.read_bytes()
        document = _object(json.loads(raw.decode("utf-8")), "gate")
    except (OSError, UnicodeError, ValueError) as error:
        raise ConfigError(f"gate {path}: {error}") from error
    _keys(document, "gate", _GATE_KEYS)
    if document["schema_version"] != SCHEMA_VERSION:
        raise ConfigError(f"schema_version: expected {SCHEMA_VERSION!r}")
    return GateConfig(
        name=_string(document["name"], "name"),
        min_completed_pairs=_count(
            document["min_completed_pairs"], "min_completed_pairs", _MINIMUM_PAIRS
        ),
        max_missing_pairs=_count(
            document["max_missing_pairs"], "max_missing_pairs", _MINIMUM_COUNT
        ),
        max_incompatible_pairs=_count(
            document["max_incompatible_pairs"], "max_incompatible_pairs", _MINIMUM_COUNT
        ),
        rules=_rules(document["rules"]),
        sha256=sha256_bytes(raw),
    )


def gate_config_json_schema() -> dict[str, Any]:
    """Build the closed metric-specific schema; the loader also enforces unique ids."""
    text = {"type": "string", "minLength": 1, "pattern": r"\S"}
    shared = {"id": text, "required": {"type": "boolean"}, "units": {"type": "string"}}
    success = _closed_object(
        {
            **shared,
            "metric": {"const": _METRICS[0]},
            "threshold": {"type": "number"},
            "compare": {"const": "candidate_minus_baseline"},
        },
        _RULE_KEYS + _SUCCESS_KEYS,
    )
    success["description"] = _SUCCESS_DESCRIPTION
    separation = _closed_object(
        {
            **shared,
            "metric": {"const": _METRICS[1]},
            "signal": {"type": "string", "pattern": _SIGNAL_PATTERN},
            "threshold": {"type": "number", "minimum": 0},
            "tolerance": {"type": "number", "minimum": 0},
            "min_coverage": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        },
        _RULE_KEYS + _SEPARATION_KEYS,
    )
    separation["description"] = _SEPARATION_DESCRIPTION
    return {
        "$schema": _SCHEMA_DIALECT,
        "$id": "gate_config.schema.json",
        **_closed_object(
            {
                "schema_version": {"const": SCHEMA_VERSION},
                "name": text,
                "min_completed_pairs": {"type": "integer", "minimum": _MINIMUM_PAIRS},
                "max_missing_pairs": {"type": "integer", "minimum": _MINIMUM_COUNT},
                "max_incompatible_pairs": {"type": "integer", "minimum": _MINIMUM_COUNT},
                "rules": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"oneOf": [success, separation]},
                },
            },
            _GATE_KEYS,
        ),
    }


def _result_rule_schema() -> dict[str, Any]:
    text = {"type": "string"}
    number = {"type": ["number", "null"]}
    count = {"type": "integer", "minimum": 0}
    props = {
        "id": text,
        "metric": {"enum": list(_METRICS)},
        "required": {"type": "boolean"},
        "verdict": {"enum": ["pass", "fail", "unknown"]},
        "units": text,
        "threshold": {"type": "number"},
        "tolerance": {"type": "number", "minimum": 0},
        **{key: number for key in ("baseline_value", "candidate_value", "delta")},
        **{key: count for key in ("samples", "wins", "losses", "ties")},
        "coverage": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": text,
        "signal": text,
    }
    result = _closed_object(
        props,
        tuple(key for key in props if key not in ("tolerance", "coverage", "reason", "signal")),
    )
    result["description"] = _SUCCESS_DESCRIPTION + " " + _SEPARATION_DESCRIPTION
    result["allOf"] = [
        {
            "if": {"properties": {"metric": {"const": _METRICS[1]}}},
            "then": {"required": ["tolerance", "coverage"]},
            "else": {"not": {"anyOf": [{"required": ["tolerance"]}, {"required": ["coverage"]}]}},
        },
        {"if": {"properties": {"verdict": {"const": "unknown"}}}, "then": {"required": ["reason"]}},
    ]
    return result


def _regression_schema() -> dict[str, Any]:
    text = {"type": "string"}
    count = {"type": "integer", "minimum": 0}
    category = {"enum": ["crashed", "truncated", "native_fail", "constraint_fail"]}
    entry = _closed_object({"episode_id": text, "path": text}, ("episode_id", "path"))
    props = {
        "key": _closed_object(
            {"task": text, "condition": text, "seed": {"type": "integer"}},
            ("task", "condition", "seed"),
        ),
        "category": category,
        "categories": {"type": "array", "items": category, "minItems": 1, "uniqueItems": True},
        "rule_ids": {"type": "array", "items": text, "uniqueItems": True},
        "first_violation": {
            "anyOf": [
                {"type": "null"},
                _closed_object(
                    {"step": count, "time": {"type": ["number", "null"]}}, ("step", "time")
                ),
            ]
        },
        "replay_step": count,
        "replay_basis": {"enum": ["first_violation", "episode_end"]},
        "baseline": entry,
        "candidate": entry,
    }
    return _closed_object(props, tuple(props))


def gate_result_json_schema() -> dict[str, Any]:
    """Build deterministic gate verdict, rule evidence, and regression replay contract."""
    text = {"type": "string"}
    identity = _closed_object({"name": text, "sha256": text}, ("name", "sha256"))
    coverage = (
        "expected",
        "completed",
        "incompatible",
        "missing",
        "min_completed_pairs",
        "max_missing_pairs",
        "max_incompatible_pairs",
    )
    props = {
        "schema_version": {"const": SCHEMA_VERSION},
        "gate": identity,
        "comparison": identity,
        "verdict": {"enum": ["PASS", "REGRESSION", "BLOCKED"]},
        "exit_code": {"enum": [0, 1, 2]},
        "blocked_reasons": {"type": "array", "items": text},
        "coverage": _closed_object(
            {key: {"type": "integer", "minimum": 0} for key in coverage}, coverage
        ),
        "rules": {"type": "array", "items": _result_rule_schema()},
        "regressions": {"type": "array", "items": _regression_schema()},
    }
    return {
        "$schema": _SCHEMA_DIALECT,
        "$id": "gate_result.schema.json",
        **_closed_object(props, tuple(props)),
        "allOf": [
            {
                "if": {"properties": {"verdict": {"const": verdict}}},
                "then": {"properties": {"exit_code": {"const": code}}},
            }
            for verdict, code in (("PASS", 0), ("REGRESSION", 1), ("BLOCKED", 2))
        ],
    }
