"""Strict JSON comparison configs for paired policy evaluation.

A config names the task and simulator version, explicit condition/seed matrix,
exactly baseline/candidate variants, and the policy or override keys intended to
vary. Policies are names; overrides contain JSON scalars. The raw file digest
identifies the exact input bytes, including whitespace. No defaults invent runs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .traces import SCHEMA_VERSION, sha256_bytes

__all__ = [
    "config_object", "config_keys", "config_string", "config_list", "closed_object_schema",
    "ConfigError",
    "VariantSpec",
    "ComparisonConfig",
    "load_comparison_config",
    "comparison_config_json_schema",
]

_CONFIG_KEYS = (
    "schema_version",
    "name",
    "task",
    "simulator_version",
    "matrix",
    "variants",
    "intended_differences",
)
_VARIANT_NAMES = ("baseline", "candidate")
_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
_MINIMUM_SEED = 0
_MINIMUM_LIST_LENGTH = 1


class ConfigError(ValueError):
    """A comparison config is unreadable or violates its contract."""


@dataclass(frozen=True)
class VariantSpec:
    """Registered policy name and explicitly supplied scalar overrides."""

    policy: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class ComparisonConfig:
    """Validated explicit run matrix and the raw configuration file digest."""

    name: str
    task: str
    simulator_version: str
    conditions: tuple[str, ...]
    seeds: tuple[int, ...]
    variants: dict[str, VariantSpec]
    intended_differences: tuple[str, ...]
    sha256: str


def config_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: expected an object")
    return value


def config_keys(
    value: dict[str, Any], path: str, required: tuple[str, ...], optional: tuple[str, ...] = ()
) -> None:
    for key in required:
        if key not in value:
            raise ConfigError(f"{path}.{key}: missing required key")
    for key in sorted(value):
        if key not in required + optional:
            raise ConfigError(f"{path}.{key}: unknown key")


def config_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}: expected a non-empty string")
    return value


def config_list(value: Any, path: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ConfigError(f"{path}: expected {'a non-empty' if nonempty else 'a'} list")
    return value


def _conditions(value: Any) -> tuple[str, ...]:
    result: list[str] = []
    for index, item in enumerate(config_list(value, "matrix.conditions", nonempty=True)):
        path = f"matrix.conditions[{index}]"
        condition = config_string(item, path)
        if condition in result:
            raise ConfigError(f"{path}: duplicate condition")
        result.append(condition)
    return tuple(result)


def _seeds(value: Any) -> tuple[int, ...]:
    result: list[int] = []
    for index, seed in enumerate(config_list(value, "matrix.seeds", nonempty=True)):
        path = f"matrix.seeds[{index}]"
        if type(seed) is not int or seed < _MINIMUM_SEED:
            raise ConfigError(f"{path}: expected a non-negative integer (not bool)")
        if seed in result:
            raise ConfigError(f"{path}: duplicate seed")
        result.append(seed)
    return tuple(result)


def _variant(value: Any, path: str) -> VariantSpec:
    variant = config_object(value, path)
    config_keys(variant, path, ("policy",), ("overrides",))
    policy = config_string(variant["policy"], f"{path}.policy")
    overrides = config_object(variant.get("overrides", {}), f"{path}.overrides")
    for key, scalar in overrides.items():
        if not isinstance(scalar, (bool, int, float, str)) or (
            isinstance(scalar, float) and not math.isfinite(scalar)
        ):
            raise ConfigError(f"{path}.overrides.{key}: expected a finite JSON scalar")
    return VariantSpec(policy, overrides)


def _differences(value: Any, variants: dict[str, VariantSpec]) -> tuple[str, ...]:
    result: list[str] = []
    override_keys = {key for variant in variants.values() for key in variant.overrides}
    for index, item in enumerate(config_list(value, "intended_differences")):
        path = f"intended_differences[{index}]"
        entry = config_string(item, path)
        if entry != "policy" and not (
            entry.startswith("overrides.")
            and entry.removeprefix("overrides.") in override_keys
            and entry != "overrides."
        ):
            raise ConfigError(f"{path}: expected policy or overrides.<existing key>")
        if entry in result:
            raise ConfigError(f"{path}: duplicate intended difference")
        result.append(entry)
    return tuple(result)


def load_comparison_config(path: Path) -> ComparisonConfig:
    """Read and validate a config, reporting invalid keys with their full paths."""
    try:
        raw = path.read_bytes()
        document = config_object(json.loads(raw.decode("utf-8")), "comparison")
    except (OSError, UnicodeError, ValueError) as error:
        raise ConfigError(f"comparison {path}: {error}") from error
    config_keys(document, "comparison", _CONFIG_KEYS)
    if document["schema_version"] != SCHEMA_VERSION:
        raise ConfigError(f"schema_version: expected {SCHEMA_VERSION!r}")
    name = config_string(document["name"], "name")
    task = config_string(document["task"], "task")
    simulator_version = config_string(document["simulator_version"], "simulator_version")
    matrix = config_object(document["matrix"], "matrix")
    config_keys(matrix, "matrix", ("conditions", "seeds"))
    variant_objects = config_object(document["variants"], "variants")
    config_keys(variant_objects, "variants", _VARIANT_NAMES)
    variants = {key: _variant(variant_objects[key], f"variants.{key}") for key in _VARIANT_NAMES}
    return ComparisonConfig(
        name=name,
        task=task,
        simulator_version=simulator_version,
        conditions=_conditions(matrix["conditions"]),
        seeds=_seeds(matrix["seeds"]),
        variants=variants,
        intended_differences=_differences(document["intended_differences"], variants),
        sha256=sha256_bytes(raw),
    )


def closed_object_schema(properties: dict[str, Any], required: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _unique_array(items: dict[str, Any], *, nonempty: bool = False) -> dict[str, Any]:
    result = {"type": "array", "items": items, "uniqueItems": True}
    if nonempty:
        result["minItems"] = _MINIMUM_LIST_LENGTH
    return result


def comparison_config_json_schema() -> dict[str, Any]:
    """Build the config schema; the loader also checks override key references."""
    text = {"type": "string", "minLength": _MINIMUM_LIST_LENGTH, "pattern": r"\S"}
    variant = closed_object_schema(
        {
            "policy": text,
            "overrides": {
                "type": "object",
                "additionalProperties": {
                    "type": ["boolean", "integer", "number", "string"],
                },
            },
        },
        ("policy",),
    )
    matrix = closed_object_schema(
        {
            "conditions": _unique_array(text, nonempty=True),
            "seeds": _unique_array({"type": "integer", "minimum": _MINIMUM_SEED}, nonempty=True),
        },
        ("conditions", "seeds"),
    )
    body = closed_object_schema(
        {
            "schema_version": {"const": SCHEMA_VERSION},
            "name": text,
            "task": text,
            "simulator_version": text,
            "matrix": matrix,
            "variants": closed_object_schema(
                {key: variant for key in _VARIANT_NAMES}, _VARIANT_NAMES),
            "intended_differences": _unique_array(
                {
                    "type": "string",
                    "pattern": r"^(policy|overrides\..+)$",
                }
            ),
        },
        _CONFIG_KEYS,
    )
    return {"$schema": _SCHEMA_DIALECT, "$id": "comparison_config.schema.json", **body}
