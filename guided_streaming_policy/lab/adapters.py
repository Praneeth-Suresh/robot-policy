"""Shared task metadata operations; configurations contain policy, entry_point and settings."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from guided_streaming_policy.lab import schema
from guided_streaming_policy.lab.tasks.base import PolicySpec, PolicyStatus, TaskAdapter

__all__ = [
    "resolve_policy_config",
    "config_sha256",
    "load_pins",
    "find_policy",
    "format_policy_rows",
]

_HASH_PREFIX_HEX_DIGITS = 12
_NAME_COLUMN_CHARACTERS = 16
_STATUS_COLUMN_CHARACTERS = 13
_PIN_COLUMN_CHARACTERS = 36


def resolve_policy_config(spec: PolicySpec, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Merge declared tunable defaults with overrides of compatible scalar types."""
    settings = {tunable.name: tunable.default for tunable in spec.tunables}
    for key, value in overrides.items():
        if key not in settings:
            valid = ", ".join(sorted(settings)) or "(none)"
            raise ValueError(f"Unknown tunable {key!r}; valid keys: {valid}")
        default = settings[key]
        allowed = (float, int) if type(default) is float else (type(default),)
        if type(value) not in allowed:
            raise ValueError(f"Invalid type for tunable {key!r}; expected {type(default).__name__}")
        settings[key] = value
    return {"policy": spec.name, "entry_point": spec.entry_point, "settings": settings}


def config_sha256(resolved: Mapping[str, Any]) -> str:
    """Hash a resolved configuration using the shared canonical JSON digest."""
    return schema.sha256_json(dict(resolved))


def load_pins(path: Path) -> dict[str, Any]:
    """Read a JSON pin document whose tasks map contains per-task pin sections."""
    document: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return document


def find_policy(adapter: TaskAdapter, name: str) -> PolicySpec:
    """Find a declared policy or raise KeyError listing registered names."""
    policies = adapter.policies()
    for spec in policies:
        if spec.name == name:
            return spec
    names = ", ".join(sorted(spec.name for spec in policies))
    raise KeyError(f"Unknown policy {name!r}; registered policies: {names}")


def _pin_label(spec: PolicySpec, pins_for_task: Mapping[str, Any]) -> str:
    pin = pins_for_task.get("policies", {}).get(spec.name)
    if not pin or not spec.checkpoints:
        return "no pin"
    checkpoint = pin.get("checkpoints", {}).get(spec.checkpoints[0])
    config = pin.get("config_sha256")
    if not checkpoint or not config:
        return "no pin"
    return f"ckpt {checkpoint[:_HASH_PREFIX_HEX_DIGITS]} cfg {config[:_HASH_PREFIX_HEX_DIGITS]}"


def _row(name: str, status: str, pin: str, paper: str) -> str:
    return (
        f"{name:<{_NAME_COLUMN_CHARACTERS}} {status:<{_STATUS_COLUMN_CHARACTERS}} "
        f"{pin:<{_PIN_COLUMN_CHARACTERS}} {paper}"
    )


def format_policy_rows(
    statuses: Sequence[PolicyStatus], pins_for_task: Mapping[str, Any]
) -> list[str]:
    """Format policies by name with fixed columns and indented unavailable causes."""
    lines = [_row("NAME", "STATUS", "PIN", "PAPER")]
    for status in sorted(statuses, key=lambda item: item.spec.name):
        spec = status.spec
        lines.append(_row(spec.name, status.status, _pin_label(spec, pins_for_task), spec.paper))
        if status.status == "unavailable" and status.reason:
            lines.append(f"  - {status.reason}")
    return lines
