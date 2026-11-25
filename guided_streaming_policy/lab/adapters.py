"""Shared task metadata operations; configurations contain policy, entry_point and settings."""

import json
import shutil
from collections.abc import Iterator, Mapping, Sequence
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
    "read_traces", "is_trace_dir", "path_checksum", "write_report_file", "copy_tree_once",
]

_HASH_PREFIX_HEX_DIGITS = 12
_NAME_COLUMN_CHARACTERS = 16
_STATUS_COLUMN_CHARACTERS = 13
_PIN_COLUMN_CHARACTERS = 36


def read_traces(path: Path) -> Iterator[schema.Episode]:
    """Read an episode or a name-sorted directory of episodes, retaining bad telemetry."""
    try:
        directories = ([path] if (path / schema.MANIFEST_FILENAME).is_file()
                       else schema.list_episode_dirs(path))
        if not directories:
            raise schema.TraceReadError(f"No episodes found in {path}")
        for directory in directories:
            yield schema.read_episode(directory)
    except OSError as error:
        raise schema.TraceReadError(f"Cannot read traces {path}: {error}") from error


def is_trace_dir(path: Path) -> bool:
    """Return whether path contains an episode manifest or immediate episode directories."""
    return path.is_dir() and (
        (path / schema.MANIFEST_FILENAME).is_file() or bool(schema.list_episode_dirs(path))
    )


def path_checksum(path: Path) -> str:
    """Hash a file or sorted [relative POSIX path, content digest] directory entries."""
    if path.is_file():
        return schema.sha256_file(path)
    if not path.is_dir():
        raise schema.TraceReadError(f"Input does not exist: {path}")
    entries = sorted((file.relative_to(path).as_posix(), file)
                     for file in path.rglob("*") if file.is_file())
    return schema.sha256_json([[relative, schema.sha256_file(file)] for relative, file in entries])


def write_report_file(path: Path, text: str) -> None:
    """Create a UTF-8 report exclusively; identical existing bytes are left untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise FileExistsError(f"refusing to overwrite {path}") from None


def copy_tree_once(source: Path, target: Path) -> None:
    """Copy a tree, retaining identical files and refusing different existing bytes."""
    if not source.is_dir():
        raise FileNotFoundError(f"directory not found: {source}")
    target.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        destination = target / entry.name
        if entry.is_dir():
            copy_tree_once(entry, destination)
        else:
            try:
                with destination.open("xb") as stream, entry.open("rb") as original:
                    shutil.copyfileobj(original, stream)
            except FileExistsError:
                if destination.read_bytes() != entry.read_bytes():
                    raise FileExistsError(f"refusing to overwrite {destination}") from None


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
