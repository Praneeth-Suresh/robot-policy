"""Discover task packs through the `gsp_lab.tasks` entry-point group.

An installed package registers a task with an entry point whose name is the task
name and whose value is `module:Class`. Listing reads package metadata only and
never imports a pack; `get_task` imports exactly one pack. A pack that cannot be
loaded raises TaskLoadError for that name alone, so other tasks keep working.
`register` adds an in-process task for tests and the conformance kit.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from guided_streaming_policy.lab.tasks.base import CONTRACT_VERSION, TaskAdapter

__all__ = [
    "ENTRY_POINT_GROUP", "TaskEntry", "TaskLoadError", "entries", "get_task", "register",
    "task_names", "unregister",
]

ENTRY_POINT_GROUP = "gsp_lab.tasks"
_IN_PROCESS = "in-process"
_Factory = Callable[[], TaskAdapter]
_REGISTERED: dict[str, _Factory] = {}


class TaskLoadError(Exception):
    """A known task pack cannot be loaded; the message is the reason to show."""


@dataclass(frozen=True)
class TaskEntry:
    """A discoverable task: where it comes from and, when claimed twice, the conflict."""

    name: str
    target: str
    distribution: str
    version: str
    conflict: str | None = None


def register(name: str, factory: _Factory) -> None:
    """Add an in-process task; the name must not already be claimed."""
    claimed = {entry.name for entry in entries()}
    if name in claimed:
        raise ValueError(f"Task name {name!r} is already registered")
    _REGISTERED[name] = factory


def unregister(name: str) -> None:
    """Remove an in-process task added with `register`."""
    _REGISTERED.pop(name, None)


def _declared() -> list[TaskEntry]:
    """Read entry points from package metadata, without importing any pack."""
    found: list[TaskEntry] = []
    seen: set[str] = set()
    for dist in importlib.metadata.distributions():
        dist_name = str(dist.metadata["Name"] or "unknown")
        if dist_name.lower() in seen:
            continue
        seen.add(dist_name.lower())
        found.extend(
            TaskEntry(point.name, point.value, dist_name, dist.version)
            for point in dist.entry_points
            if point.group == ENTRY_POINT_GROUP
        )
    return found


def entries() -> tuple[TaskEntry, ...]:
    """Return every discoverable task by name; a doubly claimed name is marked conflicted."""
    by_name: dict[str, list[TaskEntry]] = {}
    for entry in _declared():
        by_name.setdefault(entry.name, []).append(entry)
    for name in _REGISTERED:
        by_name.setdefault(name, []).append(TaskEntry(name, _IN_PROCESS, _IN_PROCESS, "-"))
    result = []
    for name in sorted(by_name):
        claims = sorted(by_name[name], key=lambda c: (c.distribution.lower(), c.version))
        if len(claims) == 1:
            result.append(claims[0])
            continue
        owners = " and ".join(f"{c.distribution} {c.version}" for c in claims)
        result.append(TaskEntry(name, claims[0].target, claims[0].distribution,
                                claims[0].version, f"task {name!r} is claimed by {owners}"))
    return tuple(result)


def task_names() -> tuple[str, ...]:
    """Return discoverable task names in deterministic order."""
    return tuple(entry.name for entry in entries())


def _construct(entry: TaskEntry) -> object:
    if entry.name in _REGISTERED and entry.target == _IN_PROCESS:
        return _REGISTERED[entry.name]()
    module_name, _, attribute = entry.target.partition(":")
    try:
        constructor = getattr(importlib.import_module(module_name), attribute)
        return constructor()
    except Exception as error:  # a broken third-party pack must not crash the lab
        raise TaskLoadError(f"{type(error).__name__}: {error}") from error


def get_task(name: str) -> TaskAdapter:
    """Load one task pack, checking its contract version.

    Raise KeyError listing the known names for an unknown task, and TaskLoadError
    for a conflicted, unimportable or incompatible pack.
    """
    found = {entry.name: entry for entry in entries()}
    if name not in found:
        known = ", ".join(sorted(found)) or "(none)"
        raise KeyError(f"Unknown task {name!r}; known tasks: {known}")
    entry = found[name]
    if entry.conflict:
        raise TaskLoadError(entry.conflict)
    adapter = _construct(entry)
    version = getattr(adapter, "contract_version", None)
    if version != CONTRACT_VERSION:
        raise TaskLoadError(
            f"task {name!r} implements contract version {version!r}; "
            f"this lab requires version {CONTRACT_VERSION}"
        )
    declared = getattr(adapter, "name", None)
    if declared != name:
        raise TaskLoadError(f"task registered as {name!r} declares the name {declared!r}")
    return cast(TaskAdapter, adapter)
