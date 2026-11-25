"""Discover task adapters lazily without importing research runners."""

import importlib
from typing import cast

from guided_streaming_policy.lab.tasks.base import TaskAdapter

__all__ = ["TASKS", "get_task", "task_names"]

TASKS: dict[str, str] = {
    "pusht": "guided_streaming_policy.lab.tasks.pusht.adapter:PushTTask",
}


def get_task(name: str) -> TaskAdapter:
    """Instantiate a task by dotted path or raise KeyError listing known names."""
    if name not in TASKS:
        raise KeyError(f"Unknown task {name!r}; known tasks: {', '.join(task_names())}")
    module_name, class_name = TASKS[name].split(":")
    constructor = getattr(importlib.import_module(module_name), class_name)
    return cast(TaskAdapter, constructor())


def task_names() -> tuple[str, ...]:
    """Return registered task names in deterministic order."""
    return tuple(sorted(TASKS))
