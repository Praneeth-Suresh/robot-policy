"""Read original Push-T episodes without loading images or modifying native data.

The source has no revision or split metadata. Time is derived from the adapter's
control period; state/action telemetry, including non-finite values, is preserved.
"""

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from guided_streaming_policy.lab import schema
from guided_streaming_policy.lab.tasks.pusht.adapter import PushTTask

__all__ = ["read_demonstrations"]

_COMMAND_TEMPLATE = "gsp lab doctor {path}"
_SOURCE_REVISION = "unversioned (no revision metadata in the source)"
_STATE_DIMENSIONS = 5
_ACTION_DIMENSIONS = 2
_AGENT_POSITION = slice(0, 2)
_BLOCK_POSITION = slice(2, 4)
_BLOCK_ANGLE_INDEX = 4
_TIME_PRECISION_DECIMALS = 6


def _arrays(group: Any) -> tuple[Any, Any, list[int]]:
    np = importlib.import_module("numpy")

    state = group["data/state"]
    action = group["data/action"]
    ends = group["meta/episode_ends"][:]
    if len(state.shape) != 2 or state.shape[1] != _STATE_DIMENSIONS:
        raise ValueError(f"data/state must have shape (N, {_STATE_DIMENSIONS})")
    if action.shape != (state.shape[0], _ACTION_DIMENSIONS):
        raise ValueError(f"data/action must have shape (N, {_ACTION_DIMENSIONS}), matching state")
    if ends.ndim != 1 or not np.issubdtype(ends.dtype, np.integer):
        raise ValueError("meta/episode_ends must be a one-dimensional integer array")
    boundaries = [int(end) for end in ends]
    starts = [0] + boundaries[:-1]
    if not boundaries or any(end <= start for start, end in zip(starts, boundaries, strict=True)):
        raise ValueError("meta/episode_ends must be nonempty, positive and strictly increasing")
    if boundaries[-1] != state.shape[0]:
        raise ValueError("last meta/episode_ends must equal the number of state/action rows")
    return state, action, boundaries


def _steps(state: Any, action: Any, dt_seconds: float) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "step": index,
            "t": round(index * dt_seconds, _TIME_PRECISION_DECIMALS),
            "obs": {
                "agent_pos": [float(value) for value in row[_AGENT_POSITION]],
                "block_pos": [float(value) for value in row[_BLOCK_POSITION]],
                "block_angle": float(row[_BLOCK_ANGLE_INDEX]),
            },
            "action_executed": {"target": [float(value) for value in target]},
        }
        for index, (row, target) in enumerate(zip(state, action, strict=True))
    )


def _manifest(
    path: Path,
    index: int,
    checksum: str,
    provenance: dict[str, str],
    adapter: PushTTask,
) -> dict[str, Any]:
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "kind": schema.KIND_DEMONSTRATION,
        "episode_id": f"demo_{index:06d}",
        "source_episode_id": str(index),
        "dataset": {
            "name": path.name.removesuffix(".zarr.zip"),
            "revision": _SOURCE_REVISION,
            "checksum": checksum,
        },
        "task": adapter.name,
        "split": None,
        "split_source": "none",
        "timebase": adapter.timebase(),
        "observation_schema": adapter.observation_schema(),
        "action_schema": adapter.action_schema(),
        "fields_present": ["obs", "action_executed"],
        "provenance": dict(provenance),
    }


def read_demonstrations(path: Path, created_at: str | None = None) -> Iterator[schema.Episode]:
    """Yield native episodes lazily; raise TraceReadError for unreadable input.

    Hash the source and collect provenance once per iterator. ``created_at`` can
    fix the provenance time for deterministic consumers. Always close the store
    on exhaustion, failure or explicit iterator closure.
    """
    group = None
    try:
        zarr = importlib.import_module("zarr")

        group = zarr.open(str(path), "r")
        state, action, boundaries = _arrays(group)
        checksum = schema.sha256_file(path)
        provenance = schema.make_provenance(_COMMAND_TEMPLATE.format(path=path), created_at)
        adapter = PushTTask()
        dt_seconds = float(adapter.timebase()["dt"])
        start = 0
        for index, end in enumerate(boundaries):
            steps = _steps(state[start:end], action[start:end], dt_seconds)
            yield schema.Episode(_manifest(path, index, checksum, provenance, adapter), steps)
            start = end
    except Exception as error:
        raise schema.TraceReadError(f"Cannot read demonstrations {path}: {error}") from error
    finally:
        if group is not None:
            group.store.close()
