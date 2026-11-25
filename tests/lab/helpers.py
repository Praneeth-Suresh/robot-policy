"""Deterministic synthetic trace builders shared by lab tests."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from guided_streaming_policy.lab import schema

__all__ = ["demo_episode", "rollout_episode", "write_traces"]

_DT_SECONDS = 0.1
_TIME_DECIMALS = 6
_CREATED_AT = "2026-01-01T00:00:00Z"


def _field(shape: list[int], units: str = "px") -> dict[str, Any]:
    return {"shape": shape, "dtype": "float64", "units": units, "frame": "window"}


def demo_episode(
    episode_id: str, n_steps: int = 6, offset: float = 0.0,
    split: str | None = None, split_source: str = "none",
) -> schema.Episode:
    """Build a complete demonstration with simple absolute target actions."""
    manifest = {
        "schema_version": schema.SCHEMA_VERSION, "kind": "demonstration",
        "episode_id": episode_id, "source_episode_id": episode_id, "task": "pusht",
        "dataset": {"name": "synthetic", "revision": "fixture-v1", "checksum": "fixture"},
        "split": split, "split_source": split_source, "synthetic": True,
        "timebase": {"unit": "s", "mode": "fixed_dt", "dt": _DT_SECONDS},
        "observation_schema": {"fields": {
            "agent_pos": _field([2]), "block_pos": _field([2]),
            "block_angle": _field([], "rad"),
        }},
        "action_schema": {"convention": "absolute", "fields": {"target": _field([2])}},
        "fields_present": ["obs", "action_executed"],
        "provenance": {"tool_version": "fixture", "git_commit": "fixture",
                       "command": "fixture builder", "created_at": _CREATED_AT},
    }
    steps = tuple({
        "step": index, "t": round(index * _DT_SECONDS, _TIME_DECIMALS),
        "obs": {"agent_pos": [offset + index, offset + index + 1.0],
                "block_pos": [offset + 10.0, offset + 20.0], "block_angle": 0.0},
        "action_executed": {"target": [offset + index + 2.0, offset + index + 3.0]},
    } for index in range(n_steps))
    return schema.Episode(manifest, steps)


def rollout_episode(
    episode_id: str, n_steps: int = 6, offset: float = 0.0,
    split: str | None = None, split_source: str = "none", *,
    policy: dict[str, Any] | None = None, outcome: dict[str, Any] | None = None,
    obstacle: dict[str, Any] | None = None, condition: str = "static", seed: int = 0,
) -> schema.Episode:
    """Build a complete rollout; supplied outcome values override clean defaults."""
    episode = demo_episode(episode_id, n_steps, offset, split, split_source)
    manifest = episode.manifest
    del manifest["dataset"]
    manifest.update({
        "kind": "rollout", "condition": condition, "seed": seed,
        "reset_state_fingerprint": "fixture-reset",
        "policy": policy if policy is not None else {
            "name": "fixture", "checkpoint_id": "fixture", "checkpoint_sha256": "fixture"},
        "resolved_config": {"path": "fixture.json", "sha256": "fixture"},
        "simulator": {"name": "fixture", "version": "1"},
        "outcome": {"native_success": False, "terminated": False, "truncated": True,
                    "reason": "step_limit", "length": n_steps, **(outcome or {})},
    })
    if obstacle is not None:
        manifest["geometry"] = {"obstacles": [obstacle]}
        manifest["fields_present"].append("geometry")
    return episode


def write_traces(directory: Path, episodes: Iterable[schema.Episode]) -> None:
    """Write each fixture episode below directory without replacing earlier traces."""
    directory.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        schema.write_episode(episode, directory)
