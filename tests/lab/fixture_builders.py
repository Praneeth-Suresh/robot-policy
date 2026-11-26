"""Generate small synthetic examples, never benchmark results; refuse nonempty output.

Comparison JSON records the explicit matrix and intended override difference. Gate
JSON uses the documented success delta and executed-target separation rules.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from guided_streaming_policy.lab import adapters, schema
from guided_streaming_policy.lab.tasks.pusht.adapter import PushTTask

__all__ = ["FIRST_VIOLATION_STEP", "build_fixtures", "main"]

FIRST_VIOLATION_STEP = 12
_STEPS = 40
_DECIMALS = 3
_SEEDS = (0, 1, 2, 3)
_SCENARIOS = ("seeded_regression", "two_valid_trajectories", "missing_signal",
              "mismatched_reset", "truncated", "missing_pairs")
_FRAME = "window_512_y_down"
_OBSTACLE_PX = [186.65, 146.5]
_OBSTACLE_RADIUS_PX = 15.36
_ARENA_SIZE_PX = 512
_GOAL_PX = 256
_GOAL_ANGLE_RAD = round(math.pi / 4, _DECIMALS)
_SAFE_START_PX = 300.0
_VARIANT_OFFSET_PX = 40.0
_SEED_OFFSET_PX = 5.0
_STEP_DISPLACEMENT_PX = 0.5
_BLOCK_POSITION_PX = [256.0, 240.0]
_ANGLE_STEP_RAD = 0.01
_VIOLATION_DISTANCE_PX = 10.0
_NON_MONOTONIC_OFFSET_SECONDS = 0.15
_NAN_STEP = 3
_SHAPE_STEP = 2
_TIME_STEP = 4
_REGRESSION_SEED = 2
_TRUNCATED_SEED = 1
_SUCCESS_THRESHOLD = -0.1
_SEPARATION_THRESHOLD_PX = 30.72
_MIN_COVERAGE = 0.95
_MIN_COMPLETED_PAIRS = 2
_PROVENANCE = {"tool_version": "fixture", "git_commit": "fixture",
               "command": "synthetic fixture builder", "created_at": "2026-01-01T00:00:00Z"}
_POLICY = {"name": "pusht/naive", "checkpoint_id": "synthetic-fixture",
           "checkpoint_sha256": "synthetic-fixture"}
_SIMULATOR = {"name": "pymunk", "version": "7.2.0"}


def _geometry() -> dict[str, Any]:
    return {
        "obstacles": [{"id": "obstacle_0", "kind": "virtual_point", "position": _OBSTACLE_PX,
                       "radius": _OBSTACLE_RADIUS_PX, "units": "px", "frame": _FRAME,
                       "source": "synthetic fixture; virtual executed-target separation proxy"}],
        "goal": {"pose": [_GOAL_PX, _GOAL_PX, _GOAL_ANGLE_RAD],
                 "units": "px (angle rad)", "frame": _FRAME},
        "block": {"shapes": [[[-60.0, 30.0], [60.0, 30.0], [60.0, 0.0], [-60.0, 0.0]],
                             [[-15.0, 30.0], [-15.0, 120.0], [15.0, 120.0], [15.0, 30.0]]],
                  "frame": "block_body", "pose_fields": ["block_pos", "block_angle"]},
        "arena": {"size": [_ARENA_SIZE_PX, _ARENA_SIZE_PX], "units": "px"},
    }


def _episode(episode_id: str, seed: int, offset_px: float = 0.0) -> schema.Episode:
    adapter = PushTTask()
    manifest = {
        "schema_version": schema.SCHEMA_VERSION, "kind": "demonstration",
        "episode_id": episode_id, "source_episode_id": episode_id, "task": adapter.name,
        "dataset": {"name": "synthetic", "revision": "fixture-v1", "checksum": "fixture"},
        "split": "train" if seed % 2 == 0 else "val", "split_source": "synthetic fixture",
        "timebase": adapter.timebase(), "observation_schema": adapter.observation_schema(),
        "action_schema": adapter.action_schema(), "fields_present": ["obs", "action_executed"],
        "provenance": _PROVENANCE, "synthetic": True,
    }
    start_px = _SAFE_START_PX + seed * _SEED_OFFSET_PX + offset_px
    steps = tuple({
        "step": step, "t": round(step * adapter.timebase()["dt"], _DECIMALS),
        "obs": {"agent_pos": [round(start_px + step * _STEP_DISPLACEMENT_PX, _DECIMALS),
                              start_px], "block_pos": _BLOCK_POSITION_PX,
                "block_angle": round(step * _ANGLE_STEP_RAD, _DECIMALS)},
        "action_executed": {"target": [start_px,
                                       round(start_px + step * _STEP_DISPLACEMENT_PX, _DECIMALS)]},
    } for step in range(_STEPS))
    return schema.Episode(manifest, steps)


def _write_demos(root: Path) -> None:
    for seed in _SEEDS:
        schema.write_episode(_episode(f"clean_{seed}", seed), root / "clean")
    for seed in _SEEDS[:-1]:
        episode = _episode(f"no_split_{seed}", seed)
        episode.manifest.update(split=None, split_source="none")
        schema.write_episode(episode, root / "no_split")
    for seed, name in enumerate(("nan", "shape", "time", "clean")):
        episode = _episode(name, seed)
        if name == "nan":
            episode.steps[_NAN_STEP]["obs"]["agent_pos"][0] = float("nan")
        elif name == "shape":
            episode.steps[_SHAPE_STEP]["action_executed"]["target"] = [0.0]
        elif name == "time":
            for step in episode.steps[_TIME_STEP:]:
                step["t"] = round(step["t"] - _NON_MONOTONIC_OFFSET_SECONDS, _DECIMALS)
        schema.write_episode(episode, root / "corrupt")
    for split in ("train", "val"):
        episode = _episode(split, 0)
        episode.manifest["split"] = split
        schema.write_episode(episode, root / "leakage")


def _comparison(name: str) -> dict[str, Any]:
    return {
        "schema_version": schema.SCHEMA_VERSION, "name": name, "task": "pusht",
        "simulator_version": _SIMULATOR["version"],
        "matrix": {"conditions": ["static"], "seeds": list(_SEEDS)},
        "variants": {"baseline": {"policy": _POLICY["name"]},
                     "candidate": {"policy": _POLICY["name"],
                                   "overrides": {"max_guidance_scale": 0.0}}},
        "intended_differences": ["overrides.max_guidance_scale"],
    }


def _gate(name: str) -> dict[str, Any]:
    return {
        "schema_version": schema.SCHEMA_VERSION, "name": name,
        "min_completed_pairs": _MIN_COMPLETED_PAIRS, "max_missing_pairs": 0,
        "max_incompatible_pairs": 0, "rules": [
            {"id": "success_not_worse", "metric": "native_success_rate",
             "compare": "candidate_minus_baseline", "units": "fraction of completed pairs",
             "threshold": _SUCCESS_THRESHOLD, "required": True},
            {"id": "separation_proxy", "metric": "obstacle_separation_min",
             "signal": "action_executed.target", "units": "px",
             "threshold": _SEPARATION_THRESHOLD_PX, "tolerance": 0.0,
             "min_coverage": _MIN_COVERAGE, "required": True},
        ],
    }


def _rollout(scenario: str, variant: str, seed: int, digest: str) -> schema.Episode:
    candidate = variant == "candidate"
    offset_px = _VARIANT_OFFSET_PX if candidate and scenario == "two_valid_trajectories" else 0.0
    episode = _episode(f"static__{seed}", seed, offset_px)
    manifest = episode.manifest
    del manifest["dataset"]
    truncated = candidate and scenario == "truncated" and seed == _TRUNCATED_SEED
    fingerprint = schema.sha256_json(seed)
    if candidate and scenario == "mismatched_reset" and seed in (1, 2):
        fingerprint = schema.sha256_json(["different", seed])
    manifest.update({
        "kind": "rollout", "condition": "static", "seed": seed, "policy": _POLICY,
        "simulator": _SIMULATOR, "split": None, "split_source": "none",
        "reset_state_fingerprint": fingerprint,
        "resolved_config": {"path": "resolved_config.json", "sha256": digest},
        "geometry": _geometry(), "outcome": {
            "native_success": not truncated, "terminated": not truncated, "truncated": truncated,
            "reason": "max_steps" if truncated else "complete", "length": _STEPS},
    })
    manifest["fields_present"] += ["reset_state_fingerprint", "geometry"]
    if candidate and scenario == "seeded_regression" and seed == _REGRESSION_SEED:
        episode.steps[FIRST_VIOLATION_STEP]["action_executed"]["target"] = [
            round(_OBSTACLE_PX[0] + _VIOLATION_DISTANCE_PX, _DECIMALS), _OBSTACLE_PX[1]]
    if candidate and scenario == "missing_signal":
        manifest["fields_present"].remove("action_executed")
        for step in episode.steps:
            del step["action_executed"]
    return episode


def _write_run(root: Path, scenario: str, variant: str, config: schema.ComparisonConfig) -> None:
    adapter = PushTTask()
    spec = adapters.find_policy(adapter, _POLICY["name"])
    resolved = adapters.resolve_policy_config(spec, config.variants[variant].overrides)
    directory = root / variant
    adapters.write_report_file(directory / "resolved_config.json", schema.canonical_json(resolved))
    seeds = _SEEDS[:2] if scenario == "missing_pairs" and variant == "candidate" else _SEEDS
    cells = []
    for seed in seeds:
        episode = _rollout(scenario, variant, seed, schema.sha256_json(resolved))
        schema.write_episode(episode, directory)
        cells.append({key: episode.manifest[key] for key in ("episode_id", "condition", "seed")})
    schema.write_run_manifest({
        "schema_version": schema.SCHEMA_VERSION, "run_id": f"{scenario}__{variant}",
        "task": adapter.name, "comparison": {"name": config.name, "sha256": config.sha256},
        "variant": variant, "policy": _POLICY, "simulator": _SIMULATOR,
        "provenance": _PROVENANCE, "episodes": cells, "synthetic": True,
    }, directory)


def build_fixtures(root: Path) -> None:
    """Write only into an empty or absent directory; never delete existing files."""
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"refusing to overwrite nonempty fixture directory {root}")
    root.mkdir(parents=True, exist_ok=True)
    _write_demos(root / "demos")
    for name in _SCENARIOS:
        scenario = root / "scenarios" / name
        path = scenario / "comparison.json"
        adapters.write_report_file(path, schema.canonical_json(_comparison(name)))
        adapters.write_report_file(scenario / "gate.json", schema.canonical_json(_gate(name)))
        config = schema.load_comparison_config(path)
        for variant in ("baseline", "candidate"):
            _write_run(scenario, name, variant, config)


def main(argv: list[str] | None = None) -> int:
    """Generate a fixture tree at the required destination argument."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    build_fixtures(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
