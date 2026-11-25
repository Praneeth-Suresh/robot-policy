"""Record the guided (third) episode without modifying the research runners.

The environment returns truncated == terminated, so truncation here means the
runner's step cap was reached without environment termination. Imports that
load research code or optional packages stay inside functions.
"""

import importlib
import math
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from guided_streaming_policy.lab import adapters, schema
from guided_streaming_policy.lab.tasks.base import TrialCrashed, TrialRecord, TrialRequest

__all__ = [
    "STATIC_OBSTACLE_NORM", "MAX_STEPS", "OBSTACLE_RADIUS_PX", "RecordingEnv",
    "reset_fingerprint", "run_pusht_trial",
]

STATIC_OBSTACLE_NORM = (-0.3, -0.5)  # Push-T/run_trial.py, static fixed_obstacle_pos.
MAX_STEPS = 250  # Push-T/inference/static/inference_*.py, run_episode max_steps.
OBSTACLE_RADIUS_PX = 15.36  # Same runners: TRAIN_NORM_R (0.06) * 256 px.
_WINDOW_SIZE_PX = 512  # Push-T/core/pusht_env.py, window_size.
_TEE_SCALE_PX = 30  # Push-T/core/pusht_env.py, add_tee(scale=30).
_TEE_LENGTH = 4  # Push-T/core/pusht_env.py, add_tee length.
_COORDINATE_HALVES = 2
_GOAL_ANGLE_RAD = math.pi / 4  # Push-T/core/pusht_env.py, _setup goal_pose.
_RESET_OBSERVATION_VALUES = 5
_RESET_VALUE_DECIMALS = 6
_TIME_SECONDS_DECIMALS = 6
_AGENT_POSITION = slice(0, 2)
_BLOCK_POSITION = slice(2, 4)
_BLOCK_ANGLE = 4
_GUIDED_SEGMENT_INDEX = 2
_FIELDS_PRESENT = ("obs", "action_executed", "reward", "reset_state_fingerprint", "geometry")


@dataclass
class _Segment:
    reset_obs: tuple[float, ...]
    steps: list[dict[str, Any]] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False


def reset_fingerprint(obs: Sequence[float]) -> str:
    """Hash the five reset observation values as floats rounded to six decimals."""
    if len(obs) != _RESET_OBSERVATION_VALUES:
        raise ValueError(f"Expected {_RESET_OBSERVATION_VALUES} reset observation values")
    return schema.sha256_json([round(float(value), _RESET_VALUE_DECIMALS) for value in obs])


def _observation(obs: tuple[float, ...]) -> dict[str, Any]:
    return {"agent_pos": list(obs[_AGENT_POSITION]), "block_pos": list(obs[_BLOCK_POSITION]),
            "block_angle": obs[_BLOCK_ANGLE]}


class RecordingEnv:
    """Proxy successful resets and actions, retaining independent before-action snapshots."""

    def __init__(self, env: Any, dt: float) -> None:
        self._env = env
        self._dt = dt
        self._obs: tuple[float, ...] = ()
        self.segments: list[_Segment] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Start a segment only after reset returns; propagate seeded-reset TypeError."""
        result = self._env.reset(*args, **kwargs)
        self._obs = tuple(float(value) for value in result[0])
        self.segments.append(_Segment(self._obs))
        return result

    def step(self, action: Any) -> Any:
        """Delegate a control step and record its input observation, action and reward."""
        if not self.segments:
            raise RuntimeError("Cannot record a step before a successful reset")
        segment = self.segments[-1]
        index = len(segment.steps)
        before = _observation(self._obs)
        target = [float(value) for value in action]
        result = self._env.step(action)
        obs, reward, terminated, truncated, _info = result
        segment.steps.append({
            "step": index, "t": round(index * self._dt, _TIME_SECONDS_DECIMALS), "obs": before,
            "action_executed": {"target": target}, "reward": float(reward),
        })
        segment.terminated, segment.truncated = bool(terminated), bool(truncated)
        self._obs = tuple(float(value) for value in obs)
        return result


@contextmanager
def _working_directory(root: Path) -> Iterator[None]:
    directory = str(root.resolve() / "Push-T")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    previous = Path.cwd()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(previous)


def _block_shapes() -> list[list[list[float]]]:
    # Copied from Push-T/core/pusht_env.py add_tee, in the body frame.
    scale = float(_TEE_SCALE_PX)
    half_width = _TEE_LENGTH * scale / _COORDINATE_HALVES
    half_stem = scale / _COORDINATE_HALVES
    return [
        [[-half_width, scale], [half_width, scale], [half_width, 0.0], [-half_width, 0.0]],
        [[-half_stem, scale], [-half_stem, _TEE_LENGTH * scale],
         [half_stem, _TEE_LENGTH * scale], [half_stem, scale]],
    ]


def _geometry(module: ModuleType) -> dict[str, Any]:
    import numpy as np

    position = module.unnormalize_data(
        np.array(STATIC_OBSTACLE_NORM, dtype=np.float32), module.stats["action"])
    frame = f"window_{_WINDOW_SIZE_PX}_y_down"
    return {
        "obstacles": [{
            "id": "obstacle_0", "kind": "virtual_point",
            "position": [float(value) for value in position], "radius": OBSTACLE_RADIUS_PX,
            "units": "px", "frame": frame,
            "source": "inference static runner: fixed_obstacle_pos converted with "
                      "unnormalize_data; radius is TRAIN_NORM_R*256, native collision "
                      "distance is 2*radius = 30.72 px between the executed action target "
                      "and the obstacle centre",
        }],
        "goal": {"pose": [_WINDOW_SIZE_PX / _COORDINATE_HALVES,
                          _WINDOW_SIZE_PX / _COORDINATE_HALVES, _GOAL_ANGLE_RAD],
                 "units": "px (angle rad)", "frame": frame},
        "block": {"shapes": _block_shapes(), "frame": "block_body",
                  "pose_fields": ["block_pos", "block_angle"]},
        "arena": {"size": [_WINDOW_SIZE_PX, _WINDOW_SIZE_PX], "units": "px"},
    }


def _invoke(
    module: ModuleType, function_name: str, recorder: RecordingEnv, request: TrialRequest,
) -> dict[str, Any]:
    original = module.env
    vars(module)["env"] = recorder
    try:
        metrics, _images = getattr(module, function_name)(
            seed=request.seed, fixed_obstacle_pos=list(STATIC_OBSTACLE_NORM),
            save_gif=False, **request.settings)
        return dict(metrics)
    finally:
        vars(module)["env"] = original


def _record(
    segment: _Segment, geometry: dict[str, Any] | None, metrics: dict[str, Any] | None,
) -> TrialRecord:
    truncated = not segment.terminated and len(segment.steps) == MAX_STEPS
    reason = ("coverage_reached" if segment.terminated else
              "max_steps" if truncated else "ended_without_termination")
    native = {} if metrics is None else {
        "guided_success": bool(metrics["guided_success"]),
        "guided_collision": bool(metrics["guided_collision"]),
        "guided_reward": float(metrics["guided_reward"]),
        "guided_min_dist": float(metrics["guided_min_dist"]),
    }
    return TrialRecord(
        tuple(segment.steps), reset_fingerprint(segment.reset_obs),
        None if metrics is None else bool(metrics["guided_success"]),
        segment.terminated, truncated, "crashed" if metrics is None else reason,
        native, geometry, _FIELDS_PRESENT,
    )


def run_pusht_trial(request: TrialRequest, root: Path) -> TrialRecord:
    """Run a static policy and record the third segment, preserving it on failure.

    The env's truncated duplicates terminated; only an unterminated MAX_STEPS
    episode is marked truncated. Working directory and module.env are restored
    even on interruption. Research imports are cached by importlib.
    """
    from guided_streaming_policy.lab.tasks.pusht.adapter import PushTTask

    recorder = None
    geometry = None
    try:
        with _working_directory(root):
            adapter = PushTTask(root)
            if request.condition not in adapter.conditions():
                raise ValueError(f"Unsupported condition {request.condition!r}")
            spec = adapters.find_policy(adapter, request.policy)
            module_name, function_name = spec.entry_point.split(":")
            module = importlib.import_module(module_name)
            recorder = RecordingEnv(module.env, float(adapter.timebase()["dt"]))
            geometry = _geometry(module)
            metrics = _invoke(module, function_name, recorder, request)
            if len(recorder.segments) <= _GUIDED_SEGMENT_INDEX:
                raise ValueError("Runner recorded fewer than three successful resets")
            return _record(recorder.segments[_GUIDED_SEGMENT_INDEX], geometry, metrics)
    except Exception as error:
        partial = None
        if recorder is not None and len(recorder.segments) > _GUIDED_SEGMENT_INDEX:
            partial = _record(recorder.segments[_GUIDED_SEGMENT_INDEX], geometry, None)
        raise TrialCrashed(str(error), partial) from error
