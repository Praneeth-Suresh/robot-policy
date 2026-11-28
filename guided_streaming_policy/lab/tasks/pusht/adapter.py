"""Push-T inventory and spaces without importing the environment or runners.

Push-T/core/pusht_env.py defines control_hz=10, a 512 px window, and 2D
positions/actions; its control period is 0.1 s. Frames use window y-down pixels
and the block angle modulo 2*pi radians.
"""

import importlib.metadata
import importlib.resources
import importlib.util
from pathlib import Path
from typing import Any, Iterator

from guided_streaming_policy.lab.schema import Episode
from guided_streaming_policy.lab.tasks.base import (
    CONTRACT_VERSION,
    ConditionSpec,
    FixtureSet,
    PolicySpec,
    PolicyStatus,
    ProbeResult,
    TrialRecord,
    TrialRequest,
)
from guided_streaming_policy.lab.tasks.pusht.pins import compute_pins
from guided_streaming_policy.lab.tasks.pusht.policies import POLICIES
from guided_streaming_policy.paths import repo_root

__all__ = ["PushTTask"]

_PACKAGE = "guided_streaming_policy.lab.tasks.pusht"
_POSITION_DIMENSIONS = 2
_CONTROL_DT_SECONDS = 0.1
_WINDOW_SIZE_PIXELS = 512
_PIXEL_FRAME = f"window_{_WINDOW_SIZE_PIXELS}_y_down"
_DEMONSTRATION_ZIP = "Push-T/data/pusht_cchi_v7_replay.zarr.zip"
_DEPENDENCIES = ("torch", "numpy", "pymunk", "zarr", "gym", "diffusers")


_CONDITIONS = (
    ConditionSpec("static", "Fixed obstacle at normalized [-0.3, -0.5]; runs through gsp lab run."),
    ConditionSpec(
        "dynamic",
        "Obstacle moving sinusoidally; runs through the research scripts only "
        "(gsp run infer-dynamic-*), not yet through gsp lab run.",
        runnable=False,
    ),
)


def _dependencies() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in _DEPENDENCIES}


def _position_schema() -> dict[str, Any]:
    return {
        "shape": [_POSITION_DIMENSIONS],
        "dtype": "float64",
        "units": "px",
        "frame": _PIXEL_FRAME,
    }


class PushTTask:
    """Expose static reference policies, simulator prerequisites and trace schemas."""

    name = "pusht"
    contract_version = CONTRACT_VERSION

    def __init__(self, root: Path | None = None) -> None:
        self._root = repo_root() if root is None else root

    def fixtures(self) -> FixtureSet:
        """Locate the synthetic fixtures packaged with this task."""
        root = Path(str(importlib.resources.files(_PACKAGE) / "fixtures"))
        return FixtureSet(root / "demos/clean", root / "demos/corrupt",
                          root / "scenarios/seeded_regression", root)

    def conditions(self) -> tuple[ConditionSpec, ...]:
        """Return the static condition and the declared, not yet runnable dynamic one."""
        return _CONDITIONS

    def run_trial(self, request: TrialRequest) -> TrialRecord:
        """Run and record the guided episode, raising TrialCrashed on failure."""
        from guided_streaming_policy.lab.tasks.pusht.runner import run_pusht_trial

        return run_pusht_trial(request, self._root)

    def policies(self) -> tuple[PolicySpec, ...]:
        """Return the five declared static-condition reference policies."""
        return POLICIES

    def policy_statuses(self) -> tuple[PolicyStatus, ...]:
        """Report all absent checkpoints and packages for each reference policy."""
        dependencies = _dependencies()
        statuses = []
        for spec in self.policies():
            causes = [
                f"missing checkpoint {path}"
                for path in spec.checkpoints
                if not (self._root / path).is_file()
            ]
            causes.extend(
                f"missing Python package {name} (install the pusht extra)"
                for name, present in dependencies.items()
                if not present
            )
            statuses.append(
                PolicyStatus(
                    spec,
                    "unavailable" if causes else "runnable",
                    "; ".join(causes) if causes else None,
                )
            )
        return tuple(statuses)

    def probe(self) -> ProbeResult:
        """Inspect demonstration/checkpoint presence and simulator package metadata."""
        try:
            version = importlib.metadata.version("pymunk")
        except importlib.metadata.PackageNotFoundError:
            version = None
        assets = {_DEMONSTRATION_ZIP, *(path for spec in POLICIES for path in spec.checkpoints)}
        return ProbeResult(
            "pymunk",
            version,
            {path: (self._root / path).is_file() for path in sorted(assets)},
            _dependencies(),
        )

    def observation_schema(self) -> dict[str, Any]:
        """Return pixel positions and the block's angle in radians modulo 2*pi."""
        return {
            "fields": {
                "agent_pos": _position_schema(),
                "block_pos": _position_schema(),
                "block_angle": {
                    "shape": [],
                    "dtype": "float64",
                    "units": "rad",
                    "frame": "block_mod_2pi",
                },
            }
        }

    def action_schema(self) -> dict[str, Any]:
        """Return absolute target positions in the window's pixel frame."""
        return {"convention": "absolute", "fields": {"target": _position_schema()}}

    def timebase(self) -> dict[str, Any]:
        """Return the fixed control period in seconds."""
        return {"unit": "s", "mode": "fixed_dt", "dt": _CONTROL_DT_SECONDS}

    def read_demonstrations(self, path: Path) -> Iterator[Episode]:
        """Convert native demonstrations lazily; raise TraceReadError if unreadable."""
        from guided_streaming_policy.lab.tasks.pusht.demonstrations import read_demonstrations

        return read_demonstrations(path)

    def pins_path(self) -> Path:
        """Return the checked-in pin document packaged with this task."""
        return Path(str(importlib.resources.files(_PACKAGE) / "pins.json"))

    def current_pins(self) -> dict[str, Any]:
        """Recompute simulator metadata and file/configuration hashes from disk."""
        return compute_pins(self._root)
