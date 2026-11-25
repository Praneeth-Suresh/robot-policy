"""Push-T inventory and spaces without importing the environment or runners.

Push-T/core/pusht_env.py defines control_hz=10, a 512 px window, and 2D
positions/actions; its control period is 0.1 s. Frames use window y-down pixels
and the block angle modulo 2*pi radians.
"""

import importlib.metadata
import importlib.util
from pathlib import Path
from typing import Any, Iterator

from guided_streaming_policy.lab.schema import Episode
from guided_streaming_policy.lab.tasks.base import PolicySpec, PolicyStatus, ProbeResult
from guided_streaming_policy.lab.tasks.pusht.pins import compute_pins
from guided_streaming_policy.lab.tasks.pusht.policies import POLICIES
from guided_streaming_policy.paths import repo_root

__all__ = ["PushTTask"]

_POSITION_DIMENSIONS = 2
_CONTROL_DT_SECONDS = 0.1
_WINDOW_SIZE_PIXELS = 512
_PIXEL_FRAME = f"window_{_WINDOW_SIZE_PIXELS}_y_down"
_DEMONSTRATION_ZIP = "Push-T/data/pusht_cchi_v7_replay.zarr.zip"
_DEPENDENCIES = ("torch", "numpy", "pymunk", "zarr", "gym", "diffusers")


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

    def __init__(self, root: Path | None = None) -> None:
        self._root = repo_root() if root is None else root

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
        """Return the checked-in pin document under the repository root."""
        return self._root / "lab" / "pins.json"

    def current_pins(self) -> dict[str, Any]:
        """Recompute simulator metadata and file/configuration hashes from disk."""
        return compute_pins(self._root)
