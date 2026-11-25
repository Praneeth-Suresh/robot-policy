"""Standard-library-only task adapter contract for Policy Trace Lab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

from guided_streaming_policy.lab.schema import Episode

__all__ = [
    "TunableSpec", "PolicySpec", "PolicyStatus", "ProbeResult", "TaskAdapter",
    "TrialRequest", "TrialRecord", "TrialCrashed",
]


@dataclass(frozen=True)
class TunableSpec:
    """Describe one supported setting and its repository default."""

    name: str
    default: float | int | str | bool
    description: str


@dataclass(frozen=True)
class PolicySpec:
    """Describe a reference policy without importing its runner."""

    name: str
    summary: str
    paper: str
    entry_point: str
    checkpoints: tuple[str, ...]
    tunables: tuple[TunableSpec, ...]


@dataclass(frozen=True)
class PolicyStatus:
    """Report runnable or unavailable status and any missing prerequisites."""

    spec: PolicySpec
    status: str
    reason: str | None


@dataclass(frozen=True)
class ProbeResult:
    """Report simulator identity, asset presence and Python package availability."""

    simulator_name: str
    simulator_version: str | None
    assets: dict[str, bool]
    dependencies: dict[str, bool]


class TaskAdapter(Protocol):
    """Expose task metadata and reproducibility inputs without running a policy."""

    name: str

    def conditions(self) -> tuple[str, ...]:
        """Return supported simulation conditions."""
        ...

    def run_trial(self, request: TrialRequest) -> TrialRecord:
        """Run one policy on one matrix cell with step recording.

        Raise TrialCrashed when the trial fails.
        """
        ...

    def policies(self) -> tuple[PolicySpec, ...]:
        """Return this task's reference policy specifications."""
        ...

    def policy_statuses(self) -> tuple[PolicyStatus, ...]:
        """Return policy readiness with explicit causes for unavailability."""
        ...

    def probe(self) -> ProbeResult:
        """Inspect simulator version, assets and dependencies without loading them."""
        ...

    def observation_schema(self) -> dict[str, Any]:
        """Return named observation shapes, dtypes, units and coordinate frames."""
        ...

    def action_schema(self) -> dict[str, Any]:
        """Return action convention, shapes, dtypes, units and coordinate frames."""
        ...

    def timebase(self) -> dict[str, Any]:
        """Return the recorded control timebase and its units."""
        ...

    def read_demonstrations(self, path: Path) -> Iterator[Episode]:
        """Convert native demonstrations lazily without modifying the source.

        Raise schema.TraceReadError if the source is unreadable.
        """
        ...

    def pins_path(self) -> Path:
        """Return the path of the checked-in pin document for this task."""
        ...

    def current_pins(self) -> dict[str, Any]:
        """Recompute this task's simulator, checkpoint and default config pins."""
        ...


@dataclass(frozen=True)
class TrialRequest:
    """One explicit matrix cell with resolved policy tunables in settings."""

    policy: str
    settings: dict[str, Any]
    condition: str
    seed: int


@dataclass(frozen=True)
class TrialRecord:
    """Collected steps and native outcome, including partial telemetry on failure."""

    steps: tuple[dict[str, Any], ...]
    reset_state_fingerprint: str | None
    native_success: bool | None
    terminated: bool
    truncated: bool
    reason: str
    native: dict[str, Any]
    geometry: dict[str, Any] | None
    fields_present: tuple[str, ...]


class TrialCrashed(Exception):
    """A failed trial with whatever telemetry was recorded before the failure."""

    def __init__(self, message: str, record: TrialRecord | None = None) -> None:
        super().__init__(message)
        self.record = record
