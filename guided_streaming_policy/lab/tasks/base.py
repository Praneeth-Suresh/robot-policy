"""Standard-library-only task adapter contract for Policy Trace Lab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

from guided_streaming_policy.lab.schema import Episode

__all__ = [
    "CONTRACT_VERSION", "ConditionSpec", "TunableSpec", "PolicySpec", "PolicyStatus",
    "ProbeResult", "TaskAdapter", "TrialRequest", "TrialRecord", "TrialCrashed", "FixtureSet",
]

CONTRACT_VERSION = 1


@dataclass(frozen=True)
class ConditionSpec:
    """Name one scenario (condition) of a task pack and say what it is.

    A condition that is declared but has no run hook yet sets runnable to False;
    `gsp lab run` then refuses it with the description as the reason.
    """

    name: str
    description: str
    runnable: bool = True


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


@dataclass(frozen=True)
class FixtureSet:
    """Synthetic inputs for the offline demo and CI.

    scenario holds baseline/, candidate/, comparison.json and gate.json.
    """

    demo_clean: Path
    demo_corrupt: Path
    scenario: Path


class TaskAdapter(Protocol):
    """Expose task metadata and reproducibility inputs without running a policy.

    A task pack is one simulator, its task definition, its conditions and its
    reference policies. `probe()` and `policies()` run when packs are listed, so
    they must not import the simulator, torch or numpy; check for them with
    `importlib.util.find_spec` instead.
    """

    name: str
    contract_version: int

    def fixtures(self) -> FixtureSet:
        """Return the synthetic fixture set used by the offline demo and CI."""
        ...

    def conditions(self) -> tuple[ConditionSpec, ...]:
        """Return the named conditions (scenarios) this pack supports."""
        ...

    def run_trial(self, request: TrialRequest) -> TrialRecord:
        """Run one policy on one matrix cell with step recording.

        Raise TrialCrashed when the trial fails.
        """
        ...

    def policies(self) -> tuple[PolicySpec, ...]:
        """Return this task's reference policy specifications without importing runners."""
        ...

    def policy_statuses(self) -> tuple[PolicyStatus, ...]:
        """Return policy readiness with explicit causes for unavailability."""
        ...

    def probe(self) -> ProbeResult:
        """Inspect simulator version, assets and dependencies without importing them."""
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
