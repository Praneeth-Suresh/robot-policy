"""Pure process checks using recorded telemetry; native outcomes remain independent."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from guided_streaming_policy.lab import schema

__all__ = [
    "native_success", "failure_category", "SeparationSeries", "separation_series",
    "measurement_coverage", "SeparationResult", "evaluate_separation",
]


@dataclass(frozen=True)
class SeparationSeries:
    """Recorded times and centre distances in the manifest geometry units."""

    t: tuple[float | None, ...]
    values: tuple[float | None, ...]
    unknown_reason: str | None


@dataclass(frozen=True)
class SeparationResult:
    """Constraint verdict and observed evidence, including partial coverage."""

    verdict: str
    minimum: float | None
    minimum_step: int | None
    first_violation_step: int | None
    first_violation_time: float | None
    coverage: float
    steps: int
    reason: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return all result fields as a fresh JSON-compatible dictionary."""
        return asdict(self)


def native_success(episode: schema.Episode) -> bool | None:
    """Return the recorded native outcome verbatim, without recomputation."""
    value: bool | None = episode.manifest["outcome"]["native_success"]
    return value


def failure_category(episode: schema.Episode) -> str | None:
    """Classify native failures with crash, truncation, then failure precedence."""
    outcome = episode.manifest["outcome"]
    if outcome["reason"] == "crashed":
        return "crashed"
    if outcome["truncated"] is True:
        return "truncated"
    return "native_fail" if native_success(episode) is False else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def _position(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    numbers = tuple(_number(item) for item in value)
    if any(item is None for item in numbers):
        return None
    return tuple(item for item in numbers if item is not None)


def _distance(value: Any, obstacles: list[dict[str, Any]]) -> float | None:
    position = _position(value)
    if position is None:
        return None
    distances = []
    for obstacle in obstacles:
        other = _position(obstacle.get("position"))
        if other is None or len(other) != len(position):
            return None
        distance = math.dist(position, other)
        if not math.isfinite(distance):
            return None
        distances.append(distance)
    return min(distances)


def _unknown_reason(episode: schema.Episode, signal: str, group: str) -> str | None:
    manifest = episode.manifest
    fields = manifest["fields_present"]
    if not manifest.get("geometry") or "geometry" not in fields:
        return "no obstacle geometry recorded"
    if group not in fields:
        return f"signal {signal} was not collected"
    if not episode.steps:
        return "episode has no steps"
    if not manifest["geometry"].get("obstacles"):
        return "no obstacles in geometry"
    return None


def separation_series(episode: schema.Episode, signal: str) -> SeparationSeries:
    """Measure minimum Euclidean centre distance per recorded step, without interpolation.

    The dotted signal must name a position in the same units and coordinate frame
    as every obstacle position in the manifest. Radii are not subtracted.
    """
    group, separator, field = signal.partition(".")
    if not group or not separator or not field:
        raise ValueError("signal must be a dotted <group>.<field> path")
    times = tuple(_number(step.get("t")) for step in episode.steps)
    reason = _unknown_reason(episode, signal, group)
    if reason is not None:
        return SeparationSeries(times, (None,) * len(episode.steps), reason)
    obstacles = episode.manifest["geometry"]["obstacles"]
    values = []
    for step in episode.steps:
        recorded = step.get(group)
        value = recorded.get(field) if isinstance(recorded, dict) else None
        values.append(_distance(value, obstacles))
    return SeparationSeries(times, tuple(values), None)


def measurement_coverage(series: SeparationSeries) -> float:
    """Return the share of steps with measured distances, or zero for no steps."""
    return (sum(value is not None for value in series.values) / len(series.values)
            if series.values else 0.0)


def evaluate_separation(
    series: SeparationSeries, *, threshold: float, tolerance: float, min_coverage: float,
) -> SeparationResult:
    """Evaluate values < threshold - tolerance; all three share geometry units.

    Insufficient coverage takes precedence over an observed violation, while the
    minimum and first violation evidence are retained. Step numbers are indices
    into the recorded series; times are copied from its recorded time entries.
    """
    measured = [(index, value) for index, value in enumerate(series.values) if value is not None]
    minimum = min(measured, key=lambda item: item[1]) if measured else None
    first = next((index for index, value in measured if value < threshold - tolerance), None)
    coverage = measurement_coverage(series)
    reason = series.unknown_reason
    if reason is None and coverage < min_coverage:
        reason = f"measurement coverage {coverage} is below minimum {min_coverage}"
    verdict = "unknown" if reason is not None else "fail" if first is not None else "pass"
    return SeparationResult(
        verdict, minimum[1] if minimum else None, minimum[0] if minimum else None,
        first, series.t[first] if first is not None else None, coverage, len(series.values), reason,
    )
