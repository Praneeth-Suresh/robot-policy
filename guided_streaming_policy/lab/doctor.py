"""Pure trace diagnostics; absent telemetry and absent splits are never inferred.

Fingerprints cover the full observation/action trajectory, rounded in each field's
own units. They do not establish reset identity or detect coordinate-frame errors.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Any, TypeGuard

from guided_streaming_policy.lab import schema

__all__ = ["CHECK_IDS", "FINGERPRINT_DECIMALS", "TIME_GAP_RATIO", "check_episodes",
           "render_summary", "exit_code"]

FINGERPRINT_DECIMALS = 3  # Rounding precision in each field's own units.
TIME_GAP_RATIO = 1.5  # Intervals larger than this multiple of nominal dt are gaps.
CHECK_IDS = (
    "schema.missing_required", "shape.mismatch", "value.non_finite",
    "time.non_monotonic", "time.gap", "split.leakage", "split.duplicate_episode",
    "split.not_checked", "outcome.inconsistent",
)
_STEP_FIELDS = ("obs", "action_executed", "action_proposed", "reward", "info")
_SHAPE_GROUPS = {"obs": "observation_schema", "action_executed": "action_schema"}
_EXIT_OK = 0
_EXIT_FINDINGS = 1


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finding(
    episode: schema.Episode, check: str, message: str, hint: str, *,
    severity: str = "error", step: int | None = None, field: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "check": check, "severity": severity,
        "episode_id": str(episode.manifest.get("episode_id", "(unknown)")),
        "message": message, "hint": hint,
    }
    if step is not None:
        result["step"] = step
    if field is not None:
        result["field"] = field
    return result


def _known_kind(episode: schema.Episode) -> bool:
    return episode.manifest.get("kind") in (schema.KIND_DEMONSTRATION, schema.KIND_ROLLOUT)


def _missing_required(episode: schema.Episode) -> list[dict[str, Any]]:
    check = "schema.missing_required"
    if not _known_kind(episode):
        return [_finding(episode, check, "Missing or unknown episode kind.",
                         "Supply kind as demonstration or rollout.", field="kind")]
    findings = [_finding(episode, check, f"Manifest lacks {field}.",
                         "Supply the required manifest field from the source.", field=field)
                for field in schema.required_manifest_fields(episode.manifest["kind"])
                if field not in episode.manifest]
    required = {"step", "t"} | (set(episode.manifest.get("fields_present", [])) & set(_STEP_FIELDS))
    for index, step in enumerate(episode.steps):
        findings.extend(_finding(episode, check, f"Step lacks {field}.",
                                 "Restore collected telemetry or correct fields_present.",
                                 step=index, field=field)
                        for field in sorted(required - step.keys()))
    return findings


def _shape(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return []
    if not value:
        return [0]
    child = _shape(value[0])
    if child is None or any(_shape(item) != child for item in value[1:]):
        return None
    return [len(value), *child]


def _group_shape_findings(
    episode: schema.Episode, index: int, group: str, declared: dict[str, Any],
) -> list[dict[str, Any]]:
    values = _mapping(episode.steps[index][group])
    first = _mapping(episode.steps[0].get(group))
    findings = []
    for name in sorted(values.keys() | declared.keys()):
        field = f"{group}.{name}"
        hint = "Match the recorded field shape and manifest declaration without interpolation."
        if name not in values:
            findings.append(_finding(episode, "shape.mismatch", "Declared field is missing.",
                                     hint, step=index, field=field))
            continue
        if name not in declared:
            findings.append(_finding(episode, "shape.mismatch", "Step field is not declared.",
                                     hint, step=index, field=field))
        expected = (_mapping(declared[name]).get("shape") if name in declared
                    else _shape(first[name]) if name in first else _shape(values[name]))
        actual = _shape(values[name])
        if actual is None or actual != expected:
            findings.append(_finding(episode, "shape.mismatch",
                                     f"Shape {actual} differs from expected {expected}.",
                                     hint, step=index, field=field))
    return findings


def _shape_mismatch(episode: schema.Episode) -> list[dict[str, Any]]:
    findings = []
    for index, step in enumerate(episode.steps):
        for group, declaration in _SHAPE_GROUPS.items():
            if group in step:
                declared = _mapping(_mapping(episode.manifest.get(declaration)).get("fields"))
                findings.extend(_group_shape_findings(episode, index, group, declared))
    return findings


def _has_non_finite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, list):
        return any(_has_non_finite(item) for item in value)
    if isinstance(value, dict):
        return any(_has_non_finite(item) for item in value.values())
    return False


def _numeric_fields(step: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    for name in ("t", "reward"):
        if name in step:
            yield name, step[name]
    for group in ("obs", "action_executed", "action_proposed"):
        for name, value in _mapping(step.get(group)).items():
            yield f"{group}.{name}", value


def _non_finite(episode: schema.Episode) -> list[dict[str, Any]]:
    return [_finding(episode, "value.non_finite", "Numeric telemetry contains NaN or infinity.",
                     "Repair the source telemetry; do not interpolate missing values.",
                     step=index, field=field)
            for index, step in enumerate(episode.steps) for field, value in _numeric_fields(step)
            if _has_non_finite(value)]


def _finite_number(value: Any) -> TypeGuard[int | float]:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and not _has_non_finite(value))


def _intervals(episode: schema.Episode) -> Iterator[tuple[int, float]]:
    for index in range(1, len(episode.steps)):
        previous, current = episode.steps[index - 1].get("t"), episode.steps[index].get("t")
        if _finite_number(previous) and _finite_number(current):
            yield index, current - previous


def _non_monotonic(episode: schema.Episode) -> list[dict[str, Any]]:
    return [_finding(episode, "time.non_monotonic", "Time is not strictly increasing.",
                     "Check the source ordering and timebase.", step=index, field="t")
            for index, interval in _intervals(episode) if interval <= 0]


def _nominal_dt(episode: schema.Episode) -> float | None:
    timebase = _mapping(episode.manifest.get("timebase"))
    dt = timebase.get("dt")
    if timebase.get("mode") == "fixed_dt" and _finite_number(dt) and dt > 0:
        return float(dt)
    return None


def _time_gap(episode: schema.Episode) -> list[dict[str, Any]]:
    dt = _nominal_dt(episode)
    if dt is None:
        return []
    return [_finding(episode, "time.gap",
                     f"Time interval {interval} exceeds {TIME_GAP_RATIO} x dt.",
                     "Check for dropped control steps or an incorrect nominal dt.",
                     severity="warning", step=index, field="t")
            for index, interval in _intervals(episode) if interval > TIME_GAP_RATIO * dt]


def _rounded(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(value, FINGERPRINT_DECIMALS)
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    return value


def _fingerprint_groups(episodes: list[schema.Episode]) -> list[list[schema.Episode]]:
    groups: dict[str, list[schema.Episode]] = defaultdict(list)
    for episode in episodes:
        if any(_has_non_finite(value) for step in episode.steps
               for _, value in _numeric_fields(step)):
            continue
        trajectory = [[step.get(group) for step in episode.steps] for group in _SHAPE_GROUPS]
        groups[schema.sha256_json(_rounded(trajectory))].append(episode)
    return list(groups.values())


def _leakage(group: list[schema.Episode]) -> list[dict[str, Any]]:
    labeled = [episode for episode in group if episode.manifest.get("split") is not None]
    if len({episode.manifest["split"] for episode in labeled}) <= 1:
        return []
    return [_finding(
        episode, "split.leakage",
        "Matching trajectory across splits; other episodes: " + ", ".join(
            f"{other.manifest.get('episode_id', '(unknown)')} "
            f"(split={other.manifest['split']!r})"
            for other in labeled if other is not episode) + ".",
        "Correct split assignments or remove duplicated trajectories across splits.")
        for episode in labeled]


def _duplicates(group: list[schema.Episode]) -> list[dict[str, Any]]:
    first_by_split: dict[str | None, str] = {}
    findings = []
    for episode in group:
        split = episode.manifest.get("split")
        episode_id = str(episode.manifest.get("episode_id", "(unknown)"))
        first = first_by_split.setdefault(split, episode_id)
        if episode_id != first:
            findings.append(_finding(episode, "split.duplicate_episode",
                                     f"Trajectory duplicates {first} in split {split!r}.",
                                     "Review duplicate episodes in the source dataset.",
                                     severity="warning"))
    return findings


def _not_checked() -> dict[str, Any]:
    return {"check": "split.not_checked", "severity": "info", "episode_id": "(dataset)",
            "message": "Split leakage could not be checked; no leakage claim is made.",
            "hint": "Supply split and split_source in each episode manifest."}


def _outcome(episode: schema.Episode) -> list[dict[str, Any]]:
    if episode.manifest.get("kind") != schema.KIND_ROLLOUT:
        return []
    outcome = _mapping(episode.manifest.get("outcome"))
    conditions = (
        (outcome.get("length") != len(episode.steps), "outcome.length",
         "Outcome length differs from the recorded step count."),
        (outcome.get("terminated") is True and outcome.get("truncated") is True,
         "outcome.terminated", "Outcome is both terminated and truncated."),
        (outcome.get("reason") == "crashed" and outcome.get("native_success") is True,
         "outcome.native_success", "Crashed outcome claims native success."),
    )
    return [_finding(episode, "outcome.inconsistent", message,
                     "Correct the recorded rollout outcome using the runner's evidence.",
                     field=field)
            for condition, field, message in conditions if condition]


def _skipped(episodes: list[schema.Episode]) -> dict[str, str]:
    skipped = {}
    if not any(_nominal_dt(episode) is not None for episode in episodes):
        skipped["time.gap"] = "timebase has no nominal dt"
    if not any(episode.manifest.get("split") is not None for episode in episodes):
        skipped["split.leakage"] = "no split provenance in the input"
    if not any(episode.manifest.get("kind") == schema.KIND_ROLLOUT for episode in episodes):
        skipped["outcome.inconsistent"] = "input has no rollout episodes"
    if not episodes:
        for check in ("shape.mismatch", "value.non_finite", "time.non_monotonic",
                      "split.duplicate_episode"):
            skipped[check] = "input has no episodes with a recognized kind"
    return skipped


def _sort_key(finding: dict[str, Any]) -> tuple[int, str, int, str, str]:
    return (schema.SEVERITIES.index(finding["severity"]), finding["episode_id"],
            finding.get("step", -1), finding["check"], finding.get("field", ""))


def check_episodes(
    episodes: Iterable[schema.Episode], *, input_path: str, input_checksum: str,
) -> dict[str, Any]:
    """Consume episodes once and return a deterministic report without IO or inference."""
    findings = []
    valid = []
    for episode in episodes:
        findings.extend(_missing_required(episode))
        if _known_kind(episode):
            valid.append(episode)
            for check in (_shape_mismatch, _non_finite, _non_monotonic, _time_gap, _outcome):
                findings.extend(check(episode))
    valid.sort(key=lambda episode: str(episode.manifest.get("episode_id", "(unknown)")))
    for group in _fingerprint_groups(valid):
        findings.extend(_leakage(group))
        findings.extend(_duplicates(group))
    skipped = _skipped(valid)
    if "split.leakage" in skipped:
        findings.append(_not_checked())
    findings.sort(key=_sort_key)
    counts = {severity: sum(f["severity"] == severity for f in findings)
              for severity in schema.SEVERITIES}
    return {"schema_version": schema.SCHEMA_VERSION,
            "input": {"path": input_path, "checksum": input_checksum},
            "summary": {"counts": counts, "checks_run": sorted(set(CHECK_IDS) - skipped.keys()),
                        "checks_skipped": [{"check": check, "reason": reason}
                                           for check, reason in sorted(skipped.items())]},
            "findings": findings}


def render_summary(report: dict[str, Any]) -> str:
    """Render counts and each finding's check, episode and available location."""
    counts = report["summary"]["counts"]
    lines = ["Policy Trace Lab Doctor: " + ", ".join(
        f"{counts[severity]} {severity}" for severity in schema.SEVERITIES)]
    for finding in report["findings"]:
        location = str(finding["episode_id"])
        if "step" in finding:
            location += f" step {finding['step']}"
        if "field" in finding:
            location += f" {finding['field']}"
        lines.append(f"{finding['severity']} {finding['check']} [{location}]: {finding['message']}")
    return "\n".join(lines)


def exit_code(report: dict[str, Any]) -> int:
    """Return 1 only for error findings; warnings and provenance info remain successful."""
    return _EXIT_FINDINGS if report["summary"]["counts"]["error"] else _EXIT_OK
