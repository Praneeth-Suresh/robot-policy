"""Gate verdicts over completed pairs; native and constraint outcomes stay separate.

Success passes iff candidate_rate - baseline_rate >= threshold; negative thresholds
tolerate drops, and zero completed pairs means unknown. Separation uses recorded
centre distances: values < threshold - tolerance fail, and coverage < min_coverage
is unknown. Rule units must equal obstacle units or the episode is unknown
('units mismatch'). Candidates determine the rule: any fail, else any unknown,
else pass. Baselines only affect reported minima and paired differences; wins and
losses exceed tolerance. Coverage/integrity blocks precede required unknown rules,
then required failures. Optional rules never change the verdict. Missing evidence
is never invented.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from guided_streaming_policy.lab import checks, pairing, schema

__all__ = ["DISPLAY_DECIMALS", "evaluate_gate", "render_verdict", "gate_exit_code"]
DISPLAY_DECIMALS = 2
_EXIT_PASS, _EXIT_REGRESSION, _EXIT_BLOCKED = 0, 1, 2
_HASH_PREFIX_CHARACTERS = 12
_EMPTY_REPLAY_STEP = 0
_VARIANTS = ("baseline", "candidate")
_CATEGORIES = ("crashed", "truncated", "native_fail", "constraint_fail")
_Measurements = list[tuple[dict[str, Any], schema.Episode, schema.Episode]]


def _coverage(document: dict[str, Any], config: schema.GateConfig) -> dict[str, int]:
    counts = document.get("coverage", {}) if isinstance(document, dict) else {}
    counts = counts if isinstance(counts, dict) else {}
    return {
        **{
            key: max(0, counts[key]) if type(counts.get(key)) is int else 0
            for key in ("expected", "completed", "incompatible", "missing")
        },
        "min_completed_pairs": config.min_completed_pairs,
        "max_missing_pairs": config.max_missing_pairs,
        "max_incompatible_pairs": config.max_incompatible_pairs,
    }


def _coverage_reasons(coverage: dict[str, int]) -> list[str]:
    reasons = []
    completed, minimum = coverage["completed"], coverage["min_completed_pairs"]
    if completed < minimum:
        reasons.append(f"completed pairs {completed} below minimum {minimum}")
    for kind in ("missing", "incompatible"):
        count, maximum = coverage[kind], coverage[f"max_{kind}_pairs"]
        if count > maximum:
            reasons.append(f"{kind} pairs {count} exceed maximum {maximum}")
    return reasons


def _read_pairs(document: dict[str, Any], base_dir: Path, reasons: list[str]) -> _Measurements:
    result = []
    for pair in document["pairs"]:
        if pair["status"] == "completed":
            hash_errors = pairing.verify_trace_hashes({"pairs": [pair]}, base_dir)
            reasons.extend(hash_errors)
            if hash_errors:
                continue
            try:
                a, b = (schema.read_episode(base_dir / pair[side]["path"]) for side in _VARIANTS)
                result.append((pair, a, b))
            except (schema.TraceReadError, OSError, ValueError) as error:
                reasons.append(f"cannot read completed pair {pair['key']}: {error}")
    return result


def _base_rule(rule: schema.GateRule) -> dict[str, Any]:
    return {
        "id": rule.id, "metric": rule.metric, "required": rule.required,
        "units": rule.units, "threshold": rule.threshold,
        **dict.fromkeys(("baseline_value", "candidate_value", "delta")),
        **dict.fromkeys(("samples", "wins", "losses", "ties"), 0),
    }


def _success_rule(document: dict[str, Any], rule: schema.GateRule) -> dict[str, Any]:
    result = _base_rule(rule)
    delta = document.get("deltas", {}).get("native_success", {})
    result.update(
        baseline_value=delta.get("baseline_rate"),
        candidate_value=delta.get("candidate_rate"),
        delta=delta.get("delta"),
        samples=delta.get("n", 0),
        **{key: delta.get(key, 0) for key in ("wins", "losses", "ties")},
    )
    if result["samples"] == 0 or result["delta"] is None:
        result.update(verdict="unknown", reason="zero completed pairs")
    else:
        result["verdict"] = "pass" if result["delta"] >= rule.threshold else "fail"
    return result


def _measure(episode: schema.Episode, rule: schema.GateRule) -> checks.SeparationResult:
    assert rule.signal is not None and rule.tolerance is not None and rule.min_coverage is not None
    obstacles = episode.manifest.get("geometry", {}).get("obstacles", [])
    series = checks.separation_series(episode, rule.signal)
    if any(obstacle.get("units") != rule.units for obstacle in obstacles):
        series = checks.SeparationSeries(series.t, (None,) * len(series.values), "units mismatch")
    if series.unknown_reason is None and not any(value is not None for value in series.values):
        series = checks.SeparationSeries(
            series.t, series.values, f"signal {rule.signal} has no measurements"
        )
    return checks.evaluate_separation(
        series, threshold=rule.threshold, tolerance=rule.tolerance, min_coverage=rule.min_coverage
    )


def _minimum(results: list[checks.SeparationResult]) -> float | None:
    values = [result.minimum for result in results if result.minimum is not None]
    return min(values) if values else None


def _separation_rule(
    rule: schema.GateRule, measurements: _Measurements
) -> tuple[dict[str, Any], list[checks.SeparationResult]]:
    baseline = [_measure(a, rule) for _, a, _ in measurements]
    candidate = [_measure(b, rule) for _, _, b in measurements]
    result = _base_rule(rule)
    first, second = _minimum(baseline), _minimum(candidate)
    differences = [
        b.minimum - a.minimum
        for a, b in zip(baseline, candidate, strict=True)
        if a.minimum is not None and b.minimum is not None
    ]
    assert rule.tolerance is not None
    wins = sum(delta > rule.tolerance for delta in differences)
    losses = sum(delta < -rule.tolerance for delta in differences)
    result.update(
        baseline_value=first,
        candidate_value=second,
        delta=second - first if second is not None and first is not None else None,
        samples=len(differences),
        wins=wins,
        losses=losses,
        ties=len(differences) - wins - losses,
        tolerance=rule.tolerance,
        signal=rule.signal,
        coverage=min((item.coverage for item in candidate), default=0.0),
    )
    verdicts = [item.verdict for item in candidate]
    result["verdict"] = (
        "fail"
        if "fail" in verdicts
        else "unknown"
        if (not candidate or "unknown" in verdicts)
        else "pass"
    )
    if result["verdict"] == "unknown":
        result["reason"] = (
            "; ".join(dict.fromkeys(item.reason for item in candidate if item.reason))
            or "zero completed pairs"
        )
    return result, candidate


def _regression(
    pair: dict[str, Any],
    candidate: schema.Episode,
    violations: list[tuple[str, checks.SeparationResult]],
    success_ids: list[str],
) -> dict[str, Any] | None:
    native_loss = (
        pair["baseline"]["native_success"] is True and checks.native_success(candidate) is not True
    )
    category = (checks.failure_category(candidate) or "native_fail") if native_loss else None
    categories = [
        key for key in _CATEGORIES if key == category or (key == "constraint_fail" and violations)
    ]
    if not categories:
        return None
    first = min(
        (item for _, item in violations),
        key=lambda item: (
            item.first_violation_time if item.first_violation_time is not None else math.inf,
            item.first_violation_step,
        ),
        default=None,
    )
    first_step = (
        candidate.steps[first.first_violation_step]["step"]
        if first is not None and first.first_violation_step is not None
        else None
    )
    return {
        "key": dict(pair["key"]),
        "category": categories[0],
        "categories": categories,
        "rule_ids": (success_ids if native_loss else []) + [key for key, _ in violations],
        "first_violation": {"step": first_step, "time": first.first_violation_time}
        if first is not None
        else None,
        "replay_step": first_step
        if first_step is not None
        else (candidate.steps[-1]["step"] if candidate.steps else _EMPTY_REPLAY_STEP),
        "replay_basis": "first_violation" if first is not None else "episode_end",
        **{side: {key: pair[side][key] for key in ("episode_id", "path")} for side in _VARIANTS},
    }


def _evaluate_rules(
    document: dict[str, Any], config: schema.GateConfig, measurements: _Measurements
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rules, results = [], []
    violations: list[list[tuple[str, checks.SeparationResult]]] = [[] for _ in measurements]
    success_ids = [rule.id for rule in config.rules if rule.metric == "native_success_rate"]
    for rule in config.rules:
        if rule.metric == "native_success_rate":
            rules.append(_success_rule(document, rule))
        else:
            result, episodes = _separation_rule(rule, measurements)
            rules.append(result)
            for index, episode in enumerate(episodes):
                if episode.verdict == "fail":
                    violations[index].append((rule.id, episode))
    for (pair, _, candidate), failed in zip(measurements, violations, strict=True):
        regression = _regression(pair, candidate, failed, success_ids)
        if regression is not None:
            results.append(regression)
    results.sort(
        key=lambda item: (
            item["first_violation"]["time"]
            if item["first_violation"] and item["first_violation"]["time"] is not None
            else math.inf,
            item["key"]["condition"],
            item["key"]["seed"],
        )
    )
    return rules, results


def evaluate_gate(
    compare_document: dict[str, Any], config: schema.GateConfig, *, base_dir: Path
) -> dict[str, Any]:
    """Evaluate recorded evidence and return a deterministic PASS/REGRESSION/BLOCKED document."""
    errors = pairing.comparison_errors(compare_document)
    coverage = _coverage(compare_document, config)
    reasons = errors + _coverage_reasons(coverage)
    # Structural damage cannot be evaluated; legitimate invalid comparisons retain their evidence.
    structurally_invalid = any(reason.startswith("invalid comparison:") for reason in errors)
    source = {} if structurally_invalid else compare_document
    measurements = [] if structurally_invalid else _read_pairs(source, base_dir, reasons)
    rules, regressions = _evaluate_rules(source, config, measurements)
    reasons.extend(
        f"insufficient evidence: {rule['id']}: {rule['reason']}"
        for rule in rules
        if rule["required"] and rule["verdict"] == "unknown"
    )
    verdict, code = "PASS", _EXIT_PASS
    if reasons:
        verdict, code = "BLOCKED", _EXIT_BLOCKED
    elif any(rule["required"] and rule["verdict"] == "fail" for rule in rules):
        verdict, code = "REGRESSION", _EXIT_REGRESSION
    identity = source.get("comparison", {})
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "gate": {"name": config.name, "sha256": config.sha256},
        "comparison": {key: identity.get(key, "") for key in ("name", "sha256")},
        "verdict": verdict,
        "exit_code": code,
        "blocked_reasons": reasons,
        "coverage": coverage,
        "rules": rules,
        "regressions": regressions,
    }


def gate_exit_code(document: dict[str, Any]) -> int:
    """Return the verdict's recorded process exit code."""
    code: int = document["exit_code"]
    return code


def _display_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{DISPLAY_DECIMALS}f}" if isinstance(value, float) else str(value)


def render_verdict(document: dict[str, Any]) -> str:
    """Render portable plain text including evidence, replay anchor, and config identity."""
    lines = [f"Policy Trace Lab gate: {document['verdict']} (exit {gate_exit_code(document)})"]
    for rule in document["rules"]:
        lines.append(
            f"{rule['verdict'].upper()} {rule['id']}: "
            f"candidate={_display_number(rule['candidate_value'])} "
            f"baseline={_display_number(rule['baseline_value'])} "
            f"delta={_display_number(rule['delta'])} "
            f"threshold={_display_number(rule['threshold'])} {rule['units']}"
        )
    if document["verdict"] == "REGRESSION":
        failed = (rule["id"] for rule in document["rules"] if rule["verdict"] == "fail")
        lines.append("failing rules: " + ", ".join(failed))
        if document["regressions"]:
            pair = document["regressions"][0]
            first = pair["first_violation"]
            step, time = (first["step"], first["time"]) if first else (pair["replay_step"], None)
            lines.append(
                f"first violation: {pair['key']['condition']} seed {pair['key']['seed']} "
                f"step {step} t={_display_number(time)}"
            )
    if document["verdict"] == "BLOCKED":
        lines.extend(document["blocked_reasons"])
    identity = document["gate"]
    lines.append(f"gate config: {identity['name']} {identity['sha256'][:_HASH_PREFIX_CHARACTERS]}")
    return "\n".join(lines)
