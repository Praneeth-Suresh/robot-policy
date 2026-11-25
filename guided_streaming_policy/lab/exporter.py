"""Export explicit comparison cells as pinned, canonical episode and run traces.

Resolved configuration contains policy, entry_point and settings only. Variant
names describe the run index, never policy identity or reproducibility hashes.
"""

import importlib
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from guided_streaming_policy.lab import adapters, schema
from guided_streaming_policy.lab.tasks.base import (
    TaskAdapter,
    TrialCrashed,
    TrialRecord,
    TrialRequest,
)

__all__ = ["SetupError", "load_comparison", "default_out_dir", "export_variant"]

_SECONDS_DECIMALS = 3


class SetupError(Exception):
    """Invalid or unavailable inputs prevent starting a comparison variant."""


def load_comparison(path: Path) -> schema.ComparisonConfig:
    """Load comparison input with the exporter's setup-error convention."""
    try:
        return schema.load_comparison_config(path)
    except schema.ConfigError as error:
        raise SetupError(str(error)) from error


def default_out_dir(comparison_path: Path, variant: str) -> Path:
    """Derive the standard output path from the validated comparison name."""
    return Path("out/runs") / load_comparison(comparison_path).name / variant


def _import_hook(path: str) -> Callable[[TrialRequest], TrialRecord]:
    try:
        module_name, attribute = path.split(":")
        hook = getattr(importlib.import_module(module_name), attribute)
        if not callable(hook):
            raise TypeError("entry point is not callable")
        return cast(Callable[[TrialRequest], TrialRecord], hook)
    except Exception as error:
        raise SetupError(f"Cannot import run hook {path!r}: {error}") from error


def _registered_policy(
    adapter: TaskAdapter, variant: schema.VariantSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        spec = adapters.find_policy(adapter, variant.policy)
        status = next(item for item in adapter.policy_statuses() if item.spec.name == spec.name)
        if status.status != "runnable":
            raise SetupError(f"Policy {spec.name}: {status.status}: {status.reason}")
        resolved = adapters.resolve_policy_config(spec, variant.overrides)
        pin = adapters.load_pins(adapter.pins_path())["tasks"][adapter.name]["policies"][spec.name]
        if adapter.current_pins()["policies"][spec.name] != pin:
            raise SetupError(f"pin changed: {spec.name}")
        checkpoints = dict(sorted(pin["checkpoints"].items()))
        policy = {"name": spec.name, "checkpoint_id": ",".join(checkpoints),
                  "checkpoint_sha256": schema.sha256_json(checkpoints)}
        return resolved, policy
    except (KeyError, ValueError, OSError, StopIteration) as error:
        raise SetupError(f"Policy {variant.policy}: {error}") from error


def _resolve_variant(
    adapter: TaskAdapter, variant: schema.VariantSpec,
) -> tuple[dict[str, Any], dict[str, Any], Callable[[TrialRequest], TrialRecord]]:
    if ":" in variant.policy:
        hook = _import_hook(variant.policy)
        resolved = {"policy": variant.policy, "entry_point": variant.policy,
                    "settings": variant.overrides}
        policy = {"name": variant.policy, "checkpoint_id": "unregistered",
                  "checkpoint_sha256": "unregistered"}
        return resolved, policy, hook
    resolved, policy = _registered_policy(adapter, variant)
    return resolved, policy, adapter.run_trial


def _validate_setup(
    adapter: TaskAdapter, config: schema.ComparisonConfig, variant: str, out_dir: Path,
) -> dict[str, Any]:
    if variant not in ("baseline", "candidate"):
        raise SetupError(f"Unknown variant {variant!r}; expected baseline or candidate")
    if config.task != adapter.name:
        raise SetupError(f"Comparison task {config.task!r} does not match adapter {adapter.name!r}")
    probe = adapter.probe()
    if config.simulator_version != probe.simulator_version:
        raise SetupError(f"simulator version mismatch: expected {config.simulator_version!r}, "
                         f"found {probe.simulator_version!r}")
    for condition in config.conditions:
        if condition not in adapter.conditions():
            raise SetupError(f"Unknown condition {condition!r}; supported: {adapter.conditions()}")
        if Path(condition).name != condition or condition in (".", ".."):
            raise SetupError(f"condition must be a single directory name: {condition!r}")
    try:
        if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
            raise SetupError(f"Output directory must be empty; refusing to overwrite {out_dir}")
    except OSError as error:
        raise SetupError(f"Cannot inspect output directory {out_dir}: {error}") from error
    return {"name": probe.simulator_name, "version": probe.simulator_version}


def _call_trial(
    hook: Callable[[TrialRequest], TrialRecord], request: TrialRequest,
) -> tuple[TrialRecord, dict[str, Any]]:
    detail: str | None = None
    try:
        record = hook(request)
    except Exception as error:
        partial = error.record if isinstance(error, TrialCrashed) else None
        record = (partial if partial is not None else
                  TrialRecord((), None, None, False, False, "crashed", {}, None, ()))
        detail = str(error)
    outcome = {"native_success": record.native_success, "terminated": record.terminated,
               "truncated": record.truncated, "reason": record.reason,
               "length": len(record.steps), "native": record.native}
    if detail is not None:
        outcome.update(native_success=None, terminated=False, truncated=False,
                       reason="crashed", detail=detail)
    return record, outcome


def _episode(
    adapter: TaskAdapter, request: TrialRequest, record: TrialRecord,
    outcome: dict[str, Any], metadata: dict[str, Any],
) -> schema.Episode:
    episode_id = f"{request.condition}__seed{request.seed}"
    manifest = {
        "schema_version": schema.SCHEMA_VERSION, "kind": "rollout", "task": adapter.name,
        "episode_id": episode_id, "source_episode_id": episode_id,
        "split": None, "split_source": "none", "timebase": adapter.timebase(),
        "observation_schema": adapter.observation_schema(),
        "action_schema": adapter.action_schema(),
        "fields_present": list(record.fields_present), "condition": request.condition,
        "seed": request.seed, "reset_state_fingerprint": record.reset_state_fingerprint,
        "outcome": outcome, **metadata,
    }
    if record.geometry is not None:
        manifest["geometry"] = record.geometry
    return schema.Episode(manifest, record.steps)


def _export_cells(
    adapter: TaskAdapter, config: schema.ComparisonConfig, resolved: dict[str, Any],
    hook: Callable[[TrialRequest], TrialRecord], out_dir: Path,
    metadata: dict[str, Any], log: Callable[[str], None],
) -> list[dict[str, Any]]:
    cells = []
    for condition in config.conditions:
        for seed in config.seeds:
            request = TrialRequest(resolved["policy"], dict(resolved["settings"]), condition, seed)
            started = perf_counter()
            record, outcome = _call_trial(hook, request)
            elapsed = perf_counter() - started
            episode = _episode(adapter, request, record, outcome, metadata)
            schema.write_episode(episode, out_dir)
            cells.append({"episode_id": episode.manifest["episode_id"],
                          "condition": condition, "seed": seed})
            log(f"{condition} seed={seed} native_success={outcome['native_success']} "
                f"steps={len(record.steps)} seconds={elapsed:.{_SECONDS_DECIMALS}f} "
                f"reason={outcome['reason']}" +
                (f" detail={outcome['detail']}" if "detail" in outcome else ""))
    return cells


def export_variant(
    adapter: TaskAdapter, comparison_path: Path, variant: str, out_dir: Path, *,
    command: str, created_at: str | None = None,
    runner: Callable[[TrialRequest], TrialRecord] | None = None,
    log: Callable[[str], None] = lambda _message: None,
) -> Path:
    """Validate setup, then record every listed cell, retaining failed trials."""
    config = load_comparison(comparison_path)
    simulator = _validate_setup(adapter, config, variant, out_dir)
    resolved, policy, hook = _resolve_variant(adapter, config.variants[variant])
    provenance = schema.make_provenance(command, created_at)
    metadata = {"policy": policy, "simulator": simulator, "provenance": provenance,
                "resolved_config": {"path": "resolved_config.json",
                                    "sha256": schema.sha256_json(resolved)}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(schema.canonical_json(resolved), encoding="utf-8")
    cells = _export_cells(adapter, config, resolved, hook if runner is None else runner,
                          out_dir, metadata, log)
    schema.write_run_manifest({
        "schema_version": schema.SCHEMA_VERSION, "run_id": f"{config.name}__{variant}",
        "task": adapter.name, "comparison": {"name": config.name, "sha256": config.sha256},
        "variant": variant, "policy": policy, "simulator": simulator, "provenance": provenance,
        "episodes": cells,
    }, out_dir)
    return out_dir
