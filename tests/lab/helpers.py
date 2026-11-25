"""Deterministic synthetic trace builders shared by lab tests."""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from guided_streaming_policy.lab import adapters, schema
from guided_streaming_policy.lab.tasks.base import (
    PolicySpec,
    PolicyStatus,
    ProbeResult,
    TrialCrashed,
    TrialRecord,
    TrialRequest,
    TunableSpec,
)

__all__ = ["demo_episode", "rollout_episode", "write_traces", "FakeTask", "comparison_document",
           "comparison_config", "geometry_rollout", "rollout_run"]

_DT_SECONDS = 0.1
_TIME_DECIMALS = 6
_CREATED_AT = "2026-01-01T00:00:00Z"


def _field(shape: list[int], units: str = "px") -> dict[str, Any]:
    return {"shape": shape, "dtype": "float64", "units": units, "frame": "window"}


def demo_episode(
    episode_id: str, n_steps: int = 6, offset: float = 0.0,
    split: str | None = None, split_source: str = "none",
) -> schema.Episode:
    """Build a complete demonstration with simple absolute target actions."""
    manifest = {
        "schema_version": schema.SCHEMA_VERSION, "kind": "demonstration",
        "episode_id": episode_id, "source_episode_id": episode_id, "task": "pusht",
        "dataset": {"name": "synthetic", "revision": "fixture-v1", "checksum": "fixture"},
        "split": split, "split_source": split_source, "synthetic": True,
        "timebase": {"unit": "s", "mode": "fixed_dt", "dt": _DT_SECONDS},
        "observation_schema": {"fields": {
            "agent_pos": _field([2]), "block_pos": _field([2]),
            "block_angle": _field([], "rad"),
        }},
        "action_schema": {"convention": "absolute", "fields": {"target": _field([2])}},
        "fields_present": ["obs", "action_executed"],
        "provenance": {"tool_version": "fixture", "git_commit": "fixture",
                       "command": "fixture builder", "created_at": _CREATED_AT},
    }
    steps = tuple({
        "step": index, "t": round(index * _DT_SECONDS, _TIME_DECIMALS),
        "obs": {"agent_pos": [offset + index, offset + index + 1.0],
                "block_pos": [offset + 10.0, offset + 20.0], "block_angle": 0.0},
        "action_executed": {"target": [offset + index + 2.0, offset + index + 3.0]},
    } for index in range(n_steps))
    return schema.Episode(manifest, steps)


def rollout_episode(
    episode_id: str, n_steps: int = 6, offset: float = 0.0,
    split: str | None = None, split_source: str = "none", *,
    policy: dict[str, Any] | None = None, outcome: dict[str, Any] | None = None,
    obstacle: dict[str, Any] | None = None, condition: str = "static", seed: int = 0,
) -> schema.Episode:
    """Build a complete rollout; supplied outcome values override clean defaults."""
    episode = demo_episode(episode_id, n_steps, offset, split, split_source)
    manifest = episode.manifest
    del manifest["dataset"]
    manifest.update({
        "kind": "rollout", "condition": condition, "seed": seed,
        "reset_state_fingerprint": "fixture-reset",
        "policy": policy if policy is not None else {
            "name": "fixture", "checkpoint_id": "fixture", "checkpoint_sha256": "fixture"},
        "resolved_config": {"path": "fixture.json", "sha256": "fixture"},
        "simulator": {"name": "fixture", "version": "1"},
        "outcome": {"native_success": False, "terminated": False, "truncated": True,
                    "reason": "step_limit", "length": n_steps, **(outcome or {})},
    })
    if obstacle is not None:
        manifest["geometry"] = {"obstacles": [obstacle]}
        manifest["fields_present"].append("geometry")
    return episode


def write_traces(directory: Path, episodes: Iterable[schema.Episode]) -> None:
    """Write each fixture episode below directory without replacing earlier traces."""
    directory.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        schema.write_episode(episode, directory)


class FakeTask:
    """Deterministic complete adapter with temporary, reproducible checkpoint pins."""

    name = "toy"
    crash_seeds: tuple[int, ...] = ()
    unavailable_reason: str | None = None
    changed_pin = False

    def __init__(self, root: Path) -> None:
        self._pins = root / "pins.json"
        self._pins.write_text(schema.canonical_json({
            "tasks": {self.name: self.current_pins()},
        }), encoding="utf-8")

    def policies(self) -> tuple[PolicySpec, ...]:
        """Return two policies with the same scalar tunable."""
        return tuple(PolicySpec(name, "toy policy", "fixture", "toy:run", ("toy.ckpt",),
                                (TunableSpec("gain", 1.0, "fixture gain"),))
                     for name in ("toy/a", "toy/b"))

    def policy_statuses(self) -> tuple[PolicyStatus, ...]:
        """Expose a controllable unavailable status."""
        return tuple(PolicyStatus(spec, "unavailable" if self.unavailable_reason else "runnable",
                                  self.unavailable_reason) for spec in self.policies())

    def probe(self) -> ProbeResult:
        """Return fixed simulator identity without external dependencies."""
        return ProbeResult("toy-sim", "1.0", {}, {})

    def conditions(self) -> tuple[str, ...]:
        """Return the one supported fixture condition."""
        return ("c1",)

    def observation_schema(self) -> dict[str, Any]:
        """Reuse the standard synthetic observation builder."""
        return dict(demo_episode("toy").manifest["observation_schema"])

    def action_schema(self) -> dict[str, Any]:
        """Reuse the standard synthetic action builder."""
        return dict(demo_episode("toy").manifest["action_schema"])

    def timebase(self) -> dict[str, Any]:
        """Use the fixture control period in seconds."""
        return {"unit": "s", "mode": "fixed_dt", "dt": _DT_SECONDS}

    def read_demonstrations(self, path: Path) -> Iterator[schema.Episode]:
        """Read already serialized fixtures."""
        return adapters.read_traces(path)

    def pins_path(self) -> Path:
        """Return the temporary pin file."""
        return self._pins

    def current_pins(self) -> dict[str, Any]:
        """Expose a controllable checkpoint mismatch."""
        return {"policies": {spec.name: {
            "checkpoints": {"toy.ckpt": "changed" if self.changed_pin else "fixture-digest"},
            "config_sha256": adapters.config_sha256(adapters.resolve_policy_config(spec, {})),
        } for spec in self.policies()}}

    def run_trial(self, request: TrialRequest) -> TrialRecord:
        """Record deterministic telemetry and optionally fail with a partial record."""
        offset = request.seed + float(request.settings.get("gain", 1.0))
        offset += float(request.policy == "toy/b")
        episode = demo_episode("toy", offset=offset)
        record = TrialRecord(episode.steps, schema.sha256_json([request.seed]), True, True,
                             False, "complete", {"score": offset}, None,
                             ("obs", "action_executed", "reset_state_fingerprint"))
        if request.seed in self.crash_seeds:
            raise TrialCrashed("fixture crash", record)
        return record


def comparison_document(seeds: tuple[int, ...] = (0, 1)) -> dict[str, Any]:
    """Build the minimal explicit toy comparison matrix."""
    return {
        "schema_version": "0.1", "name": "toy-comparison", "task": "toy",
        "simulator_version": "1.0", "matrix": {"conditions": ["c1"], "seeds": list(seeds)},
        "variants": {"baseline": {"policy": "toy/a"},
                     "candidate": {"policy": "toy/a", "overrides": {"gain": 2.0}}},
        "intended_differences": ["overrides.gain"],
    }


def comparison_config(
    path: Path, *, seeds: tuple[int, ...] = (0, 1), document: dict[str, Any] | None = None,
) -> schema.ComparisonConfig:
    """Write canonical explicit comparison JSON and return the validated config."""
    path.write_text(schema.canonical_json(comparison_document(seeds) if document is None
                                          else document), encoding="utf-8")
    return schema.load_comparison_config(path)


def geometry_rollout(
    episode_id: str, *, seed: int = 0, outcome_kind: str = "success", offset: float = 0.0,
) -> schema.Episode:
    """Build fixed-start trajectories with known geometry and native outcome variants."""
    outcomes = {
        "success": {"native_success": True, "terminated": True,
                    "truncated": False, "reason": "complete"},
        "native_fail": {"native_success": False, "terminated": True,
                        "truncated": False, "reason": "complete"},
        "truncated": {"native_success": False, "terminated": False,
                      "truncated": True, "reason": "step_limit"},
        "crashed": {"native_success": None, "terminated": False,
                    "truncated": False, "reason": "crashed"},
    }
    return rollout_episode(episode_id, seed=seed, condition="c1", offset=offset,
                           outcome=outcomes[outcome_kind], obstacle={
                               "id": "point", "kind": "point", "position": [0.0, 0.0],
                               "radius": 0.0, "units": "px", "frame": "window",
                               "source": "fixture"})


def rollout_run(
    directory: Path, variant: str, config: schema.ComparisonConfig,
    episodes: Iterable[schema.Episode], *, resolved_config: dict[str, Any] | None = None,
) -> Path:
    """Write a complete run with matching resolved-config digests, without mutating inputs."""
    import copy

    spec = config.variants[variant]
    resolved = (resolved_config if resolved_config is not None else {
        "policy": spec.policy, "entry_point": "toy:run", "settings": {"gain": 1.0,
                                                                    **spec.overrides}})
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "resolved_config.json").write_text(schema.canonical_json(resolved),
                                                   encoding="utf-8")
    policy = {"name": spec.policy, "checkpoint_id": "fixture", "checkpoint_sha256": "fixture"}
    simulator = {"name": "fixture", "version": config.simulator_version}
    cells = []
    for original in episodes:
        episode = copy.deepcopy(original)
        episode.manifest.update(task=config.task, policy=policy, simulator=simulator,
                                resolved_config={"path": "resolved_config.json",
                                                 "sha256": schema.sha256_json(resolved)})
        schema.write_episode(episode, directory)
        cells.append({key: episode.manifest[key] for key in ("episode_id", "condition", "seed")})
    schema.write_run_manifest({
        "schema_version": schema.SCHEMA_VERSION, "run_id": f"{config.name}__{variant}",
        "task": config.task, "comparison": {"name": config.name, "sha256": config.sha256},
        "variant": variant, "policy": policy, "simulator": simulator,
        "provenance": {"tool_version": "fixture", "git_commit": "fixture",
                       "command": "fixture builder", "created_at": _CREATED_AT},
        "episodes": cells,
    }, directory)
    return directory
