"""Conformance kit for task packs: standard library only, no network, no simulator needed.

`run_conformance(adapter)` returns one pass, fail or skip result per check, and
`TaskContractTests` turns them into unittest sub-tests. A pack declares its
expected fixture results in `<fixture root>/cases.json`:

    {"schema_version": "0.1", "demo_exit": 1, "cases": {
       "clean": {"path": "demos/clean", "doctor_exit": 0, "doctor_checks": []},
       "seeded_regression": {"path": "scenarios/seeded_regression", "compare_exit": 0,
           "gate_exit": 1, "coverage": {"completed": 4, "incompatible": 0, "missing": 0},
           "rules": {"separation_proxy": "fail"}}, ...}}

The ten case names are in `FIXTURE_CASES`. Checks that need a simulator or a fresh
interpreter are skipped, with the reason, when they cannot run.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from guided_streaming_policy.lab import adapters, doctor, gate, pairing, schema
from guided_streaming_policy.lab.tasks.base import (
    CONTRACT_VERSION,
    FixtureSet,
    TaskAdapter,
    TrialCrashed,
    TrialRequest,
)

__all__ = ["CheckResult", "FIXTURE_CASES", "HEAVY_MODULES", "TaskContractTests", "run_conformance"]

PASS, FAIL, SKIP = "pass", "fail", "skip"
DOCTOR_CASES = ("clean", "no_split", "corrupt", "leakage")
SCENARIO_CASES = ("seeded_regression", "two_valid_trajectories", "missing_signal",
                  "mismatched_reset", "truncated", "missing_pairs")
FIXTURE_CASES = DOCTOR_CASES + SCENARIO_CASES
HEAVY_MODULES = ("numpy", "torch", "scipy", "pymunk", "zarr", "gym", "gymnasium", "mujoco",
                 "robosuite", "cv2", "pygame", "diffusers")
CASES_FILENAME = "cases.json"
_COVERAGE_KEYS = ("completed", "incompatible", "missing")
_SUBPROCESS_TIMEOUT_SECONDS = 120
_DETAIL_CHARACTERS = 300
_STDERR_TAIL_CHARACTERS = 150
_IMPORT_PROBE = """
import importlib, json, sys
adapter = getattr(importlib.import_module({module!r}), {qualname!r})()
dependencies = list(adapter.probe().dependencies)
adapter.policies(); adapter.policy_statuses(); adapter.conditions()
loaded = [name for name in set({heavy!r}) | set(dependencies) if name in sys.modules]
print(json.dumps(sorted(loaded)))
"""
_DEMO_RUN = """
import importlib, sys
from guided_streaming_policy.lab.tasks import registry
factory = getattr(importlib.import_module({module!r}), {qualname!r})
if {name!r} not in registry.task_names():
    registry.register({name!r}, factory)
from guided_streaming_policy.cli import main
raise SystemExit(main(["lab", "demo", "--task", {name!r}, "--out", {out!r}]))
"""


@dataclass(frozen=True)
class CheckResult:
    """One conformance check: its name, `pass`, `fail` or `skip`, and the reason."""

    name: str
    status: str
    detail: str = ""


def _fail(name: str, detail: str) -> CheckResult:
    return CheckResult(name, FAIL, detail[:_DETAIL_CHARACTERS])


def _attempt(name: str, check: Callable[[], CheckResult]) -> CheckResult:
    try:
        return check()
    except Exception as error:  # a broken pack must produce a failed check, not a crash
        return _fail(name, f"{type(error).__name__}: {error}")


def _contract_version(adapter: TaskAdapter) -> CheckResult:
    version = getattr(adapter, "contract_version", None)
    if version == CONTRACT_VERSION:
        return CheckResult("contract_version", PASS, f"version {version}")
    return _fail("contract_version", f"declares {version!r}; the lab requires {CONTRACT_VERSION}")


def _conditions(adapter: TaskAdapter) -> CheckResult:
    specs = adapter.conditions()
    names = [spec.name for spec in specs]
    if not names:
        return _fail("conditions", "no conditions declared")
    if len(set(names)) != len(names) or not all(names):
        return _fail("conditions", f"names must be unique and non-empty: {names}")
    if not all(spec.description for spec in specs):
        return _fail("conditions", "every condition needs a description")
    return CheckResult("conditions", PASS, ", ".join(names))


def _importable_class(adapter: TaskAdapter) -> tuple[str, str] | str:
    """Return (module, qualname) usable in a fresh interpreter, or the reason it is not."""
    cls = type(adapter)
    if cls.__module__ == "__main__" or "<locals>" in cls.__qualname__:
        return "the adapter class cannot be imported from a fresh interpreter"
    parameters = inspect.signature(cls).parameters.values()
    if any(p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
           for p in parameters):
        return "the adapter constructor needs arguments"
    return cls.__module__, cls.__qualname__


def _child_environment() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}


def _run_child(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=_child_environment(), timeout=_SUBPROCESS_TIMEOUT_SECONDS)


def _no_heavy_imports(adapter: TaskAdapter) -> CheckResult:
    target = _importable_class(adapter)
    if isinstance(target, str):
        return CheckResult("no_heavy_imports", SKIP, target)
    done = _run_child(_IMPORT_PROBE.format(module=target[0], qualname=target[1],
                                           heavy=HEAVY_MODULES))
    if done.returncode != 0:
        tail = done.stderr.strip().splitlines()[-1:] or ["the probe process failed"]
        return _fail("no_heavy_imports", tail[0])
    loaded = json.loads(done.stdout.strip().splitlines()[-1])
    if loaded:
        return _fail("no_heavy_imports", "importing the pack or calling probe(), policies() "
                     f"or conditions() imported {', '.join(loaded)}")
    return CheckResult("no_heavy_imports", PASS, "importing and probing load nothing heavy")


def _policy_pins(adapter: TaskAdapter) -> CheckResult:
    specs = adapter.policies()
    if not specs:
        return _fail("policy_pins", "no reference policies declared")
    pins = adapters.load_pins(adapter.pins_path())["tasks"][adapter.name]["policies"]
    current = adapter.current_pins()["policies"]
    for spec in specs:
        if not spec.paper.strip():
            return _fail("policy_pins", f"{spec.name} has no paper string")
        pin = pins.get(spec.name)
        if pin is None:
            return _fail("policy_pins", f"{spec.name} has no pin")
        expected = adapters.config_sha256(adapters.resolve_policy_config(spec, {}))
        if pin["config_sha256"] != expected or set(pin["checkpoints"]) != set(spec.checkpoints):
            return _fail("policy_pins", f"pin changed: {spec.name}")
        for path, digest in current.get(spec.name, {}).get("checkpoints", {}).items():
            if digest is not None and digest != pin["checkpoints"].get(path):
                return _fail("policy_pins", f"pin changed: {spec.name} checkpoint {path}")
    return CheckResult("policy_pins", PASS, f"{len(specs)} policies pinned, each with a paper")


def _case_path(fixtures: FixtureSet, entry: dict[str, Any]) -> Path:
    path = (fixtures.root / str(entry["path"])).resolve()
    if fixtures.root.resolve() not in path.parents:
        raise ValueError(f"case path escapes the fixture root: {entry['path']}")
    return path


def _doctor_case(name: str, entry: dict[str, Any], path: Path) -> CheckResult:
    report = doctor.check_episodes(adapters.read_traces(path), input_path=str(path),
                                   input_checksum="")
    checks = sorted({finding["check"] for finding in report["findings"]})
    code = doctor.exit_code(report)
    if code != entry["doctor_exit"] or checks != sorted(entry["doctor_checks"]):
        return _fail(f"fixture:{name}", f"Doctor gave exit {code} with {checks}; expected "
                     f"exit {entry['doctor_exit']} with {sorted(entry['doctor_checks'])}")
    return CheckResult(f"fixture:{name}", PASS, f"Doctor exit {code}")


def _scenario_actual(
    adapter: TaskAdapter, path: Path,
) -> dict[str, Any]:
    config = schema.load_comparison_config(path / "comparison.json")
    adapters.validate_conditions(adapter, config.conditions)
    with tempfile.TemporaryDirectory() as temporary:
        comparison = pairing.build_comparison(path / "baseline", path / "candidate", config,
                                              out_dir=Path(temporary))
        result = gate.evaluate_gate(comparison, schema.load_gate_config(path / "gate.json"),
                                    base_dir=Path(temporary))
    return {"compare_exit": pairing.compare_exit_code(comparison),
            "gate_exit": gate.gate_exit_code(result),
            "coverage": {key: comparison["coverage"][key] for key in _COVERAGE_KEYS},
            "rules": {rule["id"]: rule["verdict"] for rule in result["rules"]}}


def _scenario_mismatches(actual: dict[str, Any], entry: dict[str, Any]) -> list[str]:
    wrong = [f"{key}: got {actual[key]}, expected {entry[key]}"
             for key in ("compare_exit", "gate_exit", "coverage") if actual[key] != entry[key]]
    wrong.extend(f"rule {rule}: got {actual['rules'].get(rule)}, expected {verdict}"
                 for rule, verdict in entry["rules"].items()
                 if actual["rules"].get(rule) != verdict)
    return wrong


def _run_case(
    adapter: TaskAdapter, fixtures: FixtureSet, name: str, entry: dict[str, Any],
) -> CheckResult:
    path = _case_path(fixtures, entry)
    if name in DOCTOR_CASES:
        return _doctor_case(name, entry, path)
    wrong = _scenario_mismatches(_scenario_actual(adapter, path), entry)
    if wrong:
        return _fail(f"fixture:{name}", "; ".join(wrong))
    return CheckResult(f"fixture:{name}", PASS, f"compare exit {entry['compare_exit']}, "
                       f"gate exit {entry['gate_exit']}")


def _fixture_cases(adapter: TaskAdapter) -> list[CheckResult]:
    try:
        fixtures = adapter.fixtures()
        document = json.loads((fixtures.root / CASES_FILENAME).read_text(encoding="utf-8"))
        cases: dict[str, Any] = document["cases"]
    except Exception as error:  # missing or unreadable cases.json
        return [_fail("fixture_cases", f"cannot read {CASES_FILENAME}: {error}")]
    results = []
    for name in FIXTURE_CASES:
        if name not in cases:
            results.append(_fail(f"fixture:{name}", f"{CASES_FILENAME} has no case {name!r}"))
            continue
        results.append(_attempt(f"fixture:{name}", functools.partial(
            _run_case, adapter, fixtures, name, cases[name])))
    return results


def _schema_sample(adapter: TaskAdapter) -> CheckResult:
    run = adapter.fixtures().scenario / "baseline"
    episodes = list(adapters.read_traces(run))
    for episode in episodes:
        manifest = episode.manifest
        for key, expected in (("observation_schema", adapter.observation_schema()),
                              ("action_schema", adapter.action_schema()),
                              ("timebase", adapter.timebase())):
            if manifest.get(key) != expected:
                return _fail("schemas_sample", f"{key} differs from the recorded sample")
        missing = [name for name in schema.required_manifest_fields(manifest["kind"])
                   if name not in manifest]
        if missing:
            return _fail("schemas_sample", f"recorded manifest lacks {', '.join(missing)}")
    report = doctor.check_episodes(episodes, input_path=str(run), input_checksum="")
    if report["summary"]["counts"]["error"]:
        return _fail("schemas_sample", "Doctor reports errors on the recorded sample")
    return CheckResult("schemas_sample", PASS, f"{len(episodes)} recorded episodes validate")


def _first_runnable(adapter: TaskAdapter) -> TrialRequest | str:
    statuses = adapter.policy_statuses()
    runnable = [status.spec for status in statuses if status.status == "runnable"]
    if not runnable:
        reasons = "; ".join(f"{status.spec.name}: {status.reason}" for status in statuses)
        return f"no runnable policy ({reasons})"[:_DETAIL_CHARACTERS]
    condition = next(spec.name for spec in adapter.conditions() if spec.runnable)
    settings = adapters.resolve_policy_config(runnable[0], {})["settings"]
    return TrialRequest(runnable[0].name, settings, condition, 0)


def _fingerprint(adapter: TaskAdapter, run_trials: bool) -> CheckResult:
    if not run_trials:
        return CheckResult("reset_fingerprint", SKIP, "trial runs were disabled by the caller")
    request = _first_runnable(adapter)
    if isinstance(request, str):
        return CheckResult("reset_fingerprint", SKIP, request)
    try:
        first, second = adapter.run_trial(request), adapter.run_trial(request)
    except TrialCrashed as error:
        return _fail("reset_fingerprint", f"trial crashed: {error}")
    if first.reset_state_fingerprint is None:
        return _fail("reset_fingerprint", "the run hook recorded no reset fingerprint")
    if first.reset_state_fingerprint != second.reset_state_fingerprint:
        return _fail("reset_fingerprint", "two runs with seed 0 gave different fingerprints")
    return CheckResult("reset_fingerprint", PASS, f"stable for {request.policy}, seed 0")


def _offline_demo(adapter: TaskAdapter) -> CheckResult:
    target = _importable_class(adapter)
    if isinstance(target, str):
        return CheckResult("offline_demo", SKIP, target)
    cases = json.loads((adapter.fixtures().root / CASES_FILENAME).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as temporary:
        done = _run_child(_DEMO_RUN.format(module=target[0], qualname=target[1],
                                           name=adapter.name, out=str(Path(temporary) / "demo")))
    if done.returncode != cases["demo_exit"]:
        return _fail("offline_demo", f"gsp lab demo exited {done.returncode}; declared "
                     f"{cases['demo_exit']}. {done.stderr.strip()[-_STDERR_TAIL_CHARACTERS:]}")
    return CheckResult("offline_demo", PASS, f"gsp lab demo exits {done.returncode} as declared")


def run_conformance(adapter: TaskAdapter, *, run_trials: bool = True) -> list[CheckResult]:
    """Run every conformance check and return one result per check, in a fixed order."""
    results = [
        _attempt("contract_version", lambda: _contract_version(adapter)),
        _attempt("conditions", lambda: _conditions(adapter)),
        _attempt("no_heavy_imports", lambda: _no_heavy_imports(adapter)),
        _attempt("policy_pins", lambda: _policy_pins(adapter)),
        _attempt("schemas_sample", lambda: _schema_sample(adapter)),
    ]
    results.extend(_fixture_cases(adapter))
    results.append(_attempt("reset_fingerprint", lambda: _fingerprint(adapter, run_trials)))
    results.append(_attempt("offline_demo", lambda: _offline_demo(adapter)))
    return results


class TaskContractTests(unittest.TestCase):
    """Subclass, set `make_adapter`, and every conformance check becomes a sub-test.

    Set `run_trials = False` to skip the simulator-backed fingerprint check.
    """

    run_trials = True

    @classmethod
    def setUpClass(cls) -> None:
        if cls is TaskContractTests:
            raise unittest.SkipTest("TaskContractTests is a base class; subclass it")

    def make_adapter(self) -> TaskAdapter:
        """Return the pack under test."""
        raise NotImplementedError

    def test_conformance(self) -> None:
        for result in run_conformance(self.make_adapter(), run_trials=self.run_trials):
            with self.subTest(check=result.name):
                if result.status == FAIL:
                    self.fail(result.detail)
                if result.status == SKIP:
                    self.skipTest(result.detail)
