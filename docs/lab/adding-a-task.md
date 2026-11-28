# Adding a simulator

A **task pack** is one simulator, its task definition, its named conditions and its reference policies, behind the `TaskAdapter` contract in `guided_streaming_policy/lab/tasks/base.py`. The lab core (trace, Doctor, paired evaluation, gate, replay) does not change when you add one. A contract test, `tests/lab/test_toy_task_contract.py`, scaffolds a pack into a temporary directory and runs the whole offline pipeline on it, so any Push-T assumption in the core fails that test.

This page uses "simulator" in headings and "task pack" in the body. It is not about `gsp list-tasks` or `gsp run <script>`: those drive the legacy research scripts.

## Two routes

| Route | Use it when | How it ships |
| --- | --- | --- |
| **A. A separate pip-installable pack** (the default) | You add a simulator of your own, in your repository | A package that registers an entry point. Nothing in this repository changes. |
| **B. A built-in pack in this repository** | You maintain the lab and the simulator belongs with it, as Push-T does | A folder under `guided_streaming_policy/lab/tasks/<task>/`, one line in `pyproject.toml`, one README row |

### Route A: a separate pack

Run these in an environment where this repository is installed (`pip install .` from a clone):

```bash
gsp lab tasks new rail --out rail-pack      # scaffold a working pack
pip install ./rail-pack                     # the pack registers itself
gsp lab tasks list                          # rail: ready
gsp lab tasks check rail                    # the conformance suite
gsp lab demo --task rail --out out/rail-demo
cd rail-pack
gsp lab run --comparison configs/comparison.json --variant baseline --out out/runs/baseline
gsp lab run --comparison configs/comparison.json --variant candidate --out out/runs/candidate
gsp lab compare out/runs/baseline out/runs/candidate --comparison configs/comparison.json --out out/compare
gsp lab gate out/compare/compare.json --gate configs/gate.json --out out/gate
```

The scaffold is a cart on a one-dimensional rail with a virtual wall (`rail/simulator.py`, about seventy lines of Python). Replace the simulator and the run hook (`rail/adapter.py`), keep `probe()` and `policies()` free of heavy imports, regenerate the fixtures and pins with `python tools/build_fixtures.py`, and run `gsp lab tasks check <name>` until it passes. The pack depends on `guided-streaming-policy>=0.1`, which is installed from this repository, not from an index.

What the scaffold contains:

| Path | Purpose |
| --- | --- |
| `pyproject.toml` | The `gsp_lab.tasks` entry point and the package data (`pins.json`, `fixtures/**`) |
| `<module>/simulator.py` | The simulator |
| `<module>/adapter.py` | The `TaskAdapter`: policies, schemas, run hook, demonstration reader, fixtures |
| `<module>/pins.json`, `<module>/fixtures/` | Pins and the packaged fixture set, found with `importlib.resources` |
| `configs/comparison.json`, `configs/gate.json` | A comparison and its release rule |
| `tools/build_fixtures.py` | Regenerates the fixtures and pins; the scenario runs are real simulator runs written by the lab's exporter |
| `tests/test_contract.py` | A contract test built on `guided_streaming_policy.lab.testing` |

### Route B: a built-in pack

A maintainer-only route. Put the adapter under `guided_streaming_policy/lab/tasks/<task>/`, its fixtures and pins next to it as package data, and register it in this repository's `pyproject.toml` exactly as a plugin would:

```toml
[project.entry-points."gsp_lab.tasks"]
pusht = "guided_streaming_policy.lab.tasks.pusht.adapter:PushTTask"
```

There is no registry dictionary to edit. A built-in pack may import `schema`, `adapters`, `tasks.base` and its own modules, and the core never imports it ([architecture](architecture.md)). A contribution may touch only its pack folder, its tests, one `pyproject.toml` line, one README row and one docs page.

## The contract

`TaskAdapter` is a `Protocol`. The contract version is `CONTRACT_VERSION = 1` in `tasks/base.py`; a pack declares `contract_version = CONTRACT_VERSION`, and the lab refuses a pack whose number differs and names both numbers. Any change to the contract bumps the number.

| Member | Used by |
| --- | --- |
| `name`, `contract_version` | the registry and every `--task` flag |
| `conditions()` | the named scenarios: a tuple of `ConditionSpec(name, description, runnable=True)`. A comparison config may list only these names (checked in `run`, `compare` and `gate`) |
| `read_demonstrations(path)` | `gsp lab doctor`: native data to episode traces, with the real split provenance if the source has one |
| `run_trial(request)` | `gsp lab run`: run one policy on one matrix cell with step recording; raise `TrialCrashed` when the trial fails |
| `TrialRecord.reset_state_fingerprint` | pair matcher: a hash of the actual starting state, or `None` |
| `TrialRecord.native_success`, `terminated`, `truncated`, `reason`, `native` | checks, Doctor: copied from the simulator, never redefined |
| `observation_schema()`, `action_schema()`, `timebase()` | Doctor and schema validation |
| `TrialRecord.geometry` and the gate rule's `signal` | the separation check: the geometry and positions it needs |
| `probe()` | simulator name and version, assets and dependencies, without importing them |
| `policies()`, `policy_statuses()`, `pins_path()`, `current_pins()` | `gsp lab policies`, comparison configs, pin checks: every reference method registered by name and pinned by checkpoint hash and resolved-config hash |
| `fixtures()` | `gsp lab demo` and the conformance kit: a `FixtureSet` of synthetic inputs and the `cases.json` that declares what each should give |

`probe()`, `policies()`, `policy_statuses()` and `conditions()` run whenever packs are listed, so they must not import the simulator, `torch` or `numpy`: use `importlib.util.find_spec`. The conformance kit checks this in a fresh interpreter.

### Discovery

The registry (`guided_streaming_policy/lab/tasks/registry.py`) reads the entry-point group `gsp_lab.tasks` from installed package metadata: the entry name is the task name and the value is `module:Class`. It keeps no list of its own.

- Listing a pack reads metadata only. Only `get_task` imports a pack, one at a time.
- A pack that fails to import, construct or match the contract raises `TaskLoadError` for that name alone. `gsp lab tasks list` shows it as `broken: <reason>`, and every other command keeps working.
- Two installed distributions claiming one name are reported as `task 'x' is claimed by a 1.0 and b 2.0`.
- `registry.register(name, factory)` adds a task in-process, for tests and the conformance kit.

## The ten fixture cases

A pack ships a synthetic fixture set that the offline demo and the kit use. `fixtures().root` holds `cases.json`, which declares the expected result of each case; the kit runs Doctor, compare and gate on the case and compares.

| Case | Kind | Expected result |
| --- | --- | --- |
| `clean` | demonstrations | Doctor exit 0, no findings |
| `no_split` | demonstrations | Doctor exit 0, only `split.not_checked` |
| `corrupt` | demonstrations | Doctor exit 1: a NaN, a wrong shape, a non-monotonic `t`, each at its episode and step |
| `leakage` | demonstrations | Doctor exit 1: `split.leakage` |
| `seeded_regression` | paired runs | gate exit 1 naming the failing rule; the native success of the violating pair stays `true` |
| `two_valid_trajectories` | paired runs | gate exit 0: different but valid trajectories from one start |
| `missing_signal` | paired runs | the separation rule is `unknown`, gate exit 2 |
| `mismatched_reset` | paired runs | incompatible pairs beyond the limit, gate exit 2 |
| `truncated` | paired runs | pairs stay `completed`; the failure is counted |
| `missing_pairs` | paired runs | pairs listed as `missing`, gate exit 2 |

```json
{"schema_version": "0.1", "demo_exit": 1, "cases": {
  "clean": {"path": "demos/clean", "doctor_exit": 0, "doctor_checks": []},
  "seeded_regression": {"path": "scenarios/seeded_regression", "compare_exit": 0,
    "gate_exit": 1, "coverage": {"completed": 4, "incompatible": 0, "missing": 0},
    "rules": {"success_not_worse": "pass", "separation_proxy": "fail"}}}}
```

## Steps and exit checks

Work through these in order. Each step has an exit check, and the next step does not start until it passes.

1. **Inspect.** Record the real observation and action spaces, the native success metric, the termination rules, the available telemetry and the required assets, each with a source reference (file and line). Use [baseline](baseline.md) as the model. *Exit check:* every field a later step uses has a source reference.
2. **Declare the schema.** Write `observation_schema()`, `action_schema()` and `timebase()` with units, frame and the action convention (absolute or delta). *Exit check:* `gsp lab tasks check` passes `schemas_sample`.
3. **Read demonstrations.** Write `read_demonstrations` and the `clean`, `no_split`, `corrupt` and `leakage` fixtures. *Exit check:* the four `fixture:` checks for those cases pass.
4. **Export rollouts.** Write `run_trial` and the reset fingerprint. Record the actual reset state, not only the seed. *Exit check:* `reset_fingerprint` passes: two runs with the same seed give the same fingerprint.
5. **Define checks.** Choose the native success metric and one evidenced process check, register and pin the reference policies, and write the gate config before you look at any results. *Exit check:* `missing_signal` gives `unknown`, not a guess; `policy_pins` passes.
6. **Add fixtures and the offline demo.** Add the six scenario cases. *Exit check:* every `fixture:` check passes and `offline_demo` shows `gsp lab demo --task <name>` exiting with the declared code.
7. **Run it for real.** Run one reference policy against a variant with the real simulator and capture the output. If it cannot run, stop and ask. *Exit check:* the result is on the pack's page, labelled "verified in simulation" or "fixture only".
8. **Wire up and document.** Install the pack, extend the offline CI job, add the README row. *Exit check:* `gsp lab tasks check <name>` exits 0, and for a built-in pack the architecture and hygiene tests pass with no change to the core.

## The conformance checks

`gsp lab tasks check <name>` and the `TaskContractTests` mixin run the same checks. The kit is `guided_streaming_policy/lab/testing.py` (standard library only); a pack's own test is one class:

```python
from guided_streaming_policy.lab import testing

class MyPackTests(testing.TaskContractTests):
    def make_adapter(self):
        return MyTask()
```

| Check | Passes when |
| --- | --- |
| `contract_version` | the pack declares the lab's contract version |
| `conditions` | conditions are non-empty, unique and described |
| `no_heavy_imports` | importing the pack and calling `probe()`, `policies()`, `policy_statuses()` and `conditions()` imports no simulator, `torch` or `numpy` (a fresh interpreter checks this) |
| `policy_pins` | every policy has a pin whose configuration hash matches and a `paper` string |
| `schemas_sample` | the recorded sample matches the declared schemas and Doctor finds no error in it |
| `fixture:<case>` (ten) | the case gives its declared Doctor, compare and gate result, and its comparison names only declared conditions |
| `reset_fingerprint` | same-seed trials give the same fingerprint; skipped, with the reason, when no policy can run |
| `offline_demo` | `gsp lab demo --task <name>` exits with `demo_exit` |

## Risks for the Robomimic release

The repository's README records issues the first Robomimic release will hit. Do not try to fix them as part of adding the pack; plan around them.

| Risk | Source |
| --- | --- |
| Lift policy and critic checkpoints are not published, so `lift` policies are registered as unavailable with that reason | README, "Setup and assets" |
| The Can demonstration replay may fail to reach success on modern MuJoCo | README, "Setup and assets" |
| The runners need the absolute-action PH data, not the ordinary relative-action files | README, "Setup and assets" |
| The lockfile pins `huggingface-hub==1.33.0`, which conflicts with the Robomimic runner's Diffusers import; the README documents an environment-only workaround | README, "Setup and assets" |
| The Robomimic runner needs its own environment (`.venv-robomimic`) and OSMesa for headless rendering, so the lab's CPU-only offline job cannot run it | README, "Setup and assets" |
| The report's top-down panels draw only 2-D position fields; a task without them still gets the signal plots (the toy-task test covers this), but a manipulation scene needs its own view | `lab/assets/report.js`, `tests/lab/test_toy_task_contract.py` |

The Robomimic methods `si`, `ccg` and `steg` become that pack's reference policies when it is added. Robomimic is not part of this release: the scaffolded rail pack is the proof that a simulator needs no change to the core.
