# Adding a task

A new task is one folder, `guided_streaming_policy/lab/tasks/<task>/`, plus its fixtures and tests, one registry line, one README row and one docs page. The core stays closed: a change that needs more than that is a design question to raise, not something to work around. `tests/lab/test_toy_task_contract.py` shows the contract in about 300 lines: it defines a toy task inside the test suite and runs Doctor, run, compare, gate, report and the demo on it.

Work through the steps in order. Each step has an exit check, and the next step does not start until it passes.

## The contract

`tasks/base.py` defines a `TaskAdapter` `Protocol`. Your task class supplies:

| Member | Used by |
| --- | --- |
| `name` | the registry and every `--task` flag |
| `read_demonstrations(path)` | `gsp lab doctor`: native data to episode traces, with the real split provenance if the source has one |
| `run_trial(request)` and `conditions()` | `gsp lab run`: run one variant on one matrix cell with step recording; raise `TrialCrashed` when the trial fails |
| reset fingerprint (a field of the `TrialRecord` you return) | pair matcher: a hash of the actual starting state, or `None` |
| native success and termination (fields of the `TrialRecord`) | checks, Doctor: copied from the simulator and never redefined |
| `observation_schema()`, `action_schema()`, `timebase()` | Doctor, schema validation |
| geometry in the `TrialRecord` and the gate rule's `signal` | checks: the geometry and timestamps a constraint needs |
| `probe()` | `gsp lab run`: simulator version and which assets are present |
| `fixtures()` | `gsp lab demo` and CI: a synthetic fixture set |
| `policies()`, `policy_statuses()`, `pins_path()`, `current_pins()` | `gsp lab policies`, comparison configs, pin checks: every reference method registered by name and pinned by checkpoint hash and resolved config hash |

Register the task with one line in `tasks/registry.py`. All commands default to `--task pusht`, so existing usage does not change.

## Steps

1. **Inspect.** Record the real observation and action spaces, the native success metric, the termination rules, the available telemetry and the required assets, each with a source reference (file and line). Use [baseline](baseline.md) as the model. *Exit check:* every field a later step uses has a source reference.
2. **Declare the schema.** Write the task's observation and action schema with units, frame and the action convention (absolute or delta). *Exit check:* the schema validates a real sample.
3. **Read demonstrations.** Write the demonstration reader and its Doctor fixtures: clean, corrupt, and a leaked copy. *Exit check:* the clean fixture passes, and each corrupt case produces its expected finding at the right episode and step.
4. **Export rollouts.** Write the run hook and the reset fingerprint. Record the actual reset state, not only the seed. *Exit check:* two runs with the same seed give the same fingerprint.
5. **Define checks.** Choose the native success metric and one evidenced process check. Register and pin the task's reference policies. Write the task's gate config before looking at any results. *Exit check:* a missing signal gives `unknown`, not a guess.
6. **Add fixtures and the offline demo.** Add synthetic fixtures for the task that cover the cases in `tests/lab/test_fixtures.py` (clean, no split, corrupt, leakage, seeded regression, two valid trajectories, missing signal, mismatched reset, truncated, missing pairs). *Exit check:* `gsp lab demo --task <task>` exits with the documented code.
7. **Run it for real.** Run one of the task's reference policies against a variant with the real simulator, capture screenshots, and add them to the README. If it cannot run, stop and ask. *Exit check:* the result is on the task's page, labelled "verified in simulation" or "fixture only".
8. **Wire up and document.** Register the task, add its row to the README task table, extend the offline CI job. *Exit check:* the architecture and hygiene tests and the README command tests pass with no change to the core.

## Risks for the Robomimic release

The repository's README records issues the first Robomimic release will hit. Do not try to fix them as part of adding the task; plan around them.

| Risk | Source |
| --- | --- |
| Lift policy and critic checkpoints are not published, so `lift` policies are registered as unavailable with that reason | README, "Setup and assets" |
| The Can demonstration replay may fail to reach success on modern MuJoCo | README, "Setup and assets" |
| The runners need the absolute-action PH data, not the ordinary relative-action files | README, "Setup and assets" |
| The lockfile pins `huggingface-hub==1.33.0`, which conflicts with the Robomimic runner's Diffusers import; the README documents an environment-only workaround | README, "Setup and assets" |
| The Robomimic runner needs its own environment (`.venv-robomimic`) and OSMesa for headless rendering, so the lab's CPU-only offline job cannot run it | README, "Setup and assets" |
| The report's top-down panels draw only 2-D position fields; a task without them still gets the signal plots (the toy-task test covers this), but a manipulation scene needs its own view | `lab/assets/report.js`, `tests/lab/test_toy_task_contract.py` |

The Robomimic methods `si`, `ccg` and `steg` become that task's reference policies when it is added.
