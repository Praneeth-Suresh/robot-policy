# Architecture

Policy Trace Lab is one subpackage, `guided_streaming_policy/lab/`. Every capability reads or writes the same versioned **episode trace**, so data checks, comparisons and replay always agree on what an episode is.

![Pipeline: simulator, episode trace, data checks, paired evaluation, replay](../brand/pipeline.svg)

The simulator and the native demonstration data stay untouched. Adapters convert them into traces. Doctor validates traces from either source. The evaluation chain is pair matcher, then checks, then gate. Replay reads the traces and the gate's regression list.

## Artifacts

| Artifact | Written by | Read by |
| --- | --- | --- |
| `manifest.json` and `steps.jsonl` per episode | demonstration reader, exporter | Doctor, pair matcher, gate, report |
| `run_manifest.json` and `resolved_config.json` per run | `gsp lab run` | pair matcher |
| `lab/comparisons/*.json` | you, before the candidate runs | `gsp lab run`, `gsp lab compare` |
| `lab/gates/*.json` | you, before results are inspected | `gsp lab gate` |
| `doctor.json` | `gsp lab doctor` | you, CI |
| `compare.json` | `gsp lab compare` | gate, report |
| `gate.json` and the exit code | `gsp lab gate` | CI, report |
| `report.html` | `gsp lab report` | you, reviewers |
| `lab/pins.json` | maintainers (deliberate edit) | `gsp lab policies`, `gsp lab run`, tests |

## Modules

| Module | Responsibility |
| --- | --- |
| `schema/` | The trace contract: constants, JSON Schemas (generated into `schemas/`), config loaders, canonical JSON, hashing, trace IO. One source of truth. |
| `adapters.py` | Shared helpers: config resolution, pin loading, policy table, trace reading, write-once file output. |
| `doctor.py` | The nine Dataset Doctor checks and the `doctor.json` report. |
| `exporter.py` | Runs one variant of a comparison through the task's run hook and writes traces. |
| `pairing.py` | Matches baseline and candidate episodes by (task, condition, seed) and writes `compare.json`. |
| `checks.py` | Native success (copied), failure categories, separation proxy, measurement coverage. |
| `gate.py` | Applies the checked-in gate config to `compare.json` and decides the exit code. |
| `report.py`, `assets/` | Self-contained `report.html` with the replay view. |
| `cli.py` | Argument parsing and wiring for `gsp lab`. |
| `tasks/base.py` | The task-adapter contract (a `Protocol` and frozen dataclasses). |
| `tasks/registry.py` | One line per task. |
| `tasks/pusht/` | The Push-T adapter: policies, pins, demonstration reader, run hook. |

## Dependency rules

Dependencies point one way. `tests/lab/test_architecture.py` parses the imports of every module with `ast` and fails on any other edge.

| Module | May import |
| --- | --- |
| `schema` | nothing else in `lab` |
| `adapters` | `schema`, `tasks.base` |
| `exporter` | `schema`, `adapters`, `tasks.base` |
| `doctor`, `pairing`, `checks` | `schema`, `tasks.base` |
| `gate` | `schema`, `pairing`, `checks`, `tasks.base` |
| `report` | `schema` |
| `cli` | everything |
| `tasks/base.py` | `schema` |
| `tasks/<task>` | `schema`, `adapters`, `tasks.base`, its own modules |

More rules, each with a test:

- The core never imports a task package. Only `cli` and `tasks/registry.py` refer to one (the registry by dotted-path string).
- The core contains no task name such as `pusht`. The only exemptions are the default `--task pusht` in `cli.py` and the registry line.
- Research code (`guided_streaming_policy/core`, `Push-T/`, `Robomimic/`) never imports `lab`. The lab imports research code only inside `tasks/pusht`.
- The default install has no new dependency. Everything outside `tasks/pusht` uses only the standard library; `numpy`, `zarr` and `torch` are imported lazily inside `tasks/pusht`.
- Every module declares `__all__`, has typed public functions, stays under 400 lines, and has no function over 50 lines or complexity over 10.

These amend the build prompt's table in two places: every core module may import `tasks.base` (the contract), and `exporter` and tasks may import `adapters`. The reasons are in [decisions](decisions.md).

## How a run is recorded

The Push-T runners are monolithic functions that run three episodes per call using a module-global `env`. The exporter swaps that global for a recording proxy, calls the unchanged runner, and keeps the third episode (the guided one). With the exporter off, the default Push-T run path is not touched. See [baseline](baseline.md) for the details and the source references.

## Adding a task

A task is one folder under `lab/tasks/` plus one registry line. See [adding a task](adding-a-task.md). A contract test (`tests/lab/test_toy_task_contract.py`) defines a toy task inside the test suite and runs the whole offline pipeline on it, so any Push-T assumption in the core fails the test.
