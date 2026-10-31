# Architecture

## Bounded Contexts

| Context | Owns | Does Not Own | Public Entry Point |
| --- | --- | --- | --- |
| Reusable policy core | Sequence windows, normalization, interpolation, flow-policy building blocks, CAR guidance | Dataset-specific loading, benchmark rendering, experiment orchestration | `guided_streaming_policy/core/` |
| Experiment adapters | Dataset loading, model construction, training and inference procedures for one benchmark | General-purpose policy abstractions | `Push-T/`, `Robomimic/` |
| CLI and configuration | Discoverable commands, task validation, environment checks, command dispatch | Core model behavior and benchmark internals | `guided_streaming_policy/cli.py` |
| Verification | Unit tests, compile checks, and deterministic Beryl checks | Experiment results and external simulator behavior | `tests/`, `.beryl/scripts/check.sh` |

## Boundary Rules

1. Shared behavior belongs in `guided_streaming_policy/core/`; benchmark code
   may depend on it, but core code must not import benchmark scripts.
2. Benchmark-specific dataset, simulator, and rendering dependencies stay in
   their benchmark directories.
3. CLI commands are thin adapters over existing experiment entry points; do
   not duplicate training or inference logic in the dispatcher.
4. Datasets, checkpoints, simulator binaries, and generated results are
   external or ignored artifacts, never package source.
5. Preserve comparable seeds, checkpoints, normalization statistics, and
   evaluation settings when comparing methods.
6. Treat simulator physics, control frequency, action convention, observation
   timing, and data preprocessing as part of the experiment contract.
7. Keep deployment concerns such as timestamped observations, scheduled future
   actions, and inference latency at the policy/environment boundary; do not
   hide them inside reusable mathematical primitives.

## Public Interfaces

- `guided_streaming_policy/core/` exports reusable policy and guidance
  components.
- `guided_streaming_policy/cli.py` backs the installed `gsp` command.
- Each benchmark script remains responsible for its own data and simulator
  contract.

## Simulator And Control Contracts

Push-T runs physics at `sim_hz = 100` and policy control at `control_hz = 10`,
so each action is applied over 10 physics substeps. Actions are target
positions tracked by a PD controller with the current documented gains
(`k_p = 100`, `k_v = 20`). Changes to timestep, substeps, damping, controller
gains, or control frequency are experiment changes and must be recorded.

Robomimic uses robosuite 1.3.2 with MuJoCo 2.1 through `mujoco-py`. Its
environment setup owns headless rendering, control-mode resolution, seeded
state caching, action sanitization, and conversion of simulator failures into
terminated episodes. Delta versus absolute OSC action mode must be established
from metadata/statistics or explicit replay calibration, not guessed from a
policy output shape.

## Evaluation Contract

Demo replay is the first environment check: initialize from the recorded state,
replay recorded actions, and compare the resulting trajectory and success. A
large replay mismatch indicates an environment, state, action-convention, or
physics problem rather than a policy problem. Evaluation should then measure
per-episode outcomes, collisions, wall time, inference latency, and uncertainty
across seeds; generated videos and CSVs remain ignored artifacts.

## Design Concept

Keep research primitives task-agnostic and make benchmark integrations thin.
The CLI provides a stable, discoverable experiment surface while the legacy
benchmark scripts retain ownership of simulator-specific details.

## Decision Record

- 2026-10-31: Use `gsp` as the documented experiment surface, because it
  centralizes environment checks and keeps benchmark commands discoverable.
- 2026-10-31: Keep Push-T and robomimic artifacts outside version control,
  because datasets, checkpoints, and rendered results are large external
  inputs or generated outputs.
- 2026-10-31: Install Beryl with agent-core, tool-shims, and checks only,
  because hooks, CI, driver workflows, and frontend skills are not needed for
  this research workspace.

## Pressure Points

- robomimic depends on MuJoCo 2.1, `mujoco-py`, OSMesa, and Linux-specific
  runtime behavior; Windows validation is limited to source-level checks.
- Reproducibility depends on matching dataset versions, normalization stats,
  seeds, checkpoints, and guidance settings.
- Legacy benchmark scripts vary in maturity; new reusable code should be
  tested before being exposed through the CLI.
- Simulation-to-sim and simulation-to-real gaps are expected pressure points:
  recorded actions may not reproduce trajectories exactly, and real hardware
  adds observation age, camera/proprioception skew, and inference scheduling.
