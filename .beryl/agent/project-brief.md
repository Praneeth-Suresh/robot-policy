# Project Brief

## Research Goal

Develop and compare reproducible robot-policy experiments, especially
diffusion, flow-matching, stochastic-interpolant, streaming, and guided
policies, across the Push-T and robomimic benchmarks.

The primary users are researchers working from a Python 3.10 environment. A
successful change makes an experiment easier to run, compare, or reuse without
silently changing data handling, model interfaces, or evaluation conditions.

## Repository Map

| Area | Responsibility | Entry point |
| --- | --- | --- |
| `guided_streaming_policy/core/` | Reusable sequence, normalization, interpolation, policy, and guidance components | Python imports |
| `guided_streaming_policy/cli.py` | Stable experiment command registry and environment checks | `gsp` |
| `Push-T/` | Push-T training and inference baselines | `gsp run ...` |
| `Robomimic/` | Low-dimensional robomimic training and inference baselines | `gsp run mimic-...` |
| `tests/` | Focused unit and smoke tests for reusable behavior | `python -m unittest discover -v` |
| `docs/`, `README.md`, `curriculum.md` | Setup, experiment, and learning documentation | Markdown |

## Primary Workflows

1. **Prepare an environment**: create `robot-policy` from `environment.yml`,
   install `robosuite==1.3.2 --no-deps` where needed, then run `gsp check-env`.
2. **Run Push-T experiments**: use `gsp run train-policy`,
   `train-dp`, `train-critic`, and the static/dynamic inference commands.
   Rendered trajectories go to `Push-T/outputs/`.
3. **Run robomimic experiments**: choose `can`, `square`, or `lift` and use
   the `mimic-train-*` and `mimic-infer-*` commands. These runs require Linux
   with MuJoCo 2.1 and OSMesa; they are not supported on this Windows host.
4. **Add reusable research code**: implement task-agnostic behavior in
   `guided_streaming_policy/core/`, expose a deliberate public API, register
   new CLI commands only when an experiment is runnable, and add focused tests.
5. **Compare experiments**: keep task, dataset, checkpoint, seed, and guidance
   settings explicit; compare unguided flow, additive guidance, CAR, CCG, and
   other baselines under matched conditions.

## Run Commands

From the repository root:

```powershell
conda env create -f environment.yml
conda activate robot-policy
gsp check-env
gsp list-tasks
python -m unittest discover -v
python -m compileall -q guided_streaming_policy Push-T tests
```

Push-T:

```powershell
gsp run train-policy
gsp run train-dp
gsp run train-critic
gsp run infer-static-naive
gsp run infer-static-steg
gsp run infer-static-ccg
gsp run infer-dynamic-naive
gsp run infer-dynamic-steg
gsp run infer-dynamic-ccg
```

Robomimic:

```powershell
gsp run mimic-train-policy --task can
gsp run mimic-train-critic --task can
gsp run mimic-infer-si --task can
gsp run mimic-infer-ccg --task can
gsp run mimic-infer-steg --task can
```

## Data And Artifacts

Datasets and checkpoints are external inputs. Push-T assets belong in
`Push-T/data/` and `Push-T/checkpoints/`; robomimic assets belong in
`Robomimic/datasets/` and `Robomimic/checkpoint/`. Generated trajectories and
results are local-only under the ignored output directories. Do not commit
datasets, checkpoints, rendered trajectories, or result dumps.

## Domain Model

The repository has four practical layers:

1. **Simulation**: Push-T uses Pymunk physics and Pygame rendering behind a
   Gym 0.26 API. Robomimic uses robosuite 1.3.2, MuJoCo 2.1 through
   `mujoco-py`, and headless OSMesa rendering.
2. **Data**: Push-T demonstrations use a Zarr replay buffer. Robomimic
   low-dimensional demonstrations use HDF5 with absolute end-effector
   actions, saved states, observations, and episode boundaries.
3. **Training and guidance**: plain PyTorch loops use diffusers schedulers,
   EMA, learning-rate schedules, and a 1-D conditional U-Net. Streaming flow
   policies, stochastic interpolants, diffusion policies, CCG/STEG, and CAR
   are compared as distinct policy or guidance methods.
4. **Experiment surface**: `gsp` wraps legacy benchmark scripts so setup,
   task discovery, training, inference, and environment diagnostics remain
   discoverable.

Use WSL2/Linux for GPU training and all robomimic work. Keep repositories in
the WSL filesystem rather than `/mnt/c` for data-loading performance. A
Windows checkout is suitable for source inspection, unit tests that have their
dependencies installed, and compilation checks, but not for robomimic
rendering.

## Experiment Record

Every reported experiment must record the benchmark and task, dataset and
checkpoint identity, seed, normalization statistics, simulator and control
settings, method/guidance configuration, command, output directory, episode
count, and evaluation metrics. Comparisons must use matched settings and
multiple seeds where feasible; report confidence intervals and inference
latency rather than only a mean success rate.

Before diagnosing a policy, replay recorded demonstrations in the environment.
If replay fails, investigate state initialization, action convention, control
mode, simulator parameters, and observation preprocessing before changing the
model.

## Theory-To-Code Map

| Concept | Repository location |
| --- | --- |
| Diffusion Policy | `Push-T/train/train_dp.py`, `Push-T/inference/` |
| Streaming flow policy | `guided_streaming_policy/core/flow_policy.py`, `Push-T/train/train_policy.py` |
| Stochastic interpolant | `guided_streaming_policy/core/` interpolation helpers |
| Conditional 1-D U-Net | `guided_streaming_policy/core/models.py` |
| CCG/STEG critic guidance | `Push-T/core/critic.py`, `Push-T/inference/` |
| CAR guidance | `guided_streaming_policy/core/car_guidance.py` |

The wider ecosystem (MuJoCo/MJX, Isaac Lab, ManiSkill, ROS 2, MoveIt 2,
LeRobot, LeRobotDataset, RLDS, Rerun, and Foxglove) is reference context for
future simulator and hardware integrations, not a requirement for current
benchmark commands.

## Non-Goals

- Building a web frontend or user-facing application.
- Making robomimic rendering work on Windows.
- Replacing benchmark-specific legacy scripts when a thin CLI adapter is
  sufficient.
- Treating generated experiment artifacts as source files.

## Definition Of Done

A change is complete when its experiment or library behavior is covered by
focused tests where practical, documentation names the exact command and
required assets, `python -m unittest discover -v` and
`python -m compileall -q guided_streaming_policy Push-T tests` pass when
applicable, and `./.beryl/scripts/check.sh` passes. Any unavailable
platform-specific check must be reported explicitly.
