# Robomimic Benchmark Guide

`Robomimic/` contains the benchmark-specific training and inference scripts
used in this repository for the robomimic low-dim tasks (`can`, `square`,
`lift`).

The recommended entrypoint is the repository root:

- [README.md](../README.md)

## Setup

From the repository root:

```bash
uv venv .venv-robomimic --python 3.10
UV_PROJECT_ENVIRONMENT=.venv-robomimic uv sync --extra robomimic
gsp check-env
```

This uses Robosuite 1.5.2, the current Robomimic development package, and
MuJoCo 3.4.0. The modern `mujoco` package bundles the runtime; no separate
MuJoCo archive or `MUJOCO_PY_MUJOCO_PATH` is required. WSL/Linux still needs
the rendering libraries:

```bash
sudo apt update
sudo apt install -y build-essential cmake patchelf \
  libosmesa6-dev libgl1-mesa-dev libglew-dev libglfw3 libglfw3-dev
```

Existing datasets and checkpoints must be replay-validated because modern
MuJoCo/controller behavior may differ from the legacy 2.1 stack.
For benchmark compatibility details, see
[Reproducibility Notes](../docs/reproducibility.md).

## Run Tasks From The Repository Root

The same scripts serve all three tasks; pick one with `--task` (default `can`),
which the CLI forwards to the scripts as the `TASK` environment variable.

```bash
gsp list-tasks
gsp run mimic-train-policy --task can
gsp run mimic-train-critic --task square
gsp run mimic-infer-si     --task lift
gsp run mimic-infer-ccg    --task can
gsp run mimic-infer-steg   --task square
```

## Direct Module Usage

If you want to run the original module commands directly (the task is selected
via the `TASK` environment variable):

```bash
cd Robomimic
TASK=can python -m train.train_policy
TASK=can python -m train.train_critic
TASK=can python -m inference.inference_si
TASK=can python -m inference.inference_ccg
TASK=can python -m inference.inference_steg
```

## Required Asset Layout

```text
Robomimic/datasets/
Robomimic/checkpoint/
Robomimic/core/
Robomimic/train/
Robomimic/inference/
```

Datasets are read from `Robomimic/datasets/<task>/ph/low_dim_abs.hdf5`,
checkpoints are read from `Robomimic/checkpoint/`, and generated GIFs are
written to per-task directories at the benchmark root
(`si_<task>_results/`, `ccg_<task>_results/`, `steg_<task>_results/`).

For a short rendered smoke run, set `TOTAL_NEEDED=1` for CCG and STEG:

```powershell
$env:TASK = "can"
$env:TOTAL_NEEDED = "1"
$env:RENDER_GIFS = "1"
python -m inference.inference_si
python -m inference.inference_ccg
python -m inference.inference_steg
```
