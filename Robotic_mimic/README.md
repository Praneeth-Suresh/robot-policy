# Robotic_mimic Benchmark Guide

`Robotic_mimic/` contains the benchmark-specific training and inference scripts
used in this repository for the robomimic low-dim tasks (`can`, `square`,
`lift`).

The recommended entrypoint is the repository root:

- [README.md](../README.md)

## Setup

From the repository root:

```bash
conda env create -f environment.yml
conda activate guided-streaming-policy
gsp check-env
```

If you prefer a pure pip flow:

```bash
pip install -e ".[robomimic]"
```

`robosuite` cannot be resolved by pip on Python 3.10 (its metadata pins
`numba<=0.53.1` / `free-mujoco-py==2.1.6`), so install it without dependencies
after either path above. This matches the experiment environment, which runs
`numba==0.63.1` / `mujoco-py==2.1.2.14`:

```bash
pip install robosuite==1.3.2 --no-deps
```

`robosuite` / `mujoco-py` also need the MuJoCo 2.1.0 binaries at runtime:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
```

These are the two supported installation paths for the public repository.
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
cd Robotic_mimic
TASK=can python -m train.train_policy
TASK=can python -m train.train_critic
TASK=can python -m inference.inference_si
TASK=can python -m inference.inference_ccg
TASK=can python -m inference.inference_steg
```

## Required Asset Layout

```text
Robotic_mimic/datasets/
Robotic_mimic/checkpoint/
Robotic_mimic/core/
Robotic_mimic/train/
Robotic_mimic/inference/
```

Datasets are read from `Robotic_mimic/datasets/<task>/ph/low_dim_abs.hdf5`,
checkpoints are read from `Robotic_mimic/checkpoint/`, and generated results are
written to per-task directories at the benchmark root (`ccg_<task>_results/`,
`steg_<task>_results/`).
