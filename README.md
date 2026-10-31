# Guided Streaming Stochastic Interpolant Policy

Reference implementation for the paper's SSIP, STEG, and CCG experiments:
[arXiv:2605.10051](https://arxiv.org/abs/2605.10051).

## Setup

The released environment uses Python 3.10. From the repository root:

```powershell
conda env create -f environment.yml
conda activate guided-streaming-policy
gsp check-env
```

If the environment already exists:

```powershell
conda install -n guided-streaming-policy -y cmake
conda run -n guided-streaming-policy python -m pip install -e ".[pusht,robomimic]"
conda run -n guided-streaming-policy python -m pip install robosuite==1.3.2 --no-deps
```

The Robomimic stack also expects `mujoco-py==2.1.2.14`, MuJoCo 2.1.0, and
Linux OSMesa/GL libraries. It is not runnable on this Windows host; use Linux
for Robomimic rendering.

## Assets

Download the released checkpoints from:

<https://drive.google.com/drive/folders/18xWIlqNR2cQ2M6FI1H13l6uUxXMkDmC3>

Place them in:

```text
Push-T/checkpoints/
Robomimic/checkpoint/
```

Datasets must be placed at:

```text
Push-T/data/pusht_cchi_v7_replay.zarr.zip
Robomimic/datasets/can/ph/low_dim_abs.hdf5
Robomimic/datasets/square/ph/low_dim_abs.hdf5
Robomimic/datasets/lift/ph/low_dim_abs.hdf5
```

The public upstream datasets used here are Push-T from
<https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip> and
Robomimic PH low-dimensional datasets from
<https://huggingface.co/datasets/amandlek/robomimic/tree/main/v1.5>.

Check assets and available commands:

```powershell
gsp check-env
gsp list-tasks
```

## Run experiments

All commands run from the repository root.

### Push-T

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

The inference scripts save rendered GIF trajectories under `Push-T/outputs/`.
The static CCG script uses its built-in 400-trial batch.

### Robomimic

Select `can`, `square`, or `lift`:

```powershell
gsp run mimic-train-policy --task can
gsp run mimic-train-critic --task can
gsp run mimic-infer-si --task can
gsp run mimic-infer-ccg --task can
gsp run mimic-infer-steg --task can
```

For rendered GIFs in PowerShell:

```powershell
$env:RENDER_GIFS = "1"
$env:TOTAL_NEEDED = "100"
gsp run mimic-infer-steg --task square
```

Robomimic writes results to `ccg_<task>_results/` and `steg_<task>_results/`.

## Development checks

```powershell
python -m unittest discover -v
python -m compileall -q guided_streaming_policy Push-T tests
```

Datasets, checkpoints, generated GIFs, and result directories are intentionally
gitignored. Only source code, configuration, tests, and documentation belong in
the GitHub repository.
