# Guided Streaming Stochastic Interpolant Policy

This repository contains the reference research release for Guided Streaming
Stochastic Interpolant Policy, with benchmark workflows for `Push-T` and `Robotic_mimic`.

The codebase includes:

- an installable helper package in `guided_streaming_policy/`
- extracted reusable model and sequence components
- the original `Push-T` and `Robotic_mimic` training and inference scripts used
by the project

This is currently a source-checkout release: clone the repository, install it
editable, and run the benchmark tasks from that checkout. The benchmark scripts
are kept numerically aligned with the reported results; see
[Reproducibility Notes](docs/reproducibility.md).

## Installation

Two installation paths are supported:

- Conda environment
- pip extras

### Recommended: Conda

```bash
conda env create -f environment.yml
conda activate guided-streaming-policy
```

### Alternative: pip extras

Requires Python >= 3.10. Your default `python` may be older (the editable
install refuses < 3.10), so create the venv with an explicit 3.10 interpreter
or just use the Conda path above:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Then install **one** of the following:

```bash
pip install -e ".[pusht]"            # Push-T only
pip install -e ".[robomimic]"        # Robotic_mimic only
pip install -e ".[pusht,robomimic]"  # both, in one unified environment
```

If you use a custom PyTorch build for your machine, install that build first and
then install the repository with `pip install -e .`.

### Robotic_mimic extra setup

`robosuite==1.3.2` is required by `Robotic_mimic` but cannot be resolved by pip on
Python 3.10 (its metadata pins `numba<=0.53.1` / `free-mujoco-py==2.1.6`). After
either install path above, add it without dependencies. This matches the original
experiment environment, which runs `numba==0.63.1` / `mujoco-py==2.1.2.14`:

```bash
pip install robosuite==1.3.2 --no-deps
```

`robosuite` / `mujoco-py` also need the MuJoCo 2.1.0 binaries at runtime:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
```

## Quickstart

Check that your environment and local artifact paths are ready:

```bash
gsp check-env
```

List all available tasks (Push-T and Robotic_mimic):

```bash
gsp list-tasks
```

Run a task from the repository root:

```bash
gsp run train-policy
gsp run train-dp
gsp run train-critic
gsp run infer-static-ccg
```

The helper CLI executes the existing `Push-T` modules in the correct working
directory, so you do not need to `cd Push-T` before every command.

## Expected Assets

The `Push-T` scripts expect these locations:

```text
Push-T/data/pusht_cchi_v7_replay.zarr.zip
Push-T/checkpoints/
Push-T/outputs/
```

Use `gsp check-env` at any time to verify whether the dataset, checkpoints, and
Python dependencies are all present.

The dataset and checkpoints are not committed to git. Before running benchmark
tasks, place the released artifacts at the paths above. If a checkpoint name
looks old or awkward, keep it unchanged unless you also update the matching
script and reproduction notes.

## Robotic_mimic

`Robotic_mimic` covers the robomimic low-dim tasks (`can`, `square`, `lift`).
The same scripts serve all three tasks; pick one with `--task` (default
`can`), which the CLI forwards to the scripts as the `TASK` environment
variable.

```bash
gsp run mimic-train-policy --task can
gsp run mimic-train-critic --task square
gsp run mimic-infer-si     --task lift
gsp run mimic-infer-ccg    --task can
gsp run mimic-infer-steg   --task square
```

As with Push-T, the CLI runs the original `Robotic_mimic` modules in the correct
working directory, so you do not need to `cd Robotic_mimic` first. Extra
environment variables the original scripts read are passed through, e.g.:

```bash
RENDER_GIFS=1 TOTAL_NEEDED=100 gsp run mimic-infer-steg --task square
```

The `Robotic_mimic` scripts expect these locations:

```text
Robotic_mimic/datasets/<task>/ph/low_dim_abs.hdf5
Robotic_mimic/checkpoint/
```

Use `gsp check-env` to verify the datasets, checkpoints, and Python
dependencies for both benchmarks.

## Pretrained Checkpoints

Datasets and checkpoints are not committed to git. Released checkpoints for
both `Push-T` and `Robotic_mimic` are available on Google Drive:

https://drive.google.com/drive/folders/18xWIlqNR2cQ2M6FI1H13l6uUxXMkDmC3?usp=drive_link

Download them and place each benchmark's files at the paths shown above
(`Push-T/checkpoints/` and `Robotic_mimic/checkpoint/`).

## Repository Layout

```text
guided_streaming_policy/core/   Reusable model and sequence utilities
guided_streaming_policy/pusht/  Push-T-specific package modules
Push-T/core/                    Compatibility import layer plus task code
Push-T/train/                   Training entrypoints
Push-T/inference/               Inference entrypoints
Robotic_mimic/core/             robomimic task code plus per-task config
Robotic_mimic/train/            Training entrypoints
Robotic_mimic/inference/        Inference entrypoints
tests/                          Smoke tests for repo-level tooling
```

## Direct Module Usage

If you prefer to invoke the original module paths directly:

```bash
cd Push-T
python -m train.train_policy
python -m train.train_dp
python -m train.train_critic
```

For `Robotic_mimic` (task selected via the `TASK` env var):

```bash
cd Robotic_mimic
TASK=can python -m train.train_policy
TASK=square python -m inference.inference_ccg
```

## Development Checks

```bash
python -m unittest discover -v
python -m compileall -q guided_streaming_policy Push-T tests
```

## License

Released under the MIT license. See [LICENSE](LICENSE).