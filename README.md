# Guided Streaming Stochastic Interpolant Policy

This repository contains the reference research release for Guided Streaming
Stochastic Interpolant Policy, with the current benchmark workflow centered on
`Push-T`.

The codebase includes:

- an installable helper package in `guided_streaming_policy/`
- extracted reusable model and sequence components
- the original `Push-T` training and inference scripts used by the project

## Installation

### Recommended: Conda

```bash
conda env create -f environment.yml
conda activate guided-streaming-policy
```

### Alternative: pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[pusht]"
```

If you use a custom PyTorch build for your machine, install that build first and
then install the repository with `pip install -e .`.

## Quickstart

Check that your environment and local artifact paths are ready:

```bash
gsp check-env
```

List available `Push-T` tasks:

```bash
gsp list-tasks
```

Run a task from the repository root:

```bash
gsp run train-policy
gsp run train-dp
gsp run train-critic
gsp run infer-static-ccg
gsp run infer-dynamic-ccg
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

## Repository Layout

```text
guided_streaming_policy/core/   Reusable model and sequence utilities
guided_streaming_policy/pusht/  Push-T-specific package modules
Push-T/core/                    Compatibility import layer plus task code
Push-T/train/                   Training entrypoints
Push-T/inference/               Inference entrypoints
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

## License

Released under the MIT license. See [LICENSE](/home/users/oscar/Code-Repository-of-Guided-Streaming-Stochastic-Interpolant-Policy/LICENSE).
