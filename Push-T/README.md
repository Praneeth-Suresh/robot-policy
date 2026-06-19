# Push-T Benchmark Guide

`Push-T/` contains the benchmark-specific training and inference scripts used in
this repository.

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
pip install -e ".[pusht]"
```

These are the two supported installation paths for the public repository.
For benchmark compatibility details, see
[Reproducibility Notes](../docs/reproducibility.md).

## Run Tasks From The Repository Root

```bash
gsp list-tasks
gsp run train-policy
gsp run train-dp
gsp run train-critic
gsp run infer-static-ccg
gsp run infer-dynamic-ccg
```

## Direct Module Usage

If you want to run the original module commands directly:

```bash
cd Push-T
python -m train.train_dp
python -m train.train_policy
python -m train.train_critic
python -m inference.static.inference_dp_gconva
python -m inference.static.inference_fp_gconva
python -m inference.static.inference_ccg
python -m inference.static.inference_naive
python -m inference.static.inference_steg
python -m inference.dynamic.inference_ccg_dynamic
python -m inference.dynamic.inference_dp_gconva_dynamic
python -m inference.dynamic.inference_fp_gconva_dynamic
python -m inference.dynamic.inference_naive_dynamic
python -m inference.dynamic.inference_steg_dynamic
```

## Required Asset Layout

```text
Push-T/data/
Push-T/checkpoints/
Push-T/outputs/
Push-T/core/
Push-T/train/
Push-T/inference/
```

The dataset is read from `Push-T/data/`, checkpoints are read from
`Push-T/checkpoints/`, and generated results are written to `Push-T/outputs/`.
