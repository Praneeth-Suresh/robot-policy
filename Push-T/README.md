# Push-T Project Run Guide

This project uses package imports such as `from core.dataset import ...`.
Therefore, run all scripts from the project root with `python -m`.

Do not run scripts by absolute file path, for example:

```bash
python /home/users/meiyi/Guided_SI_RSS/Push-T-new/inference/dynamic/inference_ccg_dynamic.py
```

That will make Python search from the script folder and may cause:

```text
ModuleNotFoundError: No module named 'core'
```

## Setup

```bash
cd /home/users/meiyi/Guided_SI_RSS/Push-T-new
conda activate streaming-flow-policy
```

## Training Commands

```bash
python -m train.train_dp
python -m train.train_policy
python -m train.train_critic
```

## Static Inference Commands

```bash
python -m inference.static.inference_dp_gconva
python -m inference.static.inference_fp_gconva
python -m inference.static.inference_ccg
python -m inference.static.inference_naive
python -m inference.static.inference_steg

```

## Dynamic Inference Commands

```bash
python -m inference.dynamic.inference_ccg_dynamic
python -m inference.dynamic.inference_dp_gconva_dynamic
python -m inference.dynamic.inference_fp_gconva_dynamic
python -m inference.dynamic.inference_naive_dynamic
python -m inference.dynamic.inference_steg_dynamic
```

## Project Paths

The scripts expect the following folders under the project root:

```text
data/
checkpoints/
outputs/
core/
train/
inference/
```

Data is loaded from `data/`, checkpoints are loaded from `checkpoints/`, and generated results are written to `outputs/`.
