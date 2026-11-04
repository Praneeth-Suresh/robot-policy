# Robot Policy

An open research workspace for general robot-policy experiments with
diffusion, flow-matching, and stochastic-interpolant policies. The repository
currently contains reproducible Push-T and robomimic baselines, plus reusable
guidance components for testing new robotics ideas.

The original Guided Streaming Stochastic Interpolant Policy (SSIP/GSSIP)
experiments remain available through the `gsp` CLI. New experiments should
reuse the task-agnostic code under `guided_streaming_policy/core/` rather than
copying benchmark scripts.

## What is included

Envrionments for the policy to operate on:


| Area      | Purpose                                                                                 |
| ----------- | ----------------------------------------------------------------------------------------- |
| Push-T    | 2-D manipulation benchmark for diffusion, flow, streaming-SI, STEG, and CCG comparisons |
| Robomimic | Low-dimensional `can`, `square`, and `lift` manipulation tasks                          |

Then, there are the algorithms that operate on these policies:


| Area             | Purpose                                                                     |
| ------------------ | ----------------------------------------------------------------------------- |
| Flow-policy core | Sequence utilities, interpolation, normalization, and model building blocks |
| CAR guidance     | Conflict-Aware Additive Guidance for compositional reward gradients         |

`ConflictAwareAdditiveGuidance` follows the public CAR-guidance implementation:
[https://github.com/yuxuehui/CAR-guidance](https://github.com/yuxuehui/CAR-guidance). It computes the all-pairs
`(1 - cosine) / 2` conflict score, applies the smootherstep conflict gate, and
adds a learned/vector correction only where reward channels disagree. The full
upstream Maze2D and ManiSkill2 experiment runners remain available in the
upstream repository; this project provides the reusable adapter for applying
the method to the policies here or to a new robot task.

### CAR smoke-test result

The local CAR component smoke test validates aligned and opposed reward
gradients: aligned rewards receive no correction, while opposed rewards receive
full correction after crossing the default conflict gate. Generated plots and
trajectory images are local-only result artifacts and are intentionally not
published.

## Setup

UV is the supported environment manager. Use WSL2/Linux for Robomimic,
MuJoCo 2.1, and GPU-based simulation. Keep the checkout in the WSL
filesystem rather than under `/mnt/c`. The released environment uses Python
3.10. Install UV first if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Push-T and shared development environment

```bash
uv python install 3.10
uv venv --python 3.10
source .venv/bin/activate
uv sync --extra pusht
gsp check-env
```

For an existing checkout, `uv sync --extra pusht` creates or updates the
project environment from `pyproject.toml` and `uv.lock`.

### Robomimic environment

Robomimic is kept in a separate environment because its legacy simulator
dependencies are not compatible with every modern robotics stack:

```bash
uv venv .venv-robomimic --python 3.10
source .venv-robomimic/bin/activate
UV_PROJECT_ENVIRONMENT=.venv-robomimic uv sync --extra robomimic
uv pip install --no-deps robosuite==1.3.2
gsp check-env
```

Robomimic additionally requires `mujoco-py==2.1.2.14`, MuJoCo 2.1.0, and
Linux OSMesa/GL libraries. Install the system prerequisites before building
`mujoco-py`:

```bash
sudo apt update
sudo apt install -y build-essential cmake patchelf \
  libosmesa6-dev libgl1-mesa-dev libglew-dev libglfw3 libglfw3-dev
```

Place MuJoCo 2.1.0 under `~/.mujoco/mujoco210/` and add its libraries:

```bash
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc
source ~/.bashrc
```

Robomimic rendering is not supported on the Windows host; use WSL2/Linux.

## Assets and available tasks

Download the released checkpoints from:
[https://drive.google.com/drive/folders/18xWIlqNR2cQ2M6FI1H13l6uUxXMkDmC3](https://drive.google.com/drive/folders/18xWIlqNR2cQ2M6FI1H13l6uUxXMkDmC3)

Place Push-T assets under `Push-T/data/` and `Push-T/checkpoints/`, and
robomimic assets under `Robomimic/datasets/` and `Robomimic/checkpoint/`.
The public datasets are Push-T from
[https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip](https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip) and
robomimic PH low-dimensional data from
[https://huggingface.co/datasets/amandlek/robomimic/tree/main/v1.5](https://huggingface.co/datasets/amandlek/robomimic/tree/main/v1.5).

```bash
gsp check-env
gsp list-tasks
```

## Run the existing robot-policy baselines

All commands run from the repository root.

### Push-T

```bash
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

Rendered trajectories are written to `Push-T/outputs/`.

### Robomimic

Choose `can`, `square`, or `lift`:

```bash
gsp run mimic-train-policy --task can
gsp run mimic-train-critic --task can
gsp run mimic-infer-si --task can
gsp run mimic-infer-ccg --task can
gsp run mimic-infer-steg --task can
```

## Add a new robot flow policy

For a new robot task, keep the environment and dataset code task-specific, but
make the policy expose a velocity function with the same conceptual contract:

```python
velocity = flow_model.velocity_field(state, time, conditions)
```

Then define one base guidance function, one learned correction function, and
one function that returns per-reward gradients:

```python
from guided_streaming_policy.core.car_guidance import ConflictAwareAdditiveGuidance

car = ConflictAwareAdditiveGuidance(
    base_guidance=my_compositional_guidance,
    correction=my_online_residual,
    reward_gradients=my_reward_gradients,
    threshold=0.15,
    temperature=0.1,
)

guided_velocity = car(
    x=trajectory,
    t=time,
    base_velocity=velocity,
    conditions=conditions,
)
```

`my_reward_gradients` should return `(batch, rewards, ...)`; for a trajectory,
use `(batch, rewards, horizon, action_dim)`. The correction must have the same
shape as `base_velocity`. Train the correction online or offline against the
task reward, and compare unguided flow, additive guidance, CAR, and CCG under
the same seeds and checkpoints.

## Research ideas to compare

- **SSIP/GSSIP:** streaming stochastic-interpolant policy updates for
  receding-horizon action generation.
- **STEG:** streaming guidance baseline already exposed by the CLI.
- **CCG:** collision/constraint critic guidance already implemented for the
  included benchmarks.
- **CAR:** conflict-aware additive correction for compositional flow guidance;
  start with the default `threshold=0.15` and `temperature=0.1`, then ablate
  the gate, reward-channel count, and correction capacity.
- **New ideas:** add a task adapter, register its command in `cli.py`, and
  keep data/checkpoints/results outside version control.

For full CAR Maze2D planning and ManiSkill2 point-cloud manipulation scripts,
follow the upstream experiment READMEs:
[https://github.com/yuxuehui/CAR-guidance](https://github.com/yuxuehui/CAR-guidance).

## Development checks

```bash
uv run python -m unittest discover -v
uv run python -m compileall -q guided_streaming_policy Push-T tests
```

Datasets, checkpoints, rendered trajectories, and result directories are
intentionally gitignored.
