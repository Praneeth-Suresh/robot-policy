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

## Setup and assets

Run commands from the repository root on Linux or WSL2, using Python 3.10 and
[uv](https://docs.astral.sh/uv/). CUDA is optional, but the models are large:
for faster trials use a CUDA-capable PyTorch device and check it with
`python -c "import torch; print(torch.cuda.is_available())"`. CPU trials can
take a long time. Do not use `uv run` without the appropriate extra: it can
resync `.venv` and remove the Push-T dependencies.

```bash
uv python install 3.10
uv venv .venv --python 3.10
UV_PROJECT_ENVIRONMENT=.venv uv sync --locked --extra pusht
uv venv .venv-robomimic --python 3.10
UV_PROJECT_ENVIRONMENT=.venv-robomimic uv sync --locked --extra robomimic
```

The current lockfile combines Transformers 4.41.2 with
`huggingface-hub==1.33.0`, which fails when the Robomimic runner imports
Diffusers. Until the lockfile is corrected, install a compatible Hub version
outside the project directory so the project override is not reapplied:

```bash
(cd /tmp && uv pip install --python "$OLDPWD/.venv-robomimic/bin/python" 'huggingface-hub==0.36.0')
```

Repeat that command after each Robomimic `uv sync`. This is an environment-only
workaround, not a change to the lockfile. Robomimic uses the bundled MuJoCo 3.4
runtime and Robosuite 1.5.2. For headless Linux rendering, install OSMesa and
related libraries, then set `MUJOCO_GL=osmesa` before starting Python:

```bash
sudo apt install -y build-essential cmake patchelf libosmesa6-dev \
  libgl1-mesa-dev libglew-dev libglfw3 libglfw3-dev
```

Datasets and checkpoints are not in Git. The published
[checkpoint folder](https://drive.google.com/drive/folders/18xWIlqNR2cQ2M6FI1H13l6uUxXMkDmC3)
contains all four Push-T checkpoints and Can/Square policy and critic
checkpoints. It does **not** contain Lift policy or critic checkpoints. Restore
the release files with:

```bash
uvx --from gdown gdown --folder https://drive.google.com/drive/folders/1-ND2B-58MdJntsqkfNdUri3ywll0vEAj -O Push-T/checkpoints
uvx --from gdown gdown --folder https://drive.google.com/drive/folders/1gJU8RU2Qo670QgcUOVzrrm18GAEvo4Z2 -O Robomimic/checkpoint
```

Download the
[Push-T dataset](https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip)
and repackage its nested Zarr directory into the ZipStore layout expected by
the code:

```bash
mkdir -p Push-T/data
curl -fL https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip -o /tmp/robot-policy-pusht.zip
.venv/bin/python - <<'PYDATA'
from zipfile import ZipFile, ZIP_STORED
prefix = "pusht/pusht_cchi_v7_replay.zarr/"
with ZipFile("/tmp/robot-policy-pusht.zip") as source, ZipFile(
    "Push-T/data/pusht_cchi_v7_replay.zarr.zip", "w", compression=ZIP_STORED
) as target:
    for item in source.infolist():
        if item.filename.startswith(prefix) and not item.is_dir():
            target.writestr(item.filename[len(prefix):], source.read(item))
PYDATA
```

The runners require *absolute-action* PH data, not the ordinary relative-action
files in the original Robomimic listing. The following public mirror supplies
those files:

```bash
for task in can square lift; do
  mkdir -p "Robomimic/datasets/$task/ph"
  curl -fL "https://huggingface.co/datasets/ChaoyiPan/mip-dataset/resolve/main/robomimic/$task/ph/low_dim_abs.hdf5" \
    -o "Robomimic/datasets/$task/ph/low_dim_abs.hdf5"
done
```

Check with `.venv/bin/gsp check-env` or `.venv-robomimic/bin/gsp check-env`.
The report checks both benchmarks, so missing packages from the *other*
environment are expected. Its `Robust_critic_can.pth` check does not match the
released `Robust_critic_can_latest_checkpoint.pth` used by CCG. Lift inference
requires a Lift policy checkpoint; Lift CCG additionally requires a Lift critic.
Existing datasets and checkpoints need replay validation on modern MuJoCo:
the Can demonstration replay may fail to achieve success and the mirror's
environment version differs from the installed Robosuite version.

## Run trials

Run each benchmark from its own environment. The `gsp` commands below use the
scripts' defaults; `gsp run` does not accept a general trial-count or seed
option. Methods should be compared with the same dataset, seed and obstacle
condition. Results, checkpoints, and datasets remain outside Git.

### Push-T trials

From the repository root, set the variables and run the saved script:

```bash
METHOD=naive CONDITION=static COUNT=1 SEED=0 .venv/bin/python Push-T/run_trial.py
```

| Variable | Values | Default | Meaning |
| --- | --- | --- | --- |
| `METHOD` | `naive`, `ccg`, `steg`, `dp`, `flow` | `naive` | Policy and guidance method |
| `CONDITION` | `static`, `dynamic` | `static` | Obstacle condition |
| `COUNT` | Positive integer | `1` | Number of consecutive seeds |
| `SEED` | Nonnegative integer | `0` | First seed |
| `SDL_VIDEODRIVER` | SDL driver name, e.g. `dummy` | `dummy` | Headless rendering driver |

The static condition uses obstacle position `[-0.3, -0.5]`; each seed runs
baseline and guided rollouts. Dynamic mode 1 moves an obstacle sinusoidally;
for the previous example use `CONDITION=dynamic SEED=500`. The script prints
metrics and saves GIFs under `Push-T/outputs/one-trial/<condition>/<method>/`.
Existing files with the same names may be replaced. Guidance scales match the
previous README example; compare methods with matching seeds and conditions.

### Robomimic trials

From the repository root, run for example:

```bash
METHOD=si TASK=can COUNT=1 SEED=0 .venv-robomimic/bin/python Robomimic/run_trial.py
METHOD=steg TASK=can COUNT=1 RENDER_GIFS=1 .venv-robomimic/bin/python Robomimic/run_trial.py
```

| Variable | Values | Default | Meaning |
| --- | --- | --- | --- |
| `METHOD` | `si`, `ccg`, `steg` | `si` | Inference method |
| `TASK` | `can`, `square`, `lift` | `can` | Manipulation task |
| `COUNT` | Positive integer | `1` | SI rollouts or valid CCG/STEG trials |
| `SEED` | Nonnegative integer | `0` | First SI seed; CCG/STEG start at seed 1 |
| `RENDER_GIFS` | `0`, `1` | `1` | STEG GIF rendering; CCG and SI save GIFs independently |
| `MUJOCO_GL` | MuJoCo rendering backend, e.g. `osmesa` | `osmesa` | Rendering backend |

Robomimic inference saves normalization statistics in a `.gsp-cache/` directory
beside the selected dataset. The first run computes them; later runs reuse them
without building the training dataset. Dataset file changes or a preprocessing
version change trigger a rebuild. To force a rebuild, remove that task's
`.gsp-cache/` directory. Simulator calibration and model loading still occur
on each invocation. Training continues to load the full dataset.

SI runs exactly `COUNT` consecutive seeds and writes to
`Robomimic/si_<task>_results/`. CCG and STEG may try additional seeds until
`COUNT` baseline-valid trials are found; they write to
`Robomimic/ccg_<task>_results/` and `Robomimic/steg_<task>_results/`.
Existing files with the same names may be replaced. Lift requires its missing
policy checkpoint and, for CCG, a critic checkpoint. Existing datasets and
checkpoints require replay validation on modern MuJoCo as noted above.

Training entry points remain available via `gsp run`; preserve release
checkpoints before training because training may overwrite or resume them.

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
