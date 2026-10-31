# Engineering Curriculum: Robot Policies in Simulation and on Hardware

This curriculum is for someone who already knows diffusion and flow-matching
theory and the policy papers, but has not yet built, debugged, or evaluated
robot-learning systems. It is mostly hands-on: how simulators actually work,
how robot datasets are stored, how training and evaluation pipelines are
engineered, and which tools the field uses today in simulation and on real
robots.

Everything is tied to this repository:

| Layer | What this repo uses |
| --- | --- |
| Simulators | Push-T: Pymunk physics + Pygame rendering behind a `gym==0.26.2` API. Robomimic: robosuite 1.3.2 on MuJoCo 2.1 through `mujoco-py`, rendered offscreen with OSMesa. |
| Data | Push-T: Zarr replay buffer (`pusht_cchi_v7_replay.zarr.zip`). Robomimic: HDF5 (`low_dim_abs.hdf5`, absolute end-effector actions). |
| Training | Plain PyTorch loops, `diffusers` (`DDPMScheduler`, `EMAModel`, LR schedulers), 1-D conditional U-Net. |
| Entry points | `gsp` CLI (`guided_streaming_policy/cli.py`) wrapping the legacy scripts in `Push-T/` and `Robomimic/`. |

## How to use this file

- Each module has a **goal**, a short list of **core resources** (do these),
  **lookup references** (open when you need them), a **build task**, and a
  **done when** checklist. The build tasks are the curriculum; the resources
  support them.
- Keep a lab notebook (a markdown file or a notebook) of every command, error,
  and fix. Install and rendering errors repeat, and this log will save you days.
- Work in WSL2 Ubuntu from Module 0 onward, even for Push-T. Robomimic, ROS 2,
  Isaac Lab, and almost all real-robot stacks assume Linux.
- Estimated pace: about 6 weeks at 10 to 15 hours per week. Parts A and B
  (Modules 0 to 6) are the core. Part C (Modules 7 and 8) is the field survey
  for simulation and deployment tooling; you can read it in parallel.

---

# Part A: Engineering foundations

## Module 0: A working Linux + GPU dev setup (2 to 3 days)

**Goal:** a WSL2 Ubuntu environment with GPU-enabled PyTorch, a debugger
attached to the repo, and an understanding of why this repo's install steps
look the way they do.

### Core

| Resource | Why | Time |
| --- | --- | --- |
| [MIT Missing Semester](https://missing.csail.mit.edu/): shell, shell tools, editors, version control, debugging and profiling | The day-to-day Linux skills every robotics codebase assumes. | 5 h |
| [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) | Ubuntu on Windows with your GPU visible inside it. The NVIDIA guide explains why you install the driver on Windows only, never inside WSL. | 1 h |
| [VS Code: Developing in WSL](https://code.visualstudio.com/docs/remote/wsl) and [Python debugging in VS Code](https://code.visualstudio.com/docs/python/debugging) | Breakpoints inside training and rollout loops. You will use this more than any other tool. | 1 h |
| [pip: Dependency Resolution](https://pip.pypa.io/en/stable/topics/dependency-resolution/) | Why the repo pins `numpy==1.26.4`, installs `robosuite==1.3.2 --no-deps`, and pins `cython==0.29.37` for `mujoco-py`. Read `environment.yml` and `pyproject.toml` comments after this. | 45 min |
| [tmux: Getting Started](https://github.com/tmux/tmux/wiki/Getting-Started) | Keeping long training runs alive after you close the terminal. | 30 min |

### Lookup references

- [Docker: Get started](https://docs.docker.com/get-started/) and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Most robot deployments and many shared GPU clusters run in containers.

### Build task

1. Clone the repo inside the WSL filesystem (`~/code/...`), not under
   `/mnt/c`. File access across the Windows boundary is many times slower and
   makes data loading look like a model bug.
2. Create the environment from `README.md`, then run `gsp check-env`,
   `python -m unittest discover -v`, and
   `python -c "import torch; print(torch.cuda.is_available())"`.
3. Set a VS Code breakpoint inside `Push-T/train/train_policy.py` and step
   through one batch. Inspect tensor shapes in the debugger.

### Done when

- [ ] `torch.cuda.is_available()` is `True` inside WSL.
- [ ] You can explain each unusual install step in `environment.yml`.
- [ ] You can pause a training loop and inspect a batch.

---

## Module 1: PyTorch engineering for policy training (4 to 5 days)

**Goal:** write, debug, and speed up a policy training loop yourself, rather
than only running the existing scripts.

### Core

| Resource | Why | Time |
| --- | --- | --- |
| [Karpathy: A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) | The practical debugging order: inspect data, overfit one batch, add complexity slowly. The most useful single page here. | 45 min |
| [Google Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) | How to run hyperparameter searches without fooling yourself. Read "Choosing the initial configuration" and "A scientific approach to improving model performance". | 2 h |
| [PyTorch: Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html) | `state_dict`, optimizer state, and resuming. Compare with the `torch.save({...})` in `Push-T/train/train_policy.py`. | 30 min |
| [PyTorch: Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html) | Seeding torch, NumPy, and DataLoader workers, and which CUDA ops are non-deterministic. | 30 min |
| [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) and [Profiler recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | Finding whether you are data-bound, CPU-bound, or GPU-bound. | 1.5 h |

### Lookup references

- [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html).
- Diffusion Policy's training workspace,
  [`train_diffusion_unet_lowdim_workspace.py`](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/workspace/train_diffusion_unet_lowdim_workspace.py),
  a clean reference for EMA, checkpointing, periodic rollouts, and logging in
  one loop.

### Build task

Write your own `train_minimal.py` for the Push-T flow policy, about 150 lines,
using only `guided_streaming_policy/core/` (`ConditionalUnet1D`,
`flow_policy.py`, `sequence.py`) and the Push-T dataset:

1. Overfit a single batch until the loss is near zero. If it does not reach
   near zero, you have a bug.
2. Add EMA, cosine LR schedule, checkpoint save/resume, and a fixed seed.
3. Profile 50 steps and report the share of time spent in data loading vs.
   forward/backward.
4. Compare your loss curve to `gsp run train-policy` with the same seed.

### Done when

- [ ] You can resume training from a checkpoint and get the same loss curve.
- [ ] You know whether training is data-bound or compute-bound, and why.

---

## Module 2: Robot datasets: formats, windows, normalization (3 days)

**Goal:** understand how demonstrations are stored and how they become
`(obs_horizon, pred_horizon)` training windows.

### Core

| Resource | Why | Time |
| --- | --- | --- |
| [h5py Quick Start](https://docs.h5py.org/en/stable/quick.html) | Reading groups, datasets, and attributes in robomimic HDF5 files. | 45 min |
| [Zarr tutorial (v2.18)](https://zarr.readthedocs.io/en/v2.18.3/tutorial.html) | Chunked arrays used by the Push-T replay buffer. The repo pins `zarr==2.18.3`. | 45 min |
| [robomimic: Dataset Contents and Visualization](https://robomimic.github.io/docs/tutorials/dataset_contents.html) and [robosuite Datasets](https://robomimic.github.io/docs/datasets/robosuite.html) | HDF5 layout (`data/demo_N/{obs,actions,states}`, `env_args`), filter keys, extracting observations from saved MuJoCo states. | 1.5 h |
| Diffusion Policy [`replay_buffer.py`](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/common/replay_buffer.py) and [`sampler.py`](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/common/sampler.py) | The reference implementation of `episode_ends` indexing and edge padding. This repo's `core/sequence.py` does the same job. | 1.5 h |
| Diffusion Policy [`robomimic_dataset_conversion.py`](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/scripts/robomimic_dataset_conversion.py) | How `low_dim_abs.hdf5` is produced: replay each demo and record absolute end-effector targets instead of delta actions. | 45 min |

### Lookup references

- [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/lerobot-dataset-v3),
  today's most common open format for new datasets (Parquet + MP4, hosted on
  the Hugging Face Hub). See Module 8.

### Build task

Write `inspect_data.py` that:

1. Prints the full tree of the Push-T Zarr file and of
   `Robomimic/datasets/can/ph/low_dim_abs.hdf5`, with shapes and dtypes.
2. Plots action trajectories for 5 demos of each, and a histogram of episode
   lengths.
3. Uses `create_sequence_pointers` and `extract_sequence` from
   `guided_streaming_policy/core/sequence.py` to produce windows, and checks
   that the first and last windows of an episode are padded the same way as
   Diffusion Policy's `SequenceSampler`.
4. Checks that normalize → unnormalize returns the original actions to within
   `1e-6`.

### Done when

- [ ] You can say what each dimension of the 10-D robomimic absolute action is
      (position, 6-D rotation, gripper; see
      `Robomimic/core/myrotation_transformer_final.py`).
- [ ] You can explain what goes wrong if normalization stats are computed on a
      different dataset than the one used at inference.

---

# Part B: Simulation in depth

## Module 3: How a physics simulator works, using Push-T and MuJoCo (5 days)

**Goal:** understand timesteps, control frequency, low-level controllers,
contacts, state vs. observation, and determinism well enough to change a
simulator and trust the result.

### Core

| Resource | Why | Time |
| --- | --- | --- |
| [Pymunk Tutorials](https://www.pymunk.org/en/latest/tutorials.html) | Bodies, shapes, `space.step(dt)`, damping, collision handlers. The full physics of Push-T. | 2 h |
| [Gymnasium: Create a Custom Environment](https://gymnasium.farama.org/introduction/create_custom_env/) | The `reset`/`step`/`render` contract, spaces, seeding, and what `terminated` vs. `truncated` mean. | 1 h |
| [MuJoCo Python tutorial (Colab)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb) | Load MJCF, step physics, read contacts, render, and see how small perturbations change contact-rich trajectories. The best hands-on introduction to MuJoCo. | 3 h |
| [MuJoCo: Overview](https://mujoco.readthedocs.io/en/stable/overview.html) and [Modeling guide](https://mujoco.readthedocs.io/en/stable/modeling.html) | `mjModel` vs. `mjData`, bodies, joints, geoms, actuators, sensors, and how to tune contact parameters and timestep for stability. | 3 h |

### Lookup references

- [MuJoCo XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
  (every MJCF attribute), [Computation](https://mujoco.readthedocs.io/en/stable/computation/index.html)
  (the solver, when you need to debug contacts), and
  [Python bindings](https://mujoco.readthedocs.io/en/stable/python.html)
  (includes the `MUJOCO_GL=egl|osmesa|glfw` rendering backends).
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie):
  curated, tested MJCF models of real robots (Franka, UR5e, ALOHA, and many
  more). Start new tasks from these instead of writing robot models yourself.

### Build task

Work in `Push-T/core/pusht_env.py`, which runs physics at `sim_hz = 100` and
the policy at `control_hz = 10`. Each action is a target position tracked by a
PD controller (`k_p = 100`, `k_v = 20`) for 10 physics substeps.

1. Write a scripted rollout that replays demo 0's actions from the dataset in
   `PushTEnv` with `reset_to_state` set to the demo's first state. Plot the
   replayed block pose against the recorded pose. Small drift is normal; large
   drift means a mismatch in state setting or physics parameters.
2. Change `k_p`/`k_v`, `damping`, and `control_hz`. Replay the same demo and
   record the effect on final coverage. This is a small sim-to-sim gap
   experiment.
3. Record 5 of your own demonstrations with `teleop_agent()` and the Pygame
   window (needs a display; WSLg provides one on Windows 11). Save them in the
   same Zarr layout as the original dataset.
4. In the MuJoCo Colab, load a Menagerie arm, run a joint-space PD controller,
   and render a video offscreen.

### Done when

- [ ] You can explain the difference between the simulator timestep, the
      control period, and the action chunk length, with this repo's numbers.
- [ ] You can explain why replaying recorded actions does not always reproduce
      the recorded trajectory.

---

## Module 4: robosuite and robomimic internals (5 days)

**Goal:** understand how a robot-arm task is assembled, how actions turn into
joint torques, and how this repo builds and calibrates the robomimic
environment.

### Core

| Resource | Why | Time |
| --- | --- | --- |
| [robosuite: Modules overview](https://robosuite.ai/docs/modules/overview.html), [Environments](https://robosuite.ai/docs/modules/environments.html), [Controllers](https://robosuite.ai/docs/modules/controllers.html) | How MJCF models, robots, grippers, and objects are composed into a task, and how the OSC_POSE controller maps a 7-D action to torques at a fixed control frequency. Delta vs. absolute control is explained here. | 3 h |
| [robosuite: Demonstrations](https://robosuite.ai/docs/algorithms/demonstrations.html) | How demonstrations are collected with teleop devices and stored as MuJoCo states plus actions. | 45 min |
| [robomimic: Getting Started](https://robomimic.github.io/docs/introduction/getting_started.html) | Playback, training, and rollout scripts that ship with robomimic. | 1 h |
| [mujoco-py README](https://github.com/openai/mujoco-py) | The deprecated binding that robosuite 1.3 uses: `LD_LIBRARY_PATH`, MuJoCo 2.1 binaries, and the Cython build on first import. | 30 min |

Version note: the robosuite docs now describe v1.5, which uses the official
`mujoco` bindings. This repo is pinned to v1.3.2 to match the released
datasets and checkpoints. The task, controller, and observation concepts are
the same; the rendering and install details differ.

### Build task

Read `Robomimic/core/robomimic_env.py` top to bottom. It contains a lot of
practical engineering:

- The top of the file sets `MUJOCO_GL=osmesa` and preloads `libOSMesa`,
  `libGL`, and `libglewosmesa` with `ctypes.CDLL(..., RTLD_GLOBAL)` before
  importing `mujoco_py`. Work out why import order matters for headless
  rendering.
- `resolve_control_delta` decides between delta and absolute control from the
  dataset metadata, then action statistics, then by replaying `demo_0` both
  ways and keeping the mode with the higher reward.
- `RobomimicLowdimWrapper.reset` caches the MuJoCo state for each seed, so
  repeated evaluations start from identical states.
- `step` sanitizes NaN/Inf actions and turns a `MujocoException` into a
  terminated episode.

Then:

1. Run `python -m core.robomimic_env` from `Robomimic/` and read the
   diagnostics it prints.
2. Replay 10 demos with `env.reset_to({"states": ...})` and record the success
   rate. It should be close to 100%. If it is not, your control mode or
   environment metadata is wrong.
3. Render a rollout to MP4 with `imageio`, and measure simulation steps per
   second with and without rendering.
4. Run `gsp run mimic-infer-si --task lift` with a released checkpoint.

### Done when

- [ ] Demo replay succeeds on at least 9 of 10 demos.
- [ ] You can explain delta vs. absolute OSC actions and why the repo does not
      clip absolute actions to `[-1, 1]`.

---

## Module 5: Evaluation and experiment infrastructure (4 days)

**Goal:** turn "run a script and watch a video" into repeatable, logged,
multi-seed experiments with honest statistics.

### Core

| Resource | Why | Time |
| --- | --- | --- |
| [Gymnasium: Vector environments](https://gymnasium.farama.org/api/vector/) | Running N rollouts in parallel processes. Evaluation is usually the slowest part of a policy experiment. | 1 h |
| [Hydra: Getting started](https://hydra.cc/docs/intro/) | Config composition and command-line overrides. Used by Diffusion Policy and many lab codebases. | 1.5 h |
| [Weights & Biases quickstart](https://docs.wandb.ai/quickstart) | Logging losses, rollout metrics, and videos so you can compare runs later. | 45 min |
| [Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264) + [`rliable`](https://github.com/google-research/rliable) | Confidence intervals and aggregate metrics with few seeds. You know the theory; focus on the library and the recommended reporting. | 1.5 h |
| `docs/reproducibility.md` in this repo | Rules for not silently changing benchmark results. | 15 min |

### Build task

Build `eval_harness.py` for Push-T that:

1. Takes a checkpoint, a method name (naive, STEG, CCG), a list of seeds, and
   a number of episodes.
2. Runs episodes in parallel, records per-episode max coverage, collisions,
   wall time, and policy inference latency.
3. Saves a CSV plus a few videos (best, median, worst episode), and logs to
   W&B.
4. Reports the mean with a 95% bootstrap confidence interval using `rliable`.

Use it to compare naive (unguided) streaming inference against CCG on static
Push-T with 3 seeds and 50 episodes each. Keep results outside git.

### Done when

- [ ] Two runs with the same seed give identical numbers.
- [ ] Your report includes confidence intervals and inference latency, not
      just mean success.

---

## Module 6: Capstone: add a new task end to end (1 to 2 weeks)

**Goal:** go through every engineering step a new research idea needs: build
an environment, collect demonstrations, train, guide, evaluate, and integrate
with the CLI.

Pick one task:

- **Easy (Pymunk):** Push-T with moving obstacles of a new shape, or a 2-D
  "push two blocks" task.
- **Medium (MuJoCo):** a Menagerie Franka or SO-101 arm reaching or pushing a
  cube while avoiding a keep-out region, using the official `mujoco` bindings.
- **Harder (robosuite):** a new robosuite task built from existing objects,
  following the [robosuite modeling docs](https://robosuite.ai/docs/modules/overview.html).

Steps:

1. Write the environment as a Gym `Env`, with a seeded `reset` and a
   `reset_to_state` path for replay.
2. Write a scripted expert, for example waypoints plus a simple IK or PD
   controller. Make it multimodal on purpose, such as going left or right
   around an obstacle. Collect 100 to 200 demos into Zarr or HDF5.
3. Train a flow policy using `guided_streaming_policy/core/`.
4. Write a reward-gradient function with shape `(batch, rewards, horizon,
   action_dim)` and wrap the policy with `ConflictAwareAdditiveGuidance`
   as shown in `README.md` (*Add a new robot flow policy*). Compare unguided,
   additive, and CAR guidance using your Module 5 harness.
5. Register the task in `guided_streaming_policy/cli.py` the same way
   `pusht_legacy.py` defines `LegacyTask` entries, and add unit tests for the
   environment (`reset` determinism, observation shapes) in `tests/`.

### Done when

- [ ] `gsp run <your-task>` trains and evaluates from a clean checkout.
- [ ] The tests pass and the results table has confidence intervals.

---

# Part C: The wider tooling landscape

## Module 7: Simulators, model formats, and benchmarks in use today (3 to 4 days, reading + one tutorial each)

**Goal:** know which simulator to choose for a new project, the robot
description formats you will meet, and the benchmarks reviewers expect.

### Simulators

| Tool | What it is used for | Start here |
| --- | --- | --- |
| **MuJoCo** (CPU) | The default for manipulation and control research; fast, accurate contacts, easy Python API. | Module 3 resources |
| **MJX and MuJoCo Warp** | MuJoCo on GPUs (JAX and NVIDIA Warp backends) for thousands of parallel environments. MJWarp favors throughput over single-step latency. | [MJX docs](https://mujoco.readthedocs.io/en/stable/mjx.html), [MJWarp docs](https://mujoco.readthedocs.io/en/latest/mjwarp/) |
| **MuJoCo Playground** | GPU environments for locomotion and manipulation with tested sim-to-real recipes, built on MJX/MJWarp. | [playground.mujoco.org](https://playground.mujoco.org/) |
| **Isaac Sim / Isaac Lab** (NVIDIA) | Large-scale GPU RL and imitation learning with photoreal rendering; common in locomotion, humanoids, and industry. Needs an RTX GPU. | [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/main/index.html), [NVIDIA "Getting Started With Isaac Lab" learning path](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/index.html) |
| **Newton** | Open-source GPU physics engine on NVIDIA Warp and OpenUSD, from NVIDIA, Google DeepMind, and Disney Research, managed by the Linux Foundation; 1.0 released March 2026 and integrated with Isaac Lab. | [GitHub](https://github.com/newton-physics/newton) |
| **ManiSkill3** (SAPIEN/PhysX) | GPU-parallel manipulation simulation and rendering, many tasks, and real-to-sim "digital twin" evaluation. CAR's upstream experiments use ManiSkill2. | [ManiSkill docs](https://maniskill.readthedocs.io/en/latest/) |
| **Genesis** | Newer multi-physics platform (rigid, soft, fluids) with a Pythonic API. | [Genesis docs](https://genesis-world.readthedocs.io/) |
| **Drake** (TRI/MIT) | Model-based planning, optimization, and simulation with careful contact modeling; used in Russ Tedrake's course. | [drake.mit.edu](https://drake.mit.edu/), [Robotic Manipulation course notes](https://manipulation.mit.edu/) |
| **Gazebo** | The standard simulator in the ROS ecosystem, used for system integration testing more than for learning. | [Gazebo docs](https://gazebosim.org/docs) |

PyBullet still appears in older papers but is no longer actively developed;
avoid it for new work.

### Robot description formats

- **URDF:** the ROS standard for kinematics, inertia, and meshes. Learn it from
  the [ROS 2 URDF tutorials](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/URDF/URDF-Main.html).
- **MJCF:** MuJoCo's XML format (Module 3). MuJoCo can load URDF, but most
  people start from Menagerie MJCF files.
- **USD (OpenUSD):** the scene format for Isaac Sim and Newton.
  [Introduction to USD](https://openusd.org/release/intro.html).

### Benchmarks you will see in policy papers

- **robomimic** and **Push-T** (in this repo).
- **LIBERO**, lifelong and multitask manipulation, common for VLA evaluation:
  [libero-project.github.io](https://libero-project.github.io/).
- **RoboCasa**, large-scale kitchen tasks on robosuite: [robocasa.ai](https://robocasa.ai/).
- **ManiSkill3** tasks (above).
- **SimplerEnv**, simulated evaluation that correlates with real-robot results:
  [simpler-env.github.io](https://simpler-env.github.io/).
- **RoboArena**, crowdsourced real-world evaluation of generalist DROID
  policies: [robo-arena.github.io](https://robo-arena.github.io/).

### Build task

Do the first tutorial of two simulators you have not used: MuJoCo Playground
(Colab, no install) and either ManiSkill3 or Isaac Lab. Record steps per
second for a single environment and for a batch of environments, and write
one paragraph on when you would choose each simulator over robosuite.

---

## Module 8: Real-robot deployment stack (reading + one hands-on path, 1 week)

**Goal:** understand what changes between a simulated rollout and a policy
running on a physical arm, and know the standard tools at each layer.

### The layers

| Layer | Standard tools | Start here |
| --- | --- | --- |
| Middleware | **ROS 2.** Current LTS releases are Jazzy Jalisco (supported to May 2029) and Lyrical Luth (released May 2026, supported to May 2031). Most lab code still targets Humble or Jazzy. | [ROS 2 Distributions](https://docs.ros.org/en/jazzy/Releases.html), [ROS 2 Tutorials](https://docs.ros.org/en/jazzy/Tutorials.html): do the Beginner CLI and Client Libraries tracks. |
| Hardware control | **ros2_control** for controllers and hardware interfaces. | [control.ros.org](https://control.ros.org/) |
| Motion planning | **MoveIt 2** for IK, collision checking, and trajectory planning. | [MoveIt 2 tutorials](https://moveit.picknik.ai/main/index.html) |
| Robot drivers | **Franka** (libfranka, franka_ros2) and **Universal Robots** (RTDE, `ur_rtde`) are the most common research arms. | [Franka docs](https://frankarobotics.github.io/docs/), [ur_rtde](https://sdurobotics.gitlab.io/ur_rtde/) |
| Low-cost learning stack | **LeRobot** (Hugging Face): drivers, teleoperation, recording, training, and deployment in pure Python, built around the low-cost SO-101 arm. The fastest way to get a real robot running. | [LeRobot docs](https://huggingface.co/docs/lerobot/index), [Hugging Face Robotics Course](https://huggingface.co/learn/robotics-course/unit0/1), [SO-101 guide](https://huggingface.co/docs/lerobot/so101) |
| Teleoperation | SpaceMouse, leader-follower arms (GELLO, SO-101 leader), and handheld grippers (UMI). | [GELLO](https://wuphilipp.github.io/gello_site/), [UMI](https://umi-gripper.github.io/) |
| Data formats | **LeRobotDataset v3** (Parquet + MP4) for new datasets; **RLDS** for Open X-Embodiment; DROID for large-scale Franka data. | [LeRobotDataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3), [RLDS](https://github.com/google-research/rlds), [Open X-Embodiment](https://robotics-transformer-x.github.io/), [DROID](https://droid-dataset.github.io/) |
| Pretrained policies | Vision-language-action models that labs fine-tune instead of training from scratch: **openpi** (π0, π0-FAST, π0.5; π0 is a flow-matching VLA), Octo, OpenVLA. | [openpi](https://github.com/Physical-Intelligence/openpi), [Octo](https://github.com/octo-models/octo), [OpenVLA](https://github.com/openvla/openvla) |
| Visualization and logging | **Rerun** for time-synced multimodal logs (used by LeRobot), **Foxglove** for ROS 2 bags. | [Rerun docs](https://rerun.io/docs), [Foxglove docs](https://docs.foxglove.dev/) |

### Deployment issues that simulation hides

Read these two codebases with the following questions in mind:

- Diffusion Policy's real-robot environment,
  [`real_env.py`](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/real_world/real_env.py):
  `step` is split into `get_obs` and `exec_actions`, observations carry
  timestamps, and actions are scheduled for future times to absorb inference
  latency.
- [UMI](https://umi-gripper.github.io/): latency matching between cameras,
  proprioception, and the robot.

Questions to answer: How old is the observation when the action executes? What
happens if inference takes longer than one control period? This is exactly
where streaming flow policies are meant to help, so measure it in your own
setup.

### Build task (choose one)

- **Hardware path:** build or buy an SO-101 pair, follow the LeRobot
  imitation-learning tutorial to record 50 demos, train ACT or Diffusion
  Policy, then port your flow policy as a LeRobot policy.
- **No-hardware path:** do the ROS 2 Beginner tutorials, load a Franka URDF in
  RViz, and write a ROS 2 node that subscribes to simulated joint states and
  publishes actions from a Push-T-style policy at 10 Hz. Measure the
  end-to-end latency.

---

## Theory-to-code map (reference)

You know the methods; this table shows where each lives in the code.

| Concept | Paper | Code |
| --- | --- | --- |
| Diffusion Policy (DDPM over action chunks) | [Chi et al., RSS 2023](https://arxiv.org/abs/2303.04137) | `Push-T/train/train_dp.py`, `inference_dp_gconva*.py` |
| Streaming flow policy | [Jiang, Ancha et al., CoRL 2025](https://arxiv.org/abs/2505.21851) | `guided_streaming_policy/core/flow_policy.py`, `Push-T/train/train_policy.py` |
| Stochastic interpolant `γ(t)` | [Albergo, Boffi, Vanden-Eijnden](https://arxiv.org/abs/2303.08797) | `gamma_t_si`, `d_gamma_dt_si` |
| 1-D conditional U-Net with FiLM | Diffusion Policy / [Diffuser](https://arxiv.org/abs/2205.09991) | `guided_streaming_policy/core/models.py` |
| Critic guidance (CCG, STEG) | Repo-specific | `Push-T/core/critic.py`, `Push-T/inference/static/inference_{ccg,steg}.py` |
| CAR guidance | [ICML 2026](https://arxiv.org/abs/2605.20758), [code](https://github.com/yuxuehui/CAR-guidance) | `guided_streaming_policy/core/car_guidance.py` |
| Related: constrained streaming flows | [arXiv 2602.15567](https://arxiv.org/abs/2602.15567) | n/a |

---

## Suggested 6-week schedule

| Week | Work | Milestone |
| --- | --- | --- |
| 1 | Modules 0, 1 | GPU env in WSL; your own training loop overfits one batch and resumes from a checkpoint. |
| 2 | Modules 2, 3 | Data inspector done; Push-T demo replay and PD/damping sweep. |
| 3 | Module 4 | Robomimic demo replay at 9/10 or better; headless MP4 rendering. |
| 4 | Module 5; read Module 7 | Multi-seed eval harness with confidence intervals and latency. |
| 5 to 6 | Module 6; read Module 8 | New task runs through `gsp`, with CAR vs. additive vs. unguided results. |

## When you get stuck

- Install and rendering errors: search closed issues for
  [robomimic](https://github.com/ARISE-Initiative/robomimic/issues),
  [robosuite](https://github.com/ARISE-Initiative/robosuite/issues),
  [mujoco-py](https://github.com/openai/mujoco-py/issues), and
  [Diffusion Policy](https://github.com/real-stanford/diffusion_policy/issues)
  first. Most errors are already answered there.
- Physics looks wrong: check, in order, timestep and substeps, units and
  scales, controller gains, contact parameters (MuJoCo Modeling guide), and
  whether you set the state before or after `reset`.
- A policy trains but fails in rollout: replay the demos first. If replay
  fails, the bug is in the environment or action convention, not the model.
