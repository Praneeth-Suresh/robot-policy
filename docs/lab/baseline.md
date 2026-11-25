# Policy Trace Lab baseline

Orientation notes (task T1). Later decisions cite this page. Paths are relative to the repository root.

## Starting point

| Item | Value |
| --- | --- |
| Starting commit | `6a0b8ad` (`Merge pull request #1 from Praneeth-Suresh/docs/robot-policy-visitor-quickstart`) |
| Branch | `feature/policy-trace-lab` |
| OS | Linux 6.6.87.2-microsoft-standard-WSL2 x86_64 (WSL2) |
| Host Python | 3.14.5 (only `python -m guided_streaming_policy ...` smoke commands use it) |
| Project Python | 3.10.20 (`.venv`), torch 2.9.1 (CPU, `cuda.is_available() == False`), numpy 1.26.4, pymunk 7.2.0, zarr 2.18.3, gym 0.26.2 |
| GPU | None. All real runs are CPU-only. |

## Smoke commands and baseline results

| Command | Exit | Trimmed output |
| --- | --- | --- |
| `python3 -m guided_streaming_policy list-tasks` | 0 | 18 rows, e.g. `infer-static-naive  inference.static.inference_naive  Run static unguided streaming inference.` |
| `python3 -m guided_streaming_policy run infer-static-naive --dry-run` | 0 | `cwd=.../robot-policy/Push-T` then `<python> -m inference.static.inference_naive` |
| `.venv/bin/python -m unittest discover -s tests` | 0 | `Ran 19 tests ... OK (skipped=1)` |
| `.venv/bin/python -m compileall -q guided_streaming_policy Push-T tests` | 0 | no output |
| `METHOD=naive CONDITION=static COUNT=1 SEED=0 .venv/bin/python Push-T/run_trial.py` | 0 | 31 s wall on CPU; `RESULT static naive 0 {... 'guided_success': True, 'guided_min_dist': 44.51456 ...}` |

## Tooling already in the repo

- Tests: `unittest` under `tests/`. No pytest in the venv.
- Lint, format, type check: none configured (`.beryl/agent/testing-policy.md` says so). The repository has no `.github/workflows` directory, so there is no existing CI to preserve.
- Beryl gate: `./.beryl/scripts/check.sh` (`--development` in a Beryl source checkout).
- Decision: add `ruff`, `mypy` and `vulture` as dev-only tools (approved by the user). They go in a `dev` optional-dependency group, never in the default install.

## How a Push-T trial runs

- Entry point: `Push-T/run_trial.py` (env vars `METHOD`, `CONDITION`, `COUNT`, `SEED`). It `chdir`s to `Push-T/` and imports `inference.static.<module>` or `inference.dynamic.<module>`.
- Each inference module is import-heavy: at import time it loads the dataset (`Push-T/data/pusht_cchi_v7_replay.zarr.zip`) for normalization statistics, builds `env = PushTEnv()` as a module global and loads checkpoints.
- A call such as `run_naive_inference(seed=..., fixed_obstacle_pos=[-0.3, -0.5], ...)` runs **three** episodes inside one nested `run_episode`:
  1. no obstacle, `record_traj=True` (only used to place a percentage obstacle; unused when `fixed_obstacle_pos` is given);
  2. obstacle, `force_no_guidance=True` (the runner's own "base" result);
  3. obstacle, guidance on (the runner's "guided" result).
- Seeding: `torch.manual_seed`, `np.random.seed`, `random.seed`, then `env.reset(seed=seed)` fails with `TypeError` (the env's `reset` takes no arguments), the bare `except` calls `env.seed(seed)`, then `env.reset()` (e.g. `inference_naive.py:103-115`). The initial state is `RandomState(seed)` draws: agent `randint(50,450)` x2, block `randint(100,400)` x2, angle `randn()*2*pi - pi` (`core/pusht_env.py:260-281`).
- Static condition: obstacle at `fixed_obstacle_pos=[-0.3, -0.5]` in **normalized action space** (`run_trial.py`), converted to pixels with `unnormalize_data(pos, stats['action'])`. Dynamic condition: moving obstacle via `inference/dynamic/*`; **not in this release** (one condition at launch).
- Return value: `(metrics_dict, images)`. `metrics_dict` holds `base_*` and `guided_*` collision, reward, success and min_dist.

## Spaces, units and frame (`Push-T/core/pusht_env.py`)

| Item | Value | Source |
| --- | --- | --- |
| Observation | 5 float64: `agent_x, agent_y, block_x, block_y, block_angle`. Pixels of a 512 x 512 window; angle in radians mod 2*pi | `:330-335`; space bounds `:224-238` |
| Action | 2 float: absolute target position `(x, y)` for the agent, in the same pixel frame (0..512). Tracked by a PD controller (`k_p=100`, `k_v=20`) | `:232-238`, `:283-296` |
| Timing | `sim_hz=100`, `control_hz=10`, so `dt = 0.1 s` per control step, 10 physics substeps | `:216-219`, `:284-290` |
| Frame | pygame-style, y points down (`positive_y_is_up = False`), origin top left | `:18` |
| Goal | fixed pose `(256, 256, pi/4)` | `:482` |
| Info per step | `pos_agent`, `vel_agent`, `block_pose`, `goal_pose`, `n_contacts` | `:347-356` |
| Reward | `clip(coverage / 0.95, 0, 1)` per step | `:303-306` |

## Native success and termination

- Env: `coverage = area(goal ∩ block) / area(goal)`; `done = coverage > success_threshold` (0.95, `:489`); the env returns `terminated = truncated = done` (`:307-309`). The env has no step limit.
- Runner: loops `while not done and step_idx < max_steps` with `max_steps = 250` (e.g. `inference_naive.py:156-159`, same in `inference_ccg.py:186`, `inference_steg.py:169`, `inference_dp_gconva.py:158`, `inference_fp_gconva.py:145`). `done` is the env's third return value, so the runner treats it as `terminated`.
- **Native success** (copied verbatim by the lab): `is_success = (not is_collision) and (max_reward > 0.85)` (`inference_naive.py:263`; `inference_ccg.py:280`; `inference_steg.py:305`; `inference_dp_gconva.py:313`; `inference_fp_gconva.py:283`). `is_collision = min_dist_to_obs < COLLISION_THRESHOLD`. The success metric already includes the runner's own collision test.
- `min_dist_to_obs` is the minimum over steps of a distance to the obstacle. In `naive`, `ccg` and `steg` it is `||a_real - obstacle_px||`, where `a_real` is the **action target** about to be executed (`inference_naive.py:174-177`). **Erratum, found by the real validation run:** in `dp` and `flow` it is `||info['pos_agent'] - obstacle_px||`, the **agent position** after each step (`inference_dp_gconva.py:265-267`, `inference_fp_gconva.py:238-240`). The traces reproduce both exactly: `action_executed.target` for the first three, `obs.agent_pos` for the last two (see [real run](real-run.md)).
- `COLLISION_THRESHOLD = 0.06 * 256 * 2 = 30.72` pixels in all five runners.
- Because the env sets `terminated == truncated`, the lab records `truncated` as "the runner step cap (250) was reached without env termination" and says so in `limitations.md`.

## Telemetry the runner exposes, and what it does not

| Available (by wrapping the module-level `env`) | Not available |
| --- | --- |
| Reset observation and reset info (actual start state) | Policy's proposed action before post-processing (`action_proposed`) is not separable from the executed action in the five runners |
| Per control step: observation, executed action (`a_real`), reward, `terminated`, `truncated`, info (`pos_agent`, `vel_agent`, `block_pose`, `n_contacts`) | Per-step timestamps. Time is derived from the fixed control rate (`dt = 0.1 s`), so the timebase is `fixed_dt` |
| Runner return value: native `guided_success`, `guided_min_dist`, `guided_reward`, `guided_collision` | Any hook in the runners. They are monolithic functions, so the exporter replaces the module global `env` with a recording proxy and leaves the research code untouched |

Obstacle geometry: the obstacle is **virtual**. It is a point in normalized action space, never added to the physics space, with an effective collision radius of 30.72 px (centre-to-action-target distance). It is static in the static condition. The separation proxy therefore measures distance between an executed action target and a fixed point, in simulator pixels (512-window units). It is a proxy, not physical contact.

## Demonstration data

- File: `Push-T/data/pusht_cchi_v7_replay.zarr.zip` (present). Zarr ZipStore with `data/{action (25650,2), state (25650,5), keypoint, img, n_contacts}` and `meta/episode_ends (206,)`.
- 206 episodes delimited by `episode_ends` (first ends: 161, 279, 420, 579, 738; last 25650).
- **No train/eval split** is stored. Doctor therefore reports `split.not_checked` (info) for the native data and never infers a split.
- Timestamps: none stored; the original data is 10 Hz (`control_hz`).
- Reader code for training (`guided_streaming_policy/pusht/dataset.py`) windows episodes; the lab reader works on the original episodes.

## Assets

| Asset | Present | Used by |
| --- | --- | --- |
| `Push-T/data/pusht_cchi_v7_replay.zarr.zip` | yes | all runners, Doctor |
| `Push-T/checkpoints/pusht_state_1000ep_si.ckpt` | yes | `naive`, `ccg`, `steg` |
| `Push-T/checkpoints/CCG_V6_ep200.pth` (critic) | yes | `ccg` |
| `Push-T/checkpoints/pusht_state_100ep_dp.ckpt` | yes | `dp` |
| `Push-T/checkpoints/pusht_state_flow_100ep.ckpt` | yes | `flow` |

Fetch instructions are in the repository README ("Setup and assets"). SHA-256 of the files in this checkout:

| File | SHA-256 |
| --- | --- |
| `pusht_state_1000ep_si.ckpt` | `e3b722c9a87425944a45dae7bd211ef4721b2bb92e50d93244ae795642257127` |
| `CCG_V6_ep200.pth` | `97c187bc1c4acf691bf6ee4aaea232a8d09f11effd34fd736d0472c37e1cf583` |
| `pusht_state_100ep_dp.ckpt` | `9465fccdb6e249b6332f687ac90f6fbaee388de5f7b5a66bf0d5204e4e211264` |
| `pusht_state_flow_100ep.ckpt` | `6c39a99056c3210291dd6311f40a748acff84f5b566afc4eadfd691c9f747899` |

## Reference-policy inventory (Push-T, static condition)

| Registered name | Runner (`Push-T/inference/static/`) | Checkpoints | Tunable settings (keyword of the runner function) | Paper |
| --- | --- | --- | --- | --- |
| `pusht/naive` | `inference_naive.run_naive_inference` | SI | `max_guidance_scale` (run_trial default 6.0; function default 5.0) | Streaming flow policy, Jiang, Ancha et al., CoRL 2025, arXiv:2505.21851; stochastic interpolants, Albergo, Boffi, Vanden-Eijnden, arXiv:2303.08797 |
| `pusht/ccg` | `inference_ccg.run_ccg_inference` | SI + critic | `max_guidance_scale` (run_trial 4.0; function default 1.0) | same SI papers; critic-guided collision avoidance as implemented in this repo (no separate paper cited in the repo) |
| `pusht/steg` | `inference_steg.run_steg_sde_inference` | SI | `guidance_scale` (1.5), `N_ensemble` (64) | same SI papers; STEG as implemented in this repo (no separate paper cited in the repo) |
| `pusht/dp` | `inference_dp_gconva.run_gconv_a_dp_inference` | DP | `guidance_scale` (2.0), `guidance_sign` (+1.0), `grad_clip_norm` (1.0) | Diffusion Policy, Chi et al., RSS 2023, arXiv:2303.04137 |
| `pusht/flow` | `inference_fp_gconva.run_gcov_a_fp_inference` | flow | `guidance_scale` (5.0) | flow policy as implemented in this repo, built on the streaming flow policy line of work |

All five run from the published release on this machine. The range the repo's own docs support for each setting is not documented anywhere except through the defaults above and the dynamic-condition scales in `Push-T/run_trial.py` (`naive 7.0`, `ccg 4.0`, `steg 7.0`, `dp 15.0`, `flow 0.7`); the README for the lab will cite exactly those values and label them as repo defaults, not recommended ranges.

## Public interfaces the lab touches (additive only)

| Interface | Change |
| --- | --- |
| `gsp` CLI (`guided_streaming_policy/cli.py`) | Add the `lab` subcommand group; all existing subcommands unchanged. |
| `pyproject.toml` | Add a `dev` optional-dependency group (`ruff`, `mypy`, `vulture`, `jsonschema`) and package data for schemas/fixtures. The default `dependencies = []` stays empty. |
| `Push-T/run_trial.py` and runners | Not modified. The exporter wraps the module-global `env`. |
| `.gitignore` | Add `out/`. |
| New: `guided_streaming_policy/lab/`, `schemas/`, `lab/` (comparison, gate and pin configs), `tests/fixtures/lab/`, `docs/lab/`, `docs/brand/`, `docs/img/`, CI workflow | New files only. |

## CLI names

The proposed names in the build prompt are confirmed unchanged: `gsp lab doctor|run|compare|gate|report|demo|policies`, with `--task` defaulting to `pusht`. `gsp lab` is a nested argparse group, which the existing `argparse` structure supports.

## Beryl workflow findings and conflicts

| Finding | Handling |
| --- | --- |
| Beryl provides no task tracker. `hierarchy.md` is for the `initial-build` workflow only, and the routing says progress stays out of canonical files. | The session task list tracks T1-T15. Only durable decisions go to `.beryl/agent/architecture.md` and `ubiquitous-language.md`. Codex invocation counts are summarised in the handoff, not committed. |
| `AGENTS.md` requires an approved plan before edits. | Plan presented and approved by the user in this session. |
| `AGENTS.md` requires an itemised list and approval before deleting root docs. | The removal manifest is shown to the user for approval before any pre-existing file is deleted. |
| `AGENTS.md` forbids sub-agents. | The user waived this for the session. Codex is run through `codex exec` only. |

## Codex

| Item | Value |
| --- | --- |
| Version | `codex-cli 0.172.0` |
| Working command | `codex exec --sandbox workspace-write --ephemeral - <<'PROMPT' ... PROMPT` run from the repository root on the feature branch (prompt read from stdin with `-`) |
| Check | A no-edit test prompt returned the branch name and left `git status` clean. |
