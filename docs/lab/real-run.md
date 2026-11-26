# Real validation run (verified in simulation)

Everything on this page comes from real Push-T simulator runs on CPU. None of it uses fixtures. The gate verdict is whatever the declared rule produced. The page makes no claim about which method is better in general: five seeds of one condition cannot support one.

## Setup

| Item | Value |
| --- | --- |
| Choice and reason | recorded before the run in [decisions](decisions.md) |
| Comparison | `lab/comparisons/pusht_guidance_v1.json`, sha256 `6756808ba36dd26c4009fd003a8e2eb85d0e914aff1ab3625dd08089d61ebb78`: `pusht/dp` (default `guidance_scale` 2.0) against `pusht/dp` with `guidance_scale` 0.0, condition `static`, seeds 0 to 4 |
| Gate | `lab/gates/pusht_v1.json`, sha256 `da9f024818cebcc7118bf6c9c9c2344dce3b380561d54d6dedb34ed9c3763c8b`, committed before the run |
| Commits | the runs were recorded at `075f07d`; Doctor, compare, gate and report were re-run on the same traces at `9727252` (the exporter did not change between them) |
| Pins | `pusht/dp` checkpoint `9465fccdb6e249b6332f687ac90f6fbaee388de5f7b5a66bf0d5204e4e211264`, resolved config `56dbb9a4df2fc5f23a3107e53394b5c232ac68e31028369503c217fe88bbb42b` (baseline) |
| Versions | Python 3.10.20, torch 2.9.1 (CPU), numpy 1.26.4, zarr 2.18.3, pymunk 7.2.0 |
| Machine | Linux 6.6.87.2 (WSL2), Intel Core Ultra 7 258V, 8 logical CPUs, no GPU |
| Duration | `pusht/dp`: 394 s to 687 s per trial (baseline 2554 s, candidate 3259 s for five trials each) |

## Commands

```bash
gsp lab policies
gsp lab doctor Push-T/data/pusht_cchi_v7_replay.zarr.zip --out out/real/doctor-demos
gsp lab run --comparison lab/comparisons/pusht_guidance_v1.json --variant baseline --out out/real/runs/baseline
gsp lab run --comparison lab/comparisons/pusht_guidance_v1.json --variant candidate --out out/real/runs/candidate
gsp lab doctor out/real/runs/baseline --out out/real/doctor-baseline
gsp lab doctor out/real/runs/candidate --out out/real/doctor-candidate
gsp lab compare out/real/runs/baseline out/real/runs/candidate --comparison lab/comparisons/pusht_guidance_v1.json --out out/real/compare
gsp lab gate out/real/compare/compare.json --gate lab/gates/pusht_v1.json --out out/real/gate
gsp lab report out/real/compare/compare.json --gate out/real/gate/gate.json --out out/real/report
```

Exit codes: `policies` 0, `doctor` 0 (native data; one info finding), `run` 0 and 0, `doctor` 0 and 0, `compare` 0, `gate` **1**, `report` 0. Run directories, traces and checkpoints are not committed. See [reproducibility](reproducibility.md).

## Results

### Data and rollouts

| Input | Doctor result |
| --- | --- |
| `Push-T/data/pusht_cchi_v7_replay.zarr.zip` (206 episodes, 25,650 steps) | 0 errors, 0 warnings, 1 info: `split.not_checked` (the file has no split, so no leakage claim is made) |
| baseline run (5 episodes) | 0 findings |
| candidate run (5 episodes) | 0 findings |

### Paired outcomes (all 5 pairs completed, none incompatible or missing)

Native success is the runner's own result. "Separation proxy" is the minimum distance in pixels between the executed action target and the obstacle centre.

| Seed | Baseline success | Baseline steps, reason | Baseline proxy | Candidate success | Candidate steps, reason | Candidate proxy |
| --- | --- | --- | ---: | --- | --- | ---: |
| 0 | false | 250, `max_steps` | 9.65 | false | 250, `max_steps` | 4.11 |
| 1 | true | 151, `coverage_reached` | 29.20 | false | 250, `max_steps` | 9.92 |
| 2 | true | 152, `coverage_reached` | 29.15 | false | 190, `coverage_reached` | 12.06 |
| 3 | false | 250, `max_steps` | 7.91 | false | 250, `coverage_reached` | 12.49 |
| 4 | false | 250, `max_steps` | 8.07 | false | 250, `max_steps` | 9.00 |

Native success rate: baseline 0.40, candidate 0.00, delta -0.40 (2 losses, 3 ties, 0 wins). Both variants have the same reset fingerprint for each seed.

### Gate (`pusht_v1`): REGRESSION, exit 1

| Rule | Verdict | Baseline | Candidate | Delta | Threshold |
| --- | --- | ---: | ---: | ---: | ---: |
| `success_not_worse` | fail | 0.40 | 0.00 | -0.40 | -0.10 |
| `separation_proxy` (signal `action_executed.target`) | fail | 7.91 px | 4.11 px | -3.80 | 30.72 px |

Regressions in order of first violation: seed 0 (`constraint_fail`, step 3, t = 0.3 s), seed 3 (`constraint_fail`, step 77), seed 1 (`truncated`, step 106), seed 2 (`native_fail`, step 110), seed 4 (`constraint_fail`, step 142). The report opens at seed 0, step 3.

## What the checks found in real data

**The separation signal that the runner measures depends on the policy.** The lab's `obstacle_separation_min` rule was written, from reading `inference_naive.py` only, to use the executed action target. The export then let the lab compare its own separation proxy with the runner's reported `guided_min_dist` on every episode:

| Policy | Runner's `guided_min_dist` is measured on | Matches the lab's proxy on that signal |
| --- | --- | --- |
| `pusht/naive`, `pusht/ccg`, `pusht/steg` | the executed action target | yes, to the printed precision (naive 44.5146, ccg 32.5785, steg 12.4702 px, seed 0) |
| `pusht/dp`, `pusht/flow` | the agent position `info['pos_agent']` after each step | not on `action_executed.target` (dp seed 0: 11.2512 native, 9.6473 proxy; flow seed 0: 60.8266 native, 137.7514 proxy); yes on `obs.agent_pos` for all 12 dp and flow episodes (for example 11.2512, 32.2830, 33.2851, 8.3138, 11.6005 for the baseline run) |

[Baseline](baseline.md) carries the erratum. The gate chosen before the run used `action_executed.target`; it stays the primary result. Because the dp runner's native quantity is the agent position, the same comparison was gated once more with a **supplementary config declared after seeing this result** (`lab/gates/pusht_agent_v1.json`, identical except for `signal` set to `obs.agent_pos`, sha256 `5248dc656c8d6e9de45cc18a1875638149f3d26f9073ebd2310de4234f2e368c`):

| Rule | Verdict | Baseline | Candidate |
| --- | --- | ---: | ---: |
| `success_not_worse` | fail | 0.40 | 0.00 |
| `separation_proxy` (signal `obs.agent_pos`) | fail | 8.31 px | 2.40 px |

The verdict is the same, REGRESSION with exit 1. First violations: seed 0 step 5, seed 3 step 80, seed 1 step 109, seed 4 step 144, seed 2 step 146.

Two further observations from the same data:

- Candidate seeds 2 and 3 reached the coverage target (`terminated` true, reward 1.0) but are `native_success` false, because the runner's own collision test failed. Native success already includes the runner's collision rule, so "reached the goal" and "succeeded" differ. The lab keeps both fields.
- In the baseline run, seeds 1 and 2 are native successes whose action targets still came within 29.2 px of the obstacle centre (threshold 30.72 px). The runner's own collision test on agent positions did not flag them (32.28 and 33.29 px). The lab reports the two measurements separately and never merges them.

## Every registered policy, one seed

Seed 0, condition `static`, default settings, export on, one comparison config per policy (the loop below), then Doctor on each result.

```bash
for p in naive ccg steg dp flow; do
  cat > /tmp/smoke_$p.json <<JSON
{"schema_version":"0.1","name":"smoke_$p","task":"pusht","simulator_version":"7.2.0",
 "matrix":{"conditions":["static"],"seeds":[0]},
 "variants":{"baseline":{"policy":"pusht/$p"},"candidate":{"policy":"pusht/$p"}},
 "intended_differences":[]}
JSON
  gsp lab run --comparison /tmp/smoke_$p.json --variant baseline --out out/real/smoke/$p
  gsp lab doctor out/real/smoke/$p --out out/real/smoke-doctor/$p
done
```

| Policy | Status | Native success | Steps, reason | Duration | Doctor |
| --- | --- | --- | --- | ---: | --- |
| `pusht/naive` | runnable | true | 98, `coverage_reached` | 60 s | 0 findings |
| `pusht/ccg` | runnable | true | 81, `coverage_reached` | 63 s | 0 findings |
| `pusht/steg` | runnable | false | 250, `max_steps` | 228 s | 0 findings |
| `pusht/dp` | runnable | false | 250, `max_steps` | 691 s | 0 findings |
| `pusht/flow` | runnable | false | 250, `max_steps` | 67 s | 0 findings |

All five were available on this machine. A policy whose checkpoint is missing is listed as `unavailable` with the reason by `gsp lab policies` and refused by `gsp lab run` with exit 2.

## Screenshots

Each image comes from this run. The README shows them with the same text pasted beside them.

| File | Shows | Run and commit |
| --- | --- | --- |
| `docs/img/real-1-policies.png` | `gsp lab policies` | `pusht_guidance_v1`, commit `9727252` |
| `docs/img/real-2-doctor.png` | Doctor on the native data and both runs | same |
| `docs/img/real-3-run.png` | the two `gsp lab run` summaries | runs recorded at commit `075f07d` |
| `docs/img/real-4-compare.png` | `gsp lab compare` with its coverage line | commit `9727252` |
| `docs/img/real-5-gate.png` | the gate verdict and exit code | same |
| `docs/img/real-6-report.png` | the report overview, with the coverage panel, results table and regression list | same |
| `docs/img/real-7-replay.png` | the replay opened at the first discrepancy (seed 0, step 3) | same |

## Verified in simulation versus fixture only

Everything on this page is verified in simulation. The offline demo and the gate cases for missing signals, mismatched resets, missing pairs and truncated trials are fixture only: see [limitations](limitations.md).
