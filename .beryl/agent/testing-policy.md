# Testing Policy

## Command Matrix

| Check | Command | Status | Notes |
| --- | --- | --- | --- |
| Markdown sanity | `./.beryl/scripts/check-md.sh` | available | Unclosed fences and tabs |
| Beryl deterministic gate | `./.beryl/scripts/check.sh` | available | Beryl surface, markdown, secrets, and project checks |
| Unit tests | `python -m unittest discover -v` | available | Focused tests in `tests/` |
| Compile smoke check | `python -m compileall -q guided_streaming_policy Push-T tests` | available | Catches syntax/import compilation errors |
| Environment check | `gsp check-env` | available after install | Dependency and asset readiness |
| Lint | not configured | unavailable | Do not invent a formatter or linter |
| Typecheck | not configured | unavailable | Python project has no typecheck command |
| Robomimic runtime | Linux + MuJoCo 2.1 + OSMesa | platform-dependent | Not runnable on this Windows host |

## Default Loop

1. Identify the behavior or experiment contract being changed.
2. Run the smallest relevant unit test or compile check.
3. For experiment changes, state the task, dataset, checkpoint, seed, output
   directory, and comparison baseline before implementation.
4. Run `python -m unittest discover -v`, the compile check, and
   `./.beryl/scripts/check.sh` when the change is broad or documentation/config
   is affected.
5. Report platform-dependent or asset-dependent checks that could not run.

## Affected Test Gate

The Beryl affected-test mechanism is retained, but this repository uses the
explicit unittest command as its reliable full test command. Related test
selection is not configured because the project tests are small and benchmark
paths do not map safely to individual test modules.

## Experiment Verification

An experiment change must preserve or explicitly document its benchmark, task,
dataset version, checkpoint, seed, normalization statistics, command, and
expected output location. Compare methods under matched settings before
interpreting a result.

## Robotics-Specific Checks

- Verify seeded reset determinism and observation/action shapes for new
  environments.
- Verify `reset_to_state` or equivalent replay initialization before evaluating
  a learned policy.
- Replay recorded demonstrations and compare trajectory/success against the
  recorded data; investigate replay failures as environment or action-mode
  defects first.
- Test sequence-window edge padding and normalize/unnormalize round trips
  within the documented tolerance.
- Record simulator timestep, control period, action chunk length, controller
  gains, damping, and control mode whenever they change.
- For evaluation harnesses, verify repeated same-seed results, per-episode
  metrics, episode counts, confidence intervals, and inference latency.
- Keep videos, CSVs, checkpoints, datasets, and rendered trajectories outside
  version control.

## Test Modification Rule

Existing tests may not be weakened to make implementation pass. Intentional
test changes must cover an explicit behavior change and be explained in the
final handoff.
