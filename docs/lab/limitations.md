# Limitations

## Non-goals

- Policy Trace Lab makes no claim about which method is better in general. A report states what the declared gate rule found for the runs it was given.
- It claims no performance lift and no physical safety guarantee.
- Out of scope for this release: LeRobot v3 interoperability, simulators other than Push-T (Robomimic comes later through [adding a task](adding-a-task.md)), hardware validation, training changes, hosted dashboards, new model architectures.
- It does not retrain policies or change simulator physics.

## Proxy metrics

- **Separation proxy.** The Push-T obstacle is virtual: a point in normalized action space that is never added to the physics. The proxy is the minimum distance, in simulator pixels (512-pixel window), between a recorded position signal and the obstacle centre. The signal is named in the gate rule: `action_executed.target` in `lab/gates/pusht_v1.json`, `obs.agent_pos` in `lab/gates/pusht_agent_v1.json`. It is not contact and says nothing about forces or physical safety. The runners differ in what their own `guided_min_dist` measures: the executed action target for `pusht/naive`, `pusht/ccg` and `pusht/steg`, the agent position after each step for `pusht/dp` and `pusht/flow`. In real runs the matching signal reproduces the runner's value exactly (see [real run](real-run.md)). Use the signal that matches the policy under test. The observation of a step is the one before its action, so the agent position after the very last step is not in the trace; a minimum reached only there would be missed.
- **Native success** is the runner's own result (`guided_success`: no collision under the runner's definition and maximum reward above 0.85). The lab copies it and never recomputes it. A crashed trial counts as not successful in the rates and is listed as `crashed`.
- **`truncated`.** The environment returns `terminated == truncated`, so the lab records `truncated` when the runner's 250-step cap ended the trial without termination.
- **Action magnitude** in the replay is the step-to-step change of the executed target, shown for reading only.
- **First discrepancy.** For regressions with no separation violation (for example a truncated trial) the replay opens at the last step of the candidate episode and says so.

## What the data checks cannot see

- The Push-T demonstration file has no train/eval split, so Doctor reports `split.not_checked` and makes no leakage claim.
- Leakage is detected by fingerprinting each original episode after rounding to three decimals in the field's own units. Near-duplicates beyond that precision are not detected.
- Doctor does not check camera frames. It works on the low-dimensional state and action data.
- Doctor checks gaps only when the timebase is known (`fixed_dt` at 0.1 s for Push-T).

## Cost of recording

Each runner call runs three episodes and the lab keeps one. A Push-T trial takes from about 20 s (`naive`, `ccg`, `flow`) to about 275 s (`dp`) on the CPU of the development machine. There is no GPU path in this release.

## Fixtures versus real runs

| Capability | Verified on a real Push-T run | Fixtures only |
| --- | --- | --- |
| Demonstration reader and Doctor on the native data | yes | corrupt and leakage cases |
| Exporter and `gsp lab run` | yes | none |
| Reset fingerprints equal for equal seeds | yes | none |
| Compare, gate, report | yes (see [real run](real-run.md)) | all gate verdict paths (missing signal, mismatched resets, missing pairs, truncated, success that violates separation) |
| Offline demo | no (uses synthetic fixtures) | yes |

Fixture data is labelled `synthetic` in every manifest and in the report header. It is never a benchmark.

## Known risks for the next task (Robomimic)

Listed in [adding a task](adding-a-task.md).
