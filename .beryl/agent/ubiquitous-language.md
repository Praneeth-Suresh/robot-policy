# Ubiquitous Language

| Business Term | Technical Symbol | Definition | Constraints | Avoid |
| --- | --- | --- | --- | --- |
| Design Concept | `DesignConcept` | The shared organizing model guiding architecture and implementation choices. | Must be coherent across contexts. | `Idea`, `GeneralModel` |
| Decision Record | `DecisionRecord` | One-line entries in `architecture.md` for settled durable decisions: date, choice, and reason. | One line per decision; update when a boundary, contract, or test strategy changes. | `PlanDump`, long design essays |
| Bounded Context | `BoundedContext` | A domain boundary with explicit ownership and language. | Owns its internal model and API. | `Module` (too generic) |
| Ubiquitous Language | `UbiquitousLanguage` | Shared domain vocabulary in docs and code. | Terms must be stable and explicit. | Ambiguous nouns like `Data` |
| Feedback Loop | `FeedbackLoop` | Generate-check-fix cycle using deterministic tools. | Must include real tool output. | `TryAgainLoop` |
| Success Check | `SuccessCheck` | Pre-coding acceptance condition that proves a plan, redirect, feature slice, or bug fix worked. | Must name the expected artifact, deterministic command, generated output or browser evidence when applicable, and at least one user-visible behavior. | Vague done criteria such as `LooksGood` |
| Commit Boundary | `CommitBoundary` | Proposed repo-history unit for implementation work. | Must have one purpose, expected files, and a validating check command; should not mix generated output, docs, tests, and source unless required. | Large mixed commit |
| Generated Output | `GeneratedOutput` | Built artifact, copied asset, static HTML, feed, sitemap, robots file, search index, or similar file users or crawlers receive. | Static-site changes must verify affected generated output, not source alone. | Source-only verification |
| Affected Test Gate | `.beryl/scripts/check-affected.sh` | Commit and manual check step that maps changed files to related tests or full-test fallback. | Must be deterministic and run through `.beryl/scripts/check.sh`. | Ad hoc local test picking |
| Full Test Fallback | `FULL_TEST_CMD` | Complete project test command used when a change is too broad to select related tests safely. | Configured in `.beryl/agent/affected-tests.conf`. | Silent skip for broad changes |
| Setup Script | `.beryl/scripts/setup-project.sh` | Interactive onboarding command that installs the agent control plane into a target project. | Must offer deterministic defaults and explicit AI fallback. | Manual multi-file setup checklist |
| AI Agent Fallback | `run_agent_fallback` | Opt-in setup path that hands project-specific configuration to a selected headless coding agent. | Prompt must be user-provided and not stored in tracked files. | Hidden automation |
| Entropy Hotspot | `EntropyHotspot` | High-churn and high-complexity area likely to degrade maintainability. | Used for targeted refactoring. | `MessyFile` |
| Extraction Slice | `ExtractionSlice` | One safe, non-feature refactor step identified after reading changed files from a product run. | Must preserve behavior, avoid new features, name the files involved, and identify the protecting check or missing regression test. | Broad cleanup, opportunistic redesign |
| Vertical Slice | `VerticalSlice` | Smallest end-to-end behavior change through one boundary. | Must be testable in isolation. | `BigRefactor` |
| Adapter | `Adapter` | Boundary object that isolates external systems from domain logic. | Domain must not depend on vendor details. | `ServiceHelper` |
| Episode | `episode` | One reset-to-terminal or reset-to-truncation interaction with a simulator. | Episode boundaries and terminal causes must remain explicit in datasets and evaluation. | `Run` |
| Action chunk | `action_chunk` | A horizon of predicted actions produced or executed as one policy output. | Distinguish chunk length from simulator timestep and control period. | `Batch` |
| Observation horizon | `obs_horizon` | Number of past observations supplied to the policy at one decision. | Must be recorded with the prediction horizon and dataset windowing rule. | `History` |
| Prediction horizon | `pred_horizon` | Number of future actions predicted by a policy output. | Do not confuse it with the number of physics substeps. | `Horizon` |
| Simulator timestep | `sim_dt` | Duration of one physics integration step. | Push-T currently uses 100 Hz physics. | `Control period` |
| Control period | `control_dt` | Time between policy actions sent to the environment. | Push-T currently uses 10 Hz control and 10 physics substeps per action. | `Timestep` |
| Replay from state | `reset_to_state` | Initialize an environment from a recorded simulator state before replaying actions. | Required for meaningful demo-replay diagnostics. | `Reset` |
| Demo replay | `demo_replay` | Re-executing recorded actions from recorded initial states and comparing trajectories or success. | Validate this before diagnosing a learned policy. | `Policy evaluation` |
| Delta action | `delta_action` | Action interpreted as a change relative to the current pose or target. | Must not be silently mixed with absolute actions. | `Residual action` |
| Absolute action | `absolute_action` | Action interpreted as a target pose or end-effector command. | Robomimic low-dimensional data uses absolute end-effector actions. | `Position action` |
| Sim-to-sim gap | `sim_to_sim_gap` | Behavioral difference caused by changing simulator parameters or control settings while replaying the same task. | Record gains, damping, timestep, and control frequency. | `Noise` |
| Sim-to-real gap | `sim_to_real_gap` | Difference between simulated and physical robot behavior. | Includes dynamics, sensing, calibration, timing, and actuation differences. | `Generalization` |
| Seed | `seed` | Randomness control value for training or evaluation. | Report it with the dataset, checkpoint, and method. | `Version` |
| Checkpoint | `checkpoint` | Saved model and optimizer state used to resume or evaluate a policy. | Identify the exact file or immutable artifact. | `Model` |
| Confidence interval | `confidence_interval` | Statistical uncertainty range around an evaluation estimate. | Report with the number of seeds and episodes. | `Error bar` |
| Observation age | `observation_age` | Time between sensing an observation and executing the action derived from it. | Important for streaming and hardware evaluation. | `Latency` |
| Action scheduling | `action_scheduling` | Planning execution times for predicted actions to absorb inference and transport delay. | Keep timing explicit at deployment boundaries. | `Queue` |

## Theory-To-Code Vocabulary

| Concept | Technical symbol | Repository location |
| --- | --- | --- |
| Diffusion Policy | `diffusion_policy` | `Push-T/train/train_dp.py` and `Push-T/inference/` |
| Streaming flow policy | `streaming_flow_policy` | `guided_streaming_policy/core/flow_policy.py` |
| Stochastic interpolant | `gamma_t` | `guided_streaming_policy/core/` interpolation helpers |
| Conditional 1-D U-Net | `ConditionalUnet1D` | `guided_streaming_policy/core/models.py` |
| Critic guidance | `CCG` / `STEG` | `Push-T/core/critic.py` and `Push-T/inference/` |
| Conflict-aware additive guidance | `CAR` | `guided_streaming_policy/core/car_guidance.py` |
