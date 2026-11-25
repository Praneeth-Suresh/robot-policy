# Decisions

Assumptions, alternatives considered, the brand rationale and the removal manifest for Policy Trace Lab. Newest decisions of a section are at the bottom.

## Validation run choice (recorded before the run)

Recorded before any candidate result was inspected, together with the committed configs `lab/comparisons/pusht_guidance_v1.json` and `lab/gates/pusht_v1.json`.

| Item | Choice | Reason |
| --- | --- | --- |
| Baseline | `pusht/dp` with its registered defaults (`guidance_scale` 2.0) | The diffusion policy with inference-time guidance is the repository's closest match to a standard diffusion-policy baseline. |
| Candidate | `pusht/dp` with `guidance_scale` set to 0.0 | One declared difference, no code change: guidance off. It tests whether the declared rule notices the loss of guidance. |
| Condition | `static` (obstacle at normalized `[-0.3, -0.5]`) | The only condition in this release. |
| Seeds | 0, 1, 2, 3, 4 | Explicit list. About 275 s per dp trial on this CPU (measured: seed 0, `Push-T/run_trial.py`), so 10 trials are about 46 minutes. |
| Gate | `success_not_worse` with threshold -0.1; `separation_proxy` threshold 30.72 px, tolerance 0, minimum coverage 0.95; 5 completed pairs required, no missing or incompatible pairs allowed | 30.72 px is the runners' own collision distance (`2 * 0.06 * 256`). The success tolerance of one tenth is set once here and not tuned after seeing results. |
| One-seed export of every policy | seed 0, `static`, default settings, one comparison config per policy (see `reproducibility.md`) | Section 7.2 of the build prompt. |

The verdict is whatever the declared rule produces. Nothing is tuned afterwards.

## Design decisions

| Topic | Decision | Alternatives considered |
| --- | --- | --- |
| Config format | JSON, not YAML | PyYAML would add a runtime dependency to the default install. The standard library has no YAML parser. |
| Package layout | `guided_streaming_policy/lab/` (path A) | A sibling package: a second install surface. |
| `schema` module | A package (`schema/{traces,jsonschemas,config,results}`) | One file: it passed the 400-line limit once the config and result schemas were added. |
| Import rules | The build prompt's table, with these amendments: every core module may import `tasks.base` (the contract); `exporter` and `tasks/<task>` may import `adapters`; only `cli` and `tasks.registry` refer to a task package | `exporter` needs the shared config resolution in `adapters`; a second copy would duplicate logic. |
| How runs are recorded | The exporter replaces the module-global `env` of the runner with a recording proxy and calls the unchanged runner. The third episode of each call (the guided one) is kept | Editing the runners would change research code and risk the reference policies. Cost: each call also runs two episodes that are discarded. |
| Step convention | Step `i` holds the observation before action `i`, the executed action target, and the reward returned by that step | Recording the final observation needs a step without an action. |
| `truncated` | True when the runner's 250-step cap ended the trial without environment termination | The environment returns `terminated == truncated`, so copying it would carry no information. |
| Separation proxy | Minimum distance between a recorded position signal named in the gate rule and the obstacle centre, in simulator pixels. The committed gate uses the executed action target; a supplementary gate uses `obs.agent_pos`. The runners' own `guided_min_dist` is measured on the action target for `naive`, `ccg`, `steg` and on the agent position for `dp`, `flow`; the lab reproduces each exactly (see `real-run.md`) | The obstacle is a point in normalized action space and is not in the physics, so there is no contact to measure. It is called a proxy everywhere. The first reading of the runners covered only `inference_naive.py`; the real run exposed the difference for `dp` and `flow`. The pre-declared gate was kept as the primary result, and the supplementary gate was declared after seeing it. |
| Native success | Copied from the runner (`guided_success`) and never recomputed. A crashed trial counts as not successful in rates and is listed as `crashed` | Dropping crashed trials would hide failures. |
| Pair with no reset fingerprint | `incompatible` ("reset state not recorded") | Treating a missing fingerprint as a match would make the check unverifiable. |
| Gate evidence | The separation rule is judged on candidate episodes; baseline values are reported but do not affect the verdict. A rule with both a failing and an unknown episode is `fail`. Exit 2 (blocked) is checked before exit 1 | Judging both sides would blame the candidate for baseline behaviour. |
| Integrity | `compare.json` stores sha256 of every episode's `manifest.json` and `steps.jsonl`; the gate refuses traces that changed after the comparison | A self-contained compare file would duplicate all traces. |
| Optional schema fields | `geometry`, `synthetic`, `outcome.native`, `outcome.detail` | Needed by the separation check, the report and the real-run cross-check. They are additive and optional; `schema_version` stays `0.1` because the schema had not been published. |
| Checkpoint identity | `checkpoint_sha256` in a manifest is the sha256 of the sorted `{path: sha256}` map of the policy's checkpoint files, taken from `lab/pins.json` | A policy can use two checkpoint files (`pusht/ccg`). |
| Report assets | `lab/assets/` holds byte copies of three brand SVGs; a test checks they equal `docs/brand/` | Reading `docs/brand/` at run time would break pip installs. |
| Dev tools | `ruff`, `mypy` (strict, `lab` package only), `vulture`, `jsonschema`, in an optional `dev` extra | The repository had none. They are not installed by default. |
| CI | One new workflow, `.github/workflows/lab.yml`; the repository had no CI | No existing check changed. |
| Tracking | The session task list tracked T1 to T15. Beryl has no task tracker (`hierarchy.md` is only for initial builds); only durable decisions go in `.beryl/agent/` files | Adding a progress file would break the Beryl rule against progress notes in canonical files. |
| `NOTICE` | Left unchanged. The build prompt asks to cite each paper in `NOTICE` and also never to change it; the stricter rule wins. Papers are cited in the README ("Play with the reference policies" and attribution) and in the policy registry | Appending to `NOTICE` would change a legal file. |
| Task order | The brand kit (T12) was built before the report (T9) | The report takes its colours from `tokens.json`. |

## Brand rationale

Design skill: `frontend-design` from <https://github.com/anthropics/skills> (Apache-2.0), fetched from the upstream repository for this work because none was installed. Its process was followed: a plan with palette, type, layout and principles, then a review against the common templated defaults, then the build, then screenshots and critique.

- Idea: traceability. Two lines share a path and split at a red ring: the point where a paired run goes wrong. The same device is the mark, the banner, the pipeline diagram and the cursor in the replay plots.
- Palette: cool trace-paper grey, deep ink navy, blue for baseline, purple for candidate, and green, red and slate for pass, fail and unknown. It avoids the cream-and-terracotta and black-and-acid-green defaults the skill warns about. Every text pair is checked for WCAG AA.
- Type: system stacks only, so reports work offline. A serif display face for headings and a sans body, clearly distinct.
- Wordmark: drawn as strokes, so it does not depend on a font. The dot of the "i" is the marker.
- Limits: the text inside `pipeline*.svg` uses system fonts and can differ between platforms. The 1280 x 640 social preview was rendered with Chrome from the banner and system text.

## Removal manifest

Every pre-existing file in the starting commit `6a0b8ad` has one action. Nothing has been deleted: rows marked **proposed** wait for the maintainer's approval (`AGENTS.md` requires an itemised list before any root documentation is deleted or consolidated). If you approve none of them, the tree stays valid.

### Moved, rewritten or changed

| Path | Action | Reason |
| --- | --- | --- |
| `README.md` | rewritten | Policy Trace Lab leads. Kept and moved under "Research workspace": try without downloads, CAR smoke test, setup and assets, run trials, add a new flow policy, development checks. Dropped: the "What is here" intro table (replaced by the new introduction) and the "Research ideas to compare" list (prose with no code or test behind it; git history keeps it). |
| `guided_streaming_policy/cli.py` | changed | Wires the additive `lab` group; existing subcommands and output unchanged. |
| `pyproject.toml` | changed | Adds the `dev` extra, ruff, mypy and vulture settings, and package data for report assets. `dependencies` stays empty. |
| `uv.lock` | changed | Resolves the new `dev` extra; no existing package version changed. |
| `.gitignore` | changed | Adds `out/` (lab outputs). |
| `.beryl/agent/architecture.md`, `ubiquitous-language.md`, `testing-policy.md` | changed | Durable decisions, terms and the new lint, type and dead-code commands, as `AGENTS.md` requires. |
| `.beryl/agent/project-brief.md` | changed (only if `curriculum.md` is removed) | Drops the `curriculum.md` mention in the documentation row. |

### Removed (proposed)

| Path | Action | Reason | What is lost | Where it lives |
| --- | --- | --- | --- | --- |
| `curriculum.md` | remove | A personal study plan. No code, test, command or lab document uses it, so it does not contribute to the lab. | The reading list and build tasks for learning the methods. | Git history (`6a0b8ad`). The paper references the lab needs are in the README and the policy registry. |
| `Robomimic/si_can_results/streaming_si_ode.gif`, `Robomimic/si_can_results/streaming_si_sde.gif` | remove | Generated result files committed by mistake; the repository's own rule is that result directories are gitignored (`.gitignore` ignores `Robomimic/si_*_results/`). Nothing references them. | Two sample GIFs. | Git history. |

### Kept, with the reason

| Paths | Reason |
| --- | --- |
| `LICENSE`, `NOTICE` | Legal; unchanged. |
| `Push-T/**` (research code, README, `.gitignore`), `guided_streaming_policy/core/**`, `guided_streaming_policy/pusht/**` | The reference policies and the runners the lab wraps. Unchanged. |
| `Robomimic/**` (except the two GIFs), `guided_streaming_policy/robomimic_legacy.py` | Retained research workflow and the next lab task. Unchanged. |
| `guided_streaming_policy/{__init__,__main__,doctor,paths,pusht_legacy}.py`, `environment.yml` | Used by `gsp check-env`, `gsp run` and the documented Conda fallback. |
| `docs/reproducibility.md` | The research reproducibility notes, checked by `tests/test_release_structure.py` and linked from `Push-T/README.md`. The lab's own page is `docs/lab/reproducibility.md`. |
| `tests/test_*.py` (8 files) | Existing tests for code that remains. None deleted or edited. |
| `AGENTS.md`, `CLAUDE.md`, `.codex/`, `.cursor/`, `.github/copilot-instructions.md`, `.beryl/**` (other than the three files above) | Beryl-generated agent control plane; not edited by hand. |
