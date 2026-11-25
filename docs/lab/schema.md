# Trace schema (version 0.1)

The JSON Schemas in [`schemas/`](../../schemas/) are generated from `guided_streaming_policy/lab/schema/`. Regenerate them with:

```bash
.venv/bin/python -m guided_streaming_policy.lab.schema schemas
```

A test fails if the committed files differ from the generated ones, and another checks that every schema property is described on this page. Any breaking change bumps `schema_version`.

## Layout

One episode is a directory with two files. A run directory holds `run_manifest.json`, `resolved_config.json` and one directory per episode.

```text
run/
  run_manifest.json
  resolved_config.json
  static__0/
    manifest.json
    steps.jsonl
  static__1/
    ...
```

## `manifest.json` ([episode_manifest.schema.json](../../schemas/episode_manifest.schema.json))

"B" means required for both kinds, "D" for `demonstration` only, "R" for `rollout` only.

| Field | Required | Meaning |
| --- | --- | --- |
| `schema_version` | B | `"0.1"` |
| `kind` | B | `demonstration` or `rollout` |
| `episode_id` | B | Stable ID in this trace; also the directory name |
| `source_episode_id` | B | ID in the original source |
| `dataset` (`name`, `revision`, `checksum`) | D | Exactly which data this came from |
| `task` | B | Task name |
| `split` | B | Split label or `null` |
| `split_source` | B | Where the split came from, or `none` |
| `timebase` (`unit`, `mode`, `dt`) | B | `mode` is `fixed_dt` (then `dt` is required) or `timestamps` |
| `observation_schema` (`fields`) | B | Per field: `shape`, `dtype`, `units`, `frame` |
| `action_schema` (`convention`, `fields`) | B | `convention` is `absolute` or `delta`; fields as above |
| `fields_present` | B | Which optional step fields were recorded: `obs`, `action_executed`, `action_proposed`, `reward`, `info`, `reset_state_fingerprint`, `geometry` |
| `provenance` (`tool_version`, `git_commit`, `command`, `created_at`) | B | How this file was produced |
| `media` | optional | List of `{kind, uri}` references, never embedded data |
| `synthetic` | optional | `true` marks fixture data |
| `condition` | R | The matrix cell's condition |
| `seed` | R | The matrix cell's seed |
| `reset_state_fingerprint` | R | Hash of the actual reset state, or `null` when the simulator exposes none |
| `policy` (`name`, `checkpoint_id`, `checkpoint_sha256`) | R | Policy identity. `checkpoint_sha256` is the sha256 of the sorted `{path: sha256}` map of the policy's checkpoint files from `lab/pins.json` |
| `resolved_config` (`path`, `sha256`) | R | Path (relative to the run directory) and sha256 of the compact sorted-key JSON of the fully resolved configuration |
| `simulator` (`name`, `version`) | R | Simulator identity |
| `outcome` | R | `native_success`, `terminated`, `truncated`, `reason`, `length`; optional `native` (the runner's own metrics, verbatim) and `detail` (error text of a crashed trial) |
| `geometry` | optional | `obstacles` (each with `id`, `kind`, `position`, `radius`, `units`, `frame`, `source`) plus optional `goal`, `block`, `arena` objects |

A field missing from `fields_present` means "not collected". It is never "corrupt". Telemetry is never invented or interpolated.

## `steps.jsonl` ([episode_step.schema.json](../../schemas/episode_step.schema.json))

One JSON object per line.

| Field | Required | Meaning |
| --- | --- | --- |
| `step` | yes | Step index starting at 0 |
| `t` | yes | Time in the manifest's timebase unit |
| `obs` | when collected | Named observation fields, before the action of this step |
| `action_executed` | when collected | Named action fields actually applied at this step |
| `action_proposed` | only when the runner exposes both | The policy's proposal before post-processing |
| `reward` | only when the runner provides it | Reward returned by this step |
| `info` | only when the runner provides it | A small subset of the runner's info |

Numbers may be `NaN` or infinity in a corrupt trace, which Doctor reports. The writer records them as they are.

## `run_manifest.json` ([run_manifest.schema.json](../../schemas/run_manifest.schema.json))

`schema_version`, `run_id`, `task`, `comparison` (`name`, `sha256` of the comparison file), `variant`, `policy`, `simulator`, `provenance`, `episodes` (a list of `episode_id`, `condition`, `seed`), and an optional `synthetic` flag.

## Result files

| File | Schema | Written by |
| --- | --- | --- |
| `doctor.json` | [doctor_report.schema.json](../../schemas/doctor_report.schema.json) | `gsp lab doctor` |
| `compare.json` | [compare_result.schema.json](../../schemas/compare_result.schema.json) | `gsp lab compare` |
| `gate.json` | [gate_result.schema.json](../../schemas/gate_result.schema.json) | `gsp lab gate` |

Config files you write. The comparison config has `schema_version`, `name`, `task`, `simulator_version`, `matrix`, `variants` and `intended_differences`. The gate config has `schema_version`, `name`, `min_completed_pairs`, `max_missing_pairs`, `max_incompatible_pairs` and `rules`. Every field is described in the README's configuration reference.

| File | Schema |
| --- | --- |
| Comparison config | [comparison_config.schema.json](../../schemas/comparison_config.schema.json) |
| Gate config | [gate_config.schema.json](../../schemas/gate_config.schema.json) |

All result files use sorted keys, two-space indentation and a trailing newline, and contain no timestamps, so the same input gives byte-identical output.

The top-level properties of the result files are `findings`, `input`, `summary` (doctor); `valid`, `invalid_reasons`, `comparison`, `runs`, `coverage`, `pairs`, `deltas` (compare); `verdict`, `exit_code`, `blocked_reasons`, `gate`, `rules`, `regressions` (gate). The field-by-field reference for the configs is in the README.
