# Reproducibility and Compatibility

This repository preserves the benchmark behavior used to generate existing
Push-T results. Packaging, documentation, and safety improvements are intended
to make the code easier to install and inspect, without silently changing the
training or inference behavior.

## Preserved Behavior

- Existing checkpoint names, default guidance scales, random seeds, obstacle
  presets, rollout lengths, and output directories are preserved.
- Benchmark-compatible guidance calibration in inference scripts is preserved.
  Alternative calibration methods should be introduced as explicit new options.
- The checkpoint serialization order used by benchmark training scripts is
  preserved.
- Some scripts keep legacy import paths such as `from core...` to ensure that
  the original Push-T workflow remains runnable.

## Safe Cleanup Areas

The following changes are considered safe when they do not alter training or
inference math:

- Documentation, README links, and artifact instructions.
- CLI wrappers that call the same benchmark modules.
- Import-safety changes, such as preventing long-running experiments from
  starting during module import.
- Tests for packaging, task registries, path checks, and small pure functions.

## Rule of Thumb

If a change could alter a tensor value, checkpoint, random draw order, action,
reward, or metric, keep the original behavior as the default. Introduce the
change behind a new option, or validate it with an artifact-backed regression
test before replacing existing behavior.
