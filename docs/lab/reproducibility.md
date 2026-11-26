# Reproducibility

Output of the same input is byte-identical: result files use sorted keys, fixed indentation and no timestamps, and the checked-in goldens in `tests/fixtures/lab/golden/` pin that down.

## Pinned versions

| Item | Value |
| --- | --- |
| Python (project environment) | 3.10 |
| Simulator | pymunk 7.2.0 (`pyproject.toml`, `lab/pins.json`, comparison config `simulator_version`) |
| Other pinned packages | the `pusht` extra in `pyproject.toml` (torch 2.9.1, numpy 1.26.4, zarr 2.18.3, gym 0.26.2, diffusers 0.36.0) and `uv.lock` |
| Checkpoints | sha256 values in `lab/pins.json`; `gsp lab policies` prints their prefixes and a test fails if any pin changes |
| Dev tools | `ruff`, `mypy`, `vulture`, `jsonschema` in the `dev` extra |
| Trace schema | `0.1` |

Changing a pin is a deliberate, reviewed edit of `lab/pins.json`. To recompute it after a deliberate change:

```bash
.venv/bin/python -c "from pathlib import Path; from guided_streaming_policy.lab import schema; from guided_streaming_policy.lab.tasks.pusht.pins import pin_file_document; Path('lab/pins.json').write_text(schema.canonical_json(pin_file_document(Path('.').resolve())), encoding='utf-8')"
```

## Seeds

Every seed is listed explicitly in a comparison config. The seeds of the validation run are `0, 1, 2, 3, 4` (`lab/comparisons/pusht_guidance_v1.json`). The runner seeds `torch`, `numpy` and `random` with the trial seed and resets the environment from `RandomState(seed)`; two runs with the same seed give the same reset fingerprint (a test checks it).

## Regenerating committed artifacts

| Artifact | How | Drift check |
| --- | --- | --- |
| `schemas/*.schema.json` | `.venv/bin/python -m guided_streaming_policy.lab.schema schemas` | `tests/lab/test_schema.py` |
| `tests/fixtures/lab/pusht/` | `.venv/bin/python -m tests.lab.fixture_builders <empty dir>`, then replace the committed tree | `tests/lab/test_fixtures.py` |
| `tests/fixtures/lab/golden/*` | produced by the builders in the test that names the file; the test says how if the file is missing | the same tests |
| `lab/pins.json` | the command above | `tests/lab/test_tasks.py` |
| `guided_streaming_policy/lab/assets/*.svg` | copy the same-named file from `docs/brand/` | `tests/lab/test_report_tokens.py` |
| `docs/brand/*.svg`, `social-preview.png` | designed by hand; `tokens.json` is the source of truth for colour, and a test checks the report CSS against it | `tests/lab/test_report_tokens.py` |
| `docs/img/*.png` | see below | `tests/lab/test_docs.py` (size, alt text, caption) |
| Real-run results | see [real run](real-run.md) | the manifests committed there |

## Capturing the screenshots

All screenshots are rendered by headless Chrome from real command output or the real `report.html`, then optimised to 64-colour PNG files (each under 300 KB, under 2 MB in total). Install Google Chrome, then:

CLI output as an image (the same text is pasted next to it in the README):

```bash
cli_shot() {  # usage: cli_shot output.txt out.png
  local txt=$1 out=$2 lines html
  lines=$(wc -l < "$txt"); html=$(mktemp --suffix=.html)
  { printf '%s' '<!doctype html><meta charset="utf-8"><body style="margin:0;background:#0E1726"><pre style="margin:0;padding:24px;color:#E8EEF6;font:15px/22px ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap;word-break:break-word">'
    sed 's/&/\&amp;/g;s/</\&lt;/g;s/>/\&gt;/g' "$txt"; printf '%s' '</pre>'; } > "$html"
  google-chrome --headless=new --no-sandbox --disable-gpu --hide-scrollbars \
    --window-size=1100,$((64 + 22 * (lines + 1))) --screenshot="$out" "file://$html"
}
```

The report, opened at a pair and step (the fragment is the same deep link the report's regression list uses):

```bash
google-chrome --headless=new --no-sandbox --disable-gpu --hide-scrollbars \
  --window-size=1280,2900 --virtual-time-budget=3000 --screenshot=report-full.png \
  "file://$PWD/out/real/report/report.html#pair=static:0&step=4"
```

Optimise and crop (the crop box is `left,top,right,bottom` in pixels):

```bash
.venv/bin/python - report-full.png docs/img/real-7-replay.png 0,1500,1280,2700 <<'PY'
import sys
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
box = tuple(int(v) for v in sys.argv[3].split(","))
Image.open(src).convert("RGB").crop(box).quantize(colors=64).save(dst, optimize=True)
PY
```

The brand banner and social preview were rendered the same way from the SVG files and system text.

## Running the checks

```bash
python -m unittest discover -s tests -v          # offline; simulator tests skip without assets
python -m ruff check guided_streaming_policy/lab tests/lab
python -m mypy
python -m vulture
GSP_LAB_REAL_RUN=1 python -m unittest tests.lab.test_pusht_runner   # opt-in: runs one real trial (about 30 s)
```

CI (`.github/workflows/lab.yml`) runs everything except the opt-in real trial, then `gsp lab demo` with an assertion on its exit code and on the gate and report fields.
