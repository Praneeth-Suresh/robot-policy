"""Offline presentation of recorded comparison/gate evidence; plots never decide verdicts.

Separation plots use only the signal recorded in gate rule evidence.
Trace hashes are deliberately not checked here. Display telemetry is rounded to
REPORT_DECIMALS; gate and comparison evidence remains unchanged. Assets ship as
package data. Only schema contracts cross the presentation boundary.
"""

from __future__ import annotations

import base64
import html
import json
import math
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from guided_streaming_policy.lab import schema

__all__ = ["DISPLAY_DECIMALS", "REPORT_DECIMALS", "load_result", "render_report"]

REPORT_DECIMALS = 3
DISPLAY_DECIMALS = 2
_HASH_PREFIX_CHARACTERS = 12
_VARIANTS = ("baseline", "candidate")
_STATUSES = {"PASS": "pass", "REGRESSION": "fail", "BLOCKED": "unknown"}
_MARKS = {"pass": ("Pass", "check", "M3 8l3 3 7-7"),
          "fail": ("Fail", "cross", "M4 4l8 8M12 4l-8 8"),
          "unknown": ("Unknown", "question", "M5 5a3 3 0 116 0c0 2-3 2-3 4M8 12v1")}


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _valid(value: Any, spec: dict[str, Any], root: dict[str, Any]) -> bool:
    if "$ref" in spec:
        return _valid(value, root["$defs"][spec["$ref"].split("/")[-1]], root)
    if not _type_valid(value, spec.get("type", [])):
        return False
    if "const" in spec and not _same(value, spec["const"]):
        return False
    if "enum" in spec and not any(_same(value, item) for item in spec["enum"]):
        return False
    return _branches(value, spec, root) and _contents(value, spec, root)


def _type_valid(value: Any, kinds: str | list[str]) -> bool:
    types: dict[str, type[Any] | tuple[type[Any], ...]] = {
        "object": dict, "array": list, "string": str, "integer": int,
        "number": (int, float), "boolean": bool, "null": type(None)}
    kinds = [kinds] if isinstance(kinds, str) else kinds
    return not kinds or any(isinstance(value, types[k]) and not (
        k in ("integer", "number") and isinstance(value, bool)) for k in kinds)


def _same(value: Any, expected: Any) -> bool:
    return bool(value == expected and isinstance(value, bool) == isinstance(expected, bool))


def _branches(value: Any, spec: dict[str, Any], root: dict[str, Any]) -> bool:
    for key, predicate in (("anyOf", any), ("allOf", all)):
        if key in spec and not predicate(_valid(value, item, root) for item in spec[key]):
            return False
    if "not" in spec and _valid(value, spec["not"], root):
        return False
    if "if" in spec:
        branch = "then" if _valid(value, spec["if"], root) else "else"
        return _valid(value, spec.get(branch, {}), root)
    return True


def _contents(value: Any, spec: dict[str, Any], root: dict[str, Any]) -> bool:
    if isinstance(value, dict):
        return _object_valid(value, spec, root)
    if isinstance(value, list):
        return (len(value) >= spec.get("minItems", 0)
                and (not spec.get("uniqueItems") or len({schema.canonical_json(v) for v in value})
                     == len(value)) and all(_valid(v, spec.get("items", {}), root) for v in value))
    if type(value) in (int, float):
        return (math.isfinite(value) and spec.get("minimum", -math.inf) <= value
                <= spec.get("maximum", math.inf)
                and value > spec.get("exclusiveMinimum", -math.inf))
    return True


def _object_valid(value: dict[str, Any], spec: dict[str, Any], root: dict[str, Any]) -> bool:
    if set(spec.get("required", [])) - value.keys():
        return False
    properties = spec.get("properties", {})
    extra = spec.get("additionalProperties", {})
    for key, item in value.items():
        if key not in properties and extra is False:
            return False
        child = properties.get(key, extra if isinstance(extra, dict) else {})
        if not _valid(item, child, root):
            return False
    return True


def _validate(value: Any, definition: dict[str, Any]) -> None:
    schema.canonical_json(value)
    if not _valid(value, definition, definition):
        raise ValueError("document does not match its schema contract")


def load_result(path: Path, *, gate_result: bool = False) -> dict[str, Any]:
    """Load a recorded result with standard-library validation, without checking hashes."""
    try:
        value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        definition = (schema.gate_result_json_schema() if gate_result
                      else schema.compare_result_json_schema())
        _validate(value, definition)
        return value
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
        raise schema.TraceReadError(f"Cannot read result {path}: {error}") from error


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, REPORT_DECIMALS)
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    return value


def _value_shape(value: Any, shape: list[int]) -> bool:
    if not shape:
        return type(value) in (int, float) and math.isfinite(value)
    return (isinstance(value, list) and len(value) == shape[0]
            and all(_value_shape(item, shape[1:]) for item in value))


def _episode(path: Path) -> dict[str, Any]:
    episode = schema.read_episode(path)
    manifest = episode.manifest
    _validate(manifest, schema.manifest_json_schema())
    _geometry_valid(manifest.get("geometry", {}))
    fields = {"obs": manifest["observation_schema"]["fields"],
              "action_executed": manifest["action_schema"]["fields"]}
    for step in episode.steps:
        _validate(step, schema.step_json_schema())
        for group, declarations in fields.items():
            for name, value in step.get(group, {}).items():
                if name not in declarations or not _value_shape(value, declarations[name]["shape"]):
                    raise ValueError(f"invalid {group}.{name} at {path}")
    numbers = [step["step"] for step in episode.steps]
    times = [step["t"] for step in episode.steps]
    if numbers != sorted(set(numbers)) or times != sorted(set(times)):
        raise ValueError(f"trace steps and times must increase at {path}")
    if manifest["outcome"]["length"] != len(episode.steps):
        raise ValueError(f"outcome length differs from trace at {path}")
    data: dict[str, Any] = {group: {name: [step.get(group, {}).get(name) for step in episode.steps]
                    for name in declarations} for group, declarations in fields.items()}
    data.update({key: [step.get(key) for step in episode.steps] for key in ("step", "t", "reward")})
    data.update({key: manifest.get(key, default) for key, default in (
        ("outcome", {}), ("media", []), ("geometry", {}), ("synthetic", False),
        ("timebase", {}), ("observation_schema", {}), ("action_schema", {}))})
    return data


def _block_valid(block: dict[str, Any]) -> None:
    shapes = block.get("shapes", [])
    if any(not isinstance(shape, list) or not shape
           or not all(_value_shape(point, [2]) for point in shape) for shape in shapes):
        raise ValueError("invalid block shapes")
    pose_fields = block.get("pose_fields", [])
    if pose_fields and (not isinstance(pose_fields, list) or len(pose_fields) != 2
                        or not all(isinstance(v, str) for v in pose_fields)):
        raise ValueError("invalid block pose_fields")


def _geometry_valid(geometry: dict[str, Any]) -> None:
    _block_valid(geometry.get("block", {}))
    goal = geometry.get("goal", {})
    if "pose" in goal and not _value_shape(goal["pose"], [3]):
        raise ValueError("invalid goal pose")
    for obstacle in geometry.get("obstacles", []):
        position = obstacle["position"]
        if (not isinstance(position, list) or not position
                or not _value_shape(position, [len(position)])):
            raise ValueError("invalid obstacle position")
    arena = geometry.get("arena", {})
    if "size" in arena and (not _value_shape(arena["size"], [2])
                            or any(value <= 0 for value in arena["size"])):
        raise ValueError("invalid arena size")


def _key(pair: dict[str, Any]) -> str:
    key = pair["key"]
    return f"{key['condition']}:{key['seed']}"


def _trace_data(compare: dict[str, Any], gate: dict[str, Any] | None, base: Path) -> dict[str, Any]:
    regressions = {_key(item): item for item in gate["regressions"]} if gate else {}
    separation = next((rule for rule in gate["rules"] if rule["metric"] ==
                       "obstacle_separation_min"), None) if gate else None
    pairs = []
    synthetic = False
    for pair in compare["pairs"]:
        sides = {side: _episode(base / pair[side]["path"]) for side in _VARIANTS
                 if pair[side] is not None}
        synthetic = synthetic or any(side["synthetic"] for side in sides.values())
        if pair["status"] != "completed":
            continue
        first = sides["baseline"]
        geometry = first["geometry"]
        if any(len(item["position"]) != 2 for item in geometry.get("obstacles", [])):
            geometry = {**geometry, "obstacles": [item for item in geometry["obstacles"]
                                                 if len(item["position"]) == 2]}
        pairs.append({"key": _key(pair), **sides, "geometry": geometry,
                      "time_unit": first["timebase"]["unit"],
                      "regression": regressions.get(_key(pair)),
                      "separation": separation})
    return {"pairs": pairs, "regressions": list(regressions.values()), "synthetic": synthetic}


def _asset(name: str) -> str:
    return resources.files("guided_streaming_policy.lab").joinpath("assets").joinpath(name)\
        .read_text("utf-8")


def _status(status: str) -> str:
    word, icon, path = _MARKS[status]
    return (f'<span class="status {status}"><svg data-icon="{icon}" viewBox="0 0 16 16" '
            f'aria-hidden="true"><path d="{path}"/></svg> {word}</span>')


def _identity(label: str, identity: dict[str, Any]) -> str:
    return (f'<p>{label}: {_e(identity["name"])} '
            f'<code>{_e(identity["sha256"][:_HASH_PREFIX_CHARACTERS])}</code></p>')


def _header(compare: dict[str, Any], gate: dict[str, Any] | None, commit: str,
            synthetic: bool) -> str:
    logos = ''.join(f'<div class="wordmark {mode}">{_asset(filename)}</div>' for mode, filename in (
        ("light-logo", "logo-wordmark.svg"), ("dark-logo", "logo-wordmark-dark.svg")))
    title = _e("Policy Trace Lab report: " + compare["comparison"]["name"])
    verdict = (_identity("Gate", gate["gate"]) + _status(_STATUSES[gate["verdict"]])
               + f' <strong>{_e(gate["verdict"])} (exit {_e(gate["exit_code"])})</strong>'
               if gate else '<p>No gate result supplied</p>')
    warning = '<p class="fixture">Synthetic fixture data. Not a benchmark.</p>' if synthetic else ''
    return (f'<header>{logos}<h1>{title}</h1><p>Validate robot data. Compare policies. '
            'Replay regressions.</p>' + _identity("Comparison", compare["comparison"])
            + f'<p>Baseline git commit: <code>{_e(commit)}</code></p>{verdict}{warning}</header>')


def _coverage(compare: dict[str, Any], gate: dict[str, Any] | None) -> str:
    counts = ' · '.join(f'{key}: {_e(compare["coverage"][key])}' for key in (
        "expected", "completed", "incompatible", "missing"))
    limits = ("<p>Gate limits: " + ' · '.join(f'{_e(key)}: {_e(gate["coverage"][key])}' for key in (
        "min_completed_pairs", "max_missing_pairs", "max_incompatible_pairs"))
              + "</p>") if gate else ''
    missing = ''.join(f'<li>{_e(_key(pair))}: {_e(pair["status"])}; '
                      f'{_e("; ".join(pair["reasons"]))}</li>' for pair in compare["pairs"]
                      if pair["status"] != "completed")
    blocked = (''.join(f'<li>{_e(reason)}</li>' for reason in gate["blocked_reasons"])
               if gate else '')
    return f'<section><h2>Coverage</h2><p>{counts}</p>{limits}<ul>{missing}{blocked}</ul></section>'


def _display(value: Any) -> str:
    return _e(f"{value:.{DISPLAY_DECIMALS}f}" if type(value) in (int, float) else value)


def _delta(rule: dict[str, Any]) -> str:
    return (f'{_display(rule["baseline_value"])} → {_display(rule["candidate_value"])}; '
            f'delta {_display(rule["delta"])}')


def _rule_row(rule: dict[str, Any]) -> str:
    success = rule["metric"] == "native_success_rate"
    metric = "native success rate" if success else "separation proxy"
    compliance = ('n/a' if success else f'baseline min {_display(rule["baseline_value"])}; '
                  f'candidate min {_display(rule["candidate_value"])}; '
                  f'threshold {_display(rule["threshold"])} ± {_display(rule["tolerance"])} '
                  f'{_e(rule["units"])}; {_status(rule["verdict"])}')
    coverage = f'rule coverage {_display(rule.get("coverage", "n/a"))}; ' + '; '.join(
        f'{key} {_display(rule[key])}' for key in ("samples", "wins", "losses", "ties"))
    return (f'<tr data-rule="{_e(rule["id"])}"><th scope="row">{_e(rule["id"])}<br>{metric}'
            f'<br>{_e(rule["units"])}</th><td>{_status(rule["verdict"])}</td>'
            f'<td>{_delta(rule) if success else "n/a"}</td><td>{compliance}</td>'
            f'<td>{coverage}</td></tr>')


def _results(compare: dict[str, Any], gate: dict[str, Any] | None) -> str:
    if gate is None:
        delta = compare["deltas"]["native_success"]
        return ('<section><h2>Results</h2><p>Native success: '
                + _delta({"baseline_value": delta["baseline_rate"],
                          "candidate_value": delta["candidate_rate"], "delta": delta["delta"]})
                + '</p></section>')
    headers = ''.join(f'<th scope="col">{name}</th>' for name in (
        "Rule", "Verdict", "Task success", "Constraint compliance", "Coverage"))
    legend = '<p>Status key: ' + ' · '.join(_status(key) for key in _MARKS) + '</p>'
    return ('<section><h2>Results</h2>' + legend + '<div class="table-scroll"><table>'
            f'<caption>Recorded gate rule evidence</caption><thead><tr>{headers}</tr></thead>'
            f'<tbody>{"".join(_rule_row(rule) for rule in gate["rules"])}</tbody>'
            '</table></div></section>')


def _regressions(gate: dict[str, Any] | None, data: dict[str, Any]) -> str:
    if not gate or not gate["regressions"]:
        text = "No regressing pairs." if gate else "Gate not supplied."
        return f'<section><h2>Regressions</h2><p>{text}</p></section>'
    units = {pair["key"]: pair["time_unit"] for pair in data["pairs"]}
    rows = []
    for item in gate["regressions"]:
        key, first = _key(item), item["first_violation"]
        when = (f'first violation: {_e(first["time"])} {_e(units.get(key, ""))}, '
                f'step {_e(first["step"])}') if first else "episode end"
        anchor = f'#pair={quote(key, safe=":")}&step={item["replay_step"]}'
        rows.append(f'<li>{_e(key)}: {_status("fail")} {_e(item["category"])}; {when}; '
                    f'rules {_e(", ".join(item["rule_ids"]))}; '
                    f'<a href="{anchor}">Replay {_e(key)} step {_e(item["replay_step"])}</a></li>')
    return '<section><h2>Regressions</h2><ol>' + ''.join(rows) + '</ol></section>'


def _video_uri(uri: str) -> str:
    parsed = urlsplit(uri)
    if parsed.scheme not in ("", "file") or parsed.netloc or "\\" in uri:
        raise ValueError("video URI must be an offline relative or absolute path")
    return _e(uri)


def _replay(data: dict[str, Any]) -> str:
    options = ''.join(f'<option value="{_e(pair["key"])}">{_e(pair["key"])}'
                      f'{" (regressing)" if pair["regression"] else ""}</option>'
                      for pair in data["pairs"])
    panels = ''.join(f'<article><h3>{side.title()}</h3><svg id="{side}-scene" '
                      f'class="scene" role="img" aria-label="{side.title()} replay"></svg>'
                      f'<p id="{side}-legend"></p><p id="{side}-outcome"></p></article>'
                      for side in _VARIANTS)
    media = ''.join(f'<div data-media-pair="{_e(pair["key"])}" hidden><p>{side.title()} video</p>'
                     f'<video controls preload="none" src="{_video_uri(item["uri"])}">'
                     '</video></div>'
                     for pair in data["pairs"] for side in _VARIANTS
                     for item in pair[side]["media"] if item["kind"] == "video")
    return ('<section><h2>Replay</h2><p>Paired runs started from the same reset. '
            'Display numbers are for reading only; verdicts come from recorded results.</p>'
            '<noscript>Replay needs JavaScript. Recorded results above remain readable.</noscript>'
            '<div id="replay-controls" hidden><label for="pair-select">Pair</label> '
            f'<select id="pair-select">{options}</select><div class="panels">{panels}</div>'
            '<div class="step-controls"><button id="previous" type="button">Previous step</button>'
            '<label for="step-slider">Replay step</label><input id="step-slider" type="range" '
            'min="0" max="0" value="0" step="1"><button id="next" type="button">Next step</button>'
            '<output id="step-readout" for="step-slider" aria-live="polite"></output></div>'
            '<div id="signal-plots"></div>' + media + '</div></section>')


def _comparison_valid(compare: dict[str, Any]) -> None:
    pairs, counts = compare["pairs"], compare["coverage"]
    for status in ("completed", "incompatible", "missing"):
        if counts[status] != sum(pair["status"] == status for pair in pairs):
            raise ValueError("comparison coverage does not match pairs")
    if counts["expected"] != len(pairs) or len({_key(pair) for pair in pairs}) != len(pairs):
        raise ValueError("comparison pair keys or expected count invalid")
    if any(pair[side] is None for pair in pairs if pair["status"] == "completed"
           for side in _VARIANTS):
        raise ValueError("completed pair lacks an episode")


def render_report(compare: dict[str, Any], gate: dict[str, Any] | None, *, base_dir: Path) -> str:
    """Render validated evidence and trace files to deterministic, self-contained HTML."""
    try:
        _validate(compare, schema.compare_result_json_schema())
        _comparison_valid(compare)
        if gate is not None:
            _validate(gate, schema.gate_result_json_schema())
            if any(gate["comparison"][key] != compare["comparison"][key]
                   for key in ("name", "sha256")):
                raise ValueError("gate result belongs to another comparison")
        runs = [schema.read_run_manifest(base_dir / compare["runs"][side]["path"])
                for side in _VARIANTS]
        for run in runs:
            _validate(run, schema.run_manifest_json_schema())
        data = _trace_data(compare, gate, base_dir)
        header = _header(compare, gate, runs[0]["provenance"]["git_commit"], data["synthetic"])
        body = (header + _coverage(compare, gate) + _results(compare, gate)
                + _regressions(gate, data))
        body += _replay(data)
        payload = schema.canonical_json(_rounded(data)).replace('</', '<\\/')
        favicon = base64.b64encode(_asset("favicon.svg").encode()).decode("ascii")
        title = _e("Policy Trace Lab report: " + compare["comparison"]["name"])
        return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                f'<title>{title}</title><link rel="icon" '
                f'href="data:image/svg+xml;base64,{favicon}">'
                f'<style>{_asset("report.css")}</style></head><body><main>{body}</main>'
                f'<script type="application/json" id="trace-data">{payload}</script>'
                f'<script>{_asset("report.js")}</script></body></html>\n')
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, IndexError) as error:
        raise schema.TraceReadError(f"Cannot render report: {error}") from error
