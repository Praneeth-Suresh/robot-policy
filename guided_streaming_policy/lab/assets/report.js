/* Recorded verdicts are immutable. All calculations here are display-only. */
(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("trace-data").textContent);
  const pairs = data.pairs;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const PLOT_WIDTH = 720, PLOT_HEIGHT = 180, PLOT_PAD = 32;
  const MARKER_RADIUS = 4, RADIANS_TO_DEGREES = 180 / Math.PI;
  const EXTENT_PADDING_FRACTION = 0.08, MIN_SPAN = 1, DISPLAY_DECIMALS = 2;
  const select = document.getElementById("pair-select");
  const slider = document.getElementById("step-slider");
  let pair = pairs[0], index = 0;

  function svgNode(tag, attributes = {}, text = null) {
    const node = document.createElementNS(SVG_NS, tag);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== null) node.textContent = text;
    return node;
  }

  function finite(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function point(value) {
    return Array.isArray(value) && value.length === 2 && value.every(finite);
  }

  function stepIndex(run, step) {
    // Hold the last recorded step after a shorter episode ends; never interpolate.
    let found = run.step.findIndex(value => value > step);
    return found < 0 ? run.step.length - 1 : found - 1;
  }

  function currentStep() {
    return timeline()[index];
  }

  function currentTime() {
    const step = currentStep();
    const at = pair.baseline.step.indexOf(step);
    return at >= 0 ? pair.baseline.t[at] : pair.candidate.t[pair.candidate.step.indexOf(step)];
  }

  function timeline() {
    return [...new Set([...pair.baseline.step, ...pair.candidate.step])].sort((a, b) => a - b);
  }

  function fields(run, group) {
    const declaration = group === "obs" ? run.observation_schema : run.action_schema;
    return Object.keys(declaration.fields).sort().filter(name =>
      JSON.stringify(declaration.fields[name].shape) === "[2]");
  }

  function sceneBox() {
    const arena = pair.geometry.arena;
    if (arena && point(arena.size) && arena.size.every(value => value > 0))
      return [0, 0, ...arena.size];
    const points = [];
    [pair.baseline, pair.candidate].forEach(run => {
      ["obs", "action_executed"].forEach(group => fields(run, group).forEach(name =>
        run[group][name].filter(point).forEach(value => points.push(value))));
    });
    (pair.geometry.obstacles || []).forEach(obstacle => {
      const [x, y] = obstacle.position, radius = obstacle.radius;
      points.push([x - radius, y - radius], [x + radius, y + radius]);
    });
    if (!points.length) return [0, 0, MIN_SPAN, MIN_SPAN];
    const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
    const left = Math.min(...xs), top = Math.min(...ys);
    const width = Math.max(MIN_SPAN, Math.max(...xs) - left);
    const height = Math.max(MIN_SPAN, Math.max(...ys) - top);
    const pad = Math.max(width, height) * EXTENT_PADDING_FRACTION;
    return [left - pad, top - pad, width + pad * 2, height + pad * 2];
  }

  function block(svg, pose, className) {
    if (!pose || !pose.every(finite)) return;
    const group = svgNode("g", {class: className,
      transform: `translate(${pose[0]} ${pose[1]}) rotate(${pose[2] * RADIANS_TO_DEGREES})`});
    (pair.geometry.block.shapes || []).forEach(shape =>
      group.append(svgNode("polygon", {points: shape.map(p => p.join(",")).join(" ")})));
    svg.append(group);
  }

  function obstacles(svg) {
    (pair.geometry.obstacles || []).forEach(obstacle => {
      const [cx, cy] = obstacle.position;
      svg.append(svgNode("circle", {cx, cy, r: obstacle.radius, class: "obstacle"}));
      if (pair.separation) {
        const ring = svgNode("circle", {cx, cy, r: pair.separation.threshold, class: "threshold"});
        ring.append(svgNode("title", {}, "separation threshold (proxy)"));
        svg.append(ring);
        svg.append(svgNode("text", {x: cx, y: cy - pair.separation.threshold, class: "scene-label"},
          "separation threshold (proxy)"));
      }
    });
  }

  function polylineSegments(svg, points, className) {
    let segment = [];
    function flush() {
      if (segment.length) svg.append(svgNode("polyline", {
        class: className, points: segment.map(p => p.join(",")).join(" ")}));
      segment = [];
    }
    points.forEach(value => { if (point(value)) segment.push(value); else flush(); });
    flush();
  }

  function scene(side) {
    const run = pair[side], at = stepIndex(run, currentStep());
    const svg = document.getElementById(`${side}-scene`);
    svg.replaceChildren();
    svg.setAttribute("viewBox", sceneBox().join(" "));
    const geometry = pair.geometry;
    if (geometry.block && geometry.goal) block(svg, geometry.goal.pose, "goal");
    obstacles(svg);
    const names = fields(run, "obs");
    names.forEach((name, ordinal) => polylineSegments(svg, run.obs[name].slice(0, at + 1),
      `track ${side} track-${ordinal % 3}`));
    if (geometry.block && geometry.block.pose_fields && at >= 0) {
      const [position, angle] = geometry.block.pose_fields;
      const pos = run.obs[position] && run.obs[position][at];
      const rotation = run.obs[angle] && run.obs[angle][at];
      if (point(pos)) block(svg, [...pos, rotation], `block ${side}`);
    }
    const targetName = fields(run, "action_executed")[0];
    const target = targetName && run.action_executed[targetName][at];
    if (point(target)) svg.append(svgNode("circle", {cx: target[0], cy: target[1],
      r: sceneBox()[2] * EXTENT_PADDING_FRACTION / MARKER_RADIUS, class: `target ${side}`}));
    const spatial = Object.values(run.observation_schema.fields).find(field => field.shape.length === 1);
    const frame = spatial ? `${spatial.frame}, ${spatial.units}; y points down. ` : "y points down. ";
    document.getElementById(`${side}-legend`).textContent = frame + names.map((name, ordinal) =>
      `${name} (${["solid", "short dash", "dot"][ordinal % 3]})`).join(" · ") +
      (targetName ? ` · executed action marker: ${targetName}` : "");
    const outcome = run.outcome;
    const ended = at === run.step.length - 1 && run.step[at] < currentStep();
    document.getElementById(`${side}-outcome`).textContent =
      `native success: ${outcome.native_success}, reason: ${outcome.reason}` +
      (ended ? " (episode ended; showing last recorded step)" : "");
  }

  function signal(run, path) {
    if (!path) return run.t.map(() => null);
    const [group, name] = path.split(".");
    return (run[group] && run[group][name]) || run.t.map(() => null);
  }

  function separation(run) {
    const obstacles = run.geometry.obstacles || [];
    const rule = pair.separation;
    if (!rule || !rule.signal || !obstacles.length) return run.t.map(() => null);
    const [group, name] = rule.signal.split(".");
    const declarations = group === "obs" ? run.observation_schema : run.action_schema;
    const declaration = declarations.fields[name];
    const compatible = declaration && obstacles.every(obstacle =>
      obstacle.units === rule.units && obstacle.frame === declaration.frame);
    return signal(run, rule.signal).map(value => {
      const vector = finite(value) ? [value] : value;
      if (!compatible || !Array.isArray(vector) || !vector.length || !vector.every(finite) ||
          !obstacles.every(obstacle => Array.isArray(obstacle.position) &&
            obstacle.position.length === vector.length && obstacle.position.every(finite)))
        return null;
      return Math.min(...obstacles.map(obstacle => Math.hypot(
        ...vector.map((coordinate, axis) => coordinate - obstacle.position[axis]))));
    });
  }

  function actionSize(run) {
    const names = Object.keys(run.action_executed).sort();
    function flatten(value) { return Array.isArray(value) ? value.flat(Infinity) : [value]; }
    const vectors = run.t.map((_, at) => names.flatMap(name => flatten(run.action_executed[name][at])));
    return vectors.map((vector, at) => {
      if (!at || !vector.length || !vector.every(finite) || !vectors[at - 1].every(finite))
        return null;
      return Math.hypot(...vector.map((value, component) => value - vectors[at - 1][component]));
    });
  }

  function readNumber(value) {
    return finite(value) ? value.toFixed(DISPLAY_DECIMALS) : "not collected";
  }

  function plotScale(series, threshold) {
    const times = [...pair.baseline.t, ...pair.candidate.t];
    const values = series.flat().filter(finite);
    if (finite(threshold)) values.push(threshold);
    const start = Math.min(0, ...times), end = Math.max(start + MIN_SPAN, ...times);
    const low = Math.min(0, ...values), high = Math.max(low + MIN_SPAN, ...values);
    return {x: value => PLOT_PAD + (value - start) / (end - start) * (PLOT_WIDTH - 2 * PLOT_PAD),
      y: value => PLOT_HEIGHT - PLOT_PAD - (value - low) / (high - low) * (PLOT_HEIGHT - 2 * PLOT_PAD),
      low, high, start, end};
  }

  function firstViolation(svg, scale, values) {
    const first = pair.regression && pair.regression.first_violation;
    if (!first) return;
    const at = pair.candidate.step.indexOf(first.step);
    const time = finite(first.time) ? first.time : pair.candidate.t[at];
    if (!finite(time)) return;
    const x = scale.x(time);
    svg.append(svgNode("line", {x1: x, x2: x, y1: PLOT_PAD, y2: PLOT_HEIGHT - PLOT_PAD,
      class: "violation"}));
    if (finite(values[at])) svg.append(svgNode("circle", {cx: x, cy: scale.y(values[at]),
      r: MARKER_RADIUS, class: "violation-ring"}));
    svg.append(svgNode("text", {x: Math.min(x, PLOT_WIDTH - 150), y: PLOT_PAD / 2,
      class: "violation-label"}, "first violation"));
  }

  function plot(title, series, threshold = null, markViolation = false) {
    const section = document.createElement("section");
    const heading = document.createElement("h3");
    heading.textContent = title;
    const scale = plotScale(series, threshold);
    const svg = svgNode("svg", {viewBox: `0 0 ${PLOT_WIDTH} ${PLOT_HEIGHT}`, role: "img",
      "aria-label": `${title}; baseline solid, candidate dashed`, class: "signal-plot"});
    svg.append(svgNode("path", {d: `M${PLOT_PAD} ${PLOT_PAD}V${PLOT_HEIGHT - PLOT_PAD}H${PLOT_WIDTH - PLOT_PAD}`,
      class: "axis"}));
    ["baseline", "candidate"].forEach((side, line) => {
      const points = series[line].map((value, at) => finite(value) ?
        [scale.x(pair[side].t[at]), scale.y(value)] : null);
      polylineSegments(svg, points, `signal ${side}`);
    });
    if (finite(threshold)) svg.append(svgNode("line", {x1: PLOT_PAD, x2: PLOT_WIDTH - PLOT_PAD,
      y1: scale.y(threshold), y2: scale.y(threshold), class: "threshold"}));
    const time = currentTime();
    if (finite(time)) svg.append(svgNode("line", {x1: scale.x(time), x2: scale.x(time), y1: PLOT_PAD,
      y2: PLOT_HEIGHT - PLOT_PAD, class: "cursor"}));
    svg.append(svgNode("text", {x: 0, y: PLOT_PAD}, readNumber(scale.high)));
    svg.append(svgNode("text", {x: 0, y: PLOT_HEIGHT - PLOT_PAD}, readNumber(scale.low)));
    svg.append(svgNode("text", {x: PLOT_PAD, y: PLOT_HEIGHT - 4}, `${readNumber(scale.start)} ${pair.time_unit}`));
    svg.append(svgNode("text", {x: PLOT_WIDTH - 90, y: PLOT_HEIGHT - 4},
      `${readNumber(scale.end)} ${pair.time_unit}`));
    if (markViolation) firstViolation(svg, scale, series[1]);
    const legend = document.createElement("p");
    legend.textContent = ["baseline", "candidate"].map((side, line) =>
      `${side} (${line ? "dashed" : "solid"}): ` +
      readNumber(series[line][stepIndex(pair[side], currentStep())])).join(" · ") +
      (finite(threshold) ? ` · threshold ${readNumber(threshold)}` : "");
    section.append(heading, svg, legend);
    document.getElementById("signal-plots").append(section);
  }

  function plots() {
    document.getElementById("signal-plots").replaceChildren();
    const rule = pair.separation;
    if (rule) {
      const detail = ` (${rule.signal}; ${rule.units}; tolerance ${readNumber(rule.tolerance)})`;
      plot("Separation proxy" + detail, [separation(pair.baseline), separation(pair.candidate)],
        rule.threshold, true);
    }
    const actionUnits = Object.values(pair.baseline.action_schema.fields).map(field => field.units);
    plot(`Action step size (${[...new Set(actionUnits)].join(", ")})`,
      [actionSize(pair.baseline), actionSize(pair.candidate)]);
    if ([...pair.baseline.reward, ...pair.candidate.reward].some(finite))
      plot("Native reward per step", [pair.baseline.reward, pair.candidate.reward]);
  }

  function updateHash() {
    const hash = `#pair=${encodeURIComponent(pair.key).replace(/%3A/gi, ":")}&step=${currentStep() ?? 0}`;
    history.replaceState(null, "", hash);
  }

  function render(writeHash = false) {
    slider.max = Math.max(0, timeline().length - 1);
    slider.value = index;
    const time = currentTime();
    const empty = !timeline().length;
    document.getElementById("step-readout").textContent = empty ? "No recorded steps." :
      `step ${currentStep()}, t = ${readNumber(time)} ${pair.time_unit}`;
    slider.disabled = empty;
    document.getElementById("previous").disabled = empty || index === 0;
    document.getElementById("next").disabled = empty || index >= Number(slider.max);
    ["baseline", "candidate"].forEach(scene);
    plots();
    document.querySelectorAll("[data-media-pair]").forEach(node => {
      node.hidden = node.dataset.mediaPair !== pair.key;
      if (node.hidden) node.querySelector("video").pause();
    });
    if (writeHash) updateHash();
  }

  function navigate() {
    const params = new URLSearchParams(location.hash.slice(1));
    const first = data.regressions[0];
    const defaultKey = first ? `${first.key.condition}:${first.key.seed}` : pairs[0].key;
    pair = pairs.find(item => item.key === params.get("pair")) ||
      pairs.find(item => item.key === defaultKey) || pairs[0];
    const requested = params.has("step") ? Number(params.get("step")) :
      (!location.hash && first ? first.replay_step : 0);
    const step = finite(requested) ? Math.max(0, Math.floor(requested)) : 0;
    index = Math.max(0, timeline().filter(value => value <= step).length - 1);
    select.value = pair.key;
    render();
  }

  if (!pairs.length) {
    document.getElementById("replay-controls").after(document.createTextNode("No completed pairs to replay."));
    return;
  }
  document.getElementById("replay-controls").hidden = false;
  select.addEventListener("change", () => { pair = pairs.find(item => item.key === select.value); index = 0; render(true); });
  slider.addEventListener("input", () => { index = Number(slider.value); render(true); });
  document.getElementById("previous").addEventListener("click", () => { index = Math.max(0, index - 1); render(true); });
  document.getElementById("next").addEventListener("click", () => { index = Math.min(Number(slider.max), index + 1); render(true); });
  window.addEventListener("hashchange", navigate);
  navigate();
})();
