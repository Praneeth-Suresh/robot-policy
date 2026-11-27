"""Offline report evidence, escaping, replay data, and deterministic output."""

import json
import math
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from guided_streaming_policy.lab import schema
from tests.lab.helpers import report_pipeline

__all__: list[str] = []

_GOLDEN = Path(__file__).parents[1] / "fixtures/lab/golden/report_regression.html"
_DEEP_LINK_STEP = 3
_CHROME = Path("/usr/bin/google-chrome")


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def render(self, verdict: str = "REGRESSION", *, gate: bool = True) -> str:
        from guided_streaming_policy.lab import report

        comparison, result = report_pipeline(self.root, verdict)
        return report.render_report(
            report.load_result(comparison),
            report.load_result(result, gate_result=True) if gate else None,
            base_dir=comparison.parent,
        )

    def test_report_evidence_and_embedded_traces(self) -> None:
        page = self.render()
        for text in (
            "Policy Trace Lab report: toy-comparison",
            "wordmark",
            "REGRESSION (exit 1)",
            "Task success",
            "Constraint compliance",
            "Coverage",
            "expected: 1",
            'href="#pair=static:1&step=3"',
            "first violation",
            "separation proxy",
            "Synthetic fixture data. Not a benchmark.",
        ):
            self.assertIn(text, page)
        self.assertEqual(page.count('data-rule="'), 2)
        for word, icon in (("Pass", "check"), ("Fail", "cross"), ("Unknown", "question")):
            self.assertRegex(page, rf'<svg[^>]+data-icon="{icon}".*?</svg> {word}')
        data = json.loads(re.search(r'id="trace-data">(.*?)</script>', page, re.S)[1])
        pair = data["pairs"][0]
        self.assertEqual(pair["key"], "static:1")
        self.assertEqual(pair["separation"]["signal"], "action_executed.target")
        self.assertNotIn("display fallback", page)
        self.assertEqual(pair["candidate"]["action_executed"]["target"][3], [15, 10])
        self.assertEqual(pair["baseline"]["action_executed"]["target"][3], [70, 10])
        self.assertEqual(pair["geometry"]["block"]["pose_fields"], ["block_pos", "block_angle"])
        self.assertNotRegex(page, r"pusht|Push-T|contact")
        self.assertNotRegex(page, r"""(?:src|href)=["']https?://|url\(["']?https?://""")
        self.assertNotRegex(page.replace("http://www.w3.org/2000/svg", ""), r"https?://")

    def test_pass_blocked_and_no_gate(self) -> None:
        for verdict, word, code in (("PASS", "Pass", 0), ("BLOCKED", "Unknown", 2)):
            with tempfile.TemporaryDirectory() as directory:
                from guided_streaming_policy.lab import report

                compare, gate = report_pipeline(Path(directory), verdict)
                page = report.render_report(
                    report.load_result(compare),
                    report.load_result(gate, gate_result=True),
                    base_dir=compare.parent,
                )
                self.assertIn(f"{verdict} (exit {code})", page)
                self.assertIn(f"</svg> {word}", page)
        page = self.render(gate=False)
        self.assertIn("No gate result supplied", page)
        self.assertIn("Gate not supplied.", page)
        self.assertIn("Native success", page)

    def test_escaping_determinism_and_golden(self) -> None:
        from guided_streaming_policy.lab import report

        page = self.render()
        compare_path = self.root / "compare/compare.json"
        gate = report.load_result(self.root / "compare/gate.json", gate_result=True)
        compare = report.load_result(compare_path)
        self.assertEqual(page, report.render_report(compare, gate, base_dir=compare_path.parent))
        self.assertEqual(page.encode(), _GOLDEN.read_bytes())
        compare["comparison"]["name"] = '<script>alert("x")</script>'
        hostile = report.render_report(compare, None, base_dir=compare_path.parent)
        self.assertIn("&lt;script&gt;", hostile)
        self.assertNotIn('<script>alert("x")</script>', hostile)
        path = self.root / "candidate/episode/steps.jsonl"
        path.write_text(path.read_text().replace('"reward":0.3', '"reward":0.333333'))
        page = report.render_report(compare, None, base_dir=compare_path.parent)
        self.assertIn("0.333", page)
        self.assertNotIn("0.333333", page)

    def test_invalid_results_manifests_and_traces(self) -> None:
        from guided_streaming_policy.lab import report

        compare, gate = report_pipeline(self.root)
        for path in (
            compare,
            gate,
            self.root / "baseline/run_manifest.json",
            self.root / "candidate/episode/steps.jsonl",
        ):
            original = path.read_bytes()
            path.write_text("{}")
            with self.subTest(path=path), self.assertRaises(schema.TraceReadError):
                report.render_report(
                    report.load_result(compare),
                    report.load_result(gate, gate_result=True),
                    base_dir=compare.parent,
                )
            path.write_bytes(original)

    def test_missing_pairs_telemetry_media_and_script_terminators(self) -> None:
        from guided_streaming_policy.lab import report

        comparison, _ = report_pipeline(self.root)
        compare = report.load_result(comparison)
        manifest_path = self.root / "candidate/episode/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(synthetic=False, media=[{"kind": "video", "uri": "clip&one.mp4"}])
        manifest_path.write_text(schema.canonical_json(manifest))
        page = report.render_report(compare, None, base_dir=comparison.parent)
        self.assertIn('src="clip&amp;one.mp4"', page)
        manifest["media"][0]["uri"] = '</script><script>alert("x")</script>'
        manifest_path.write_text(schema.canonical_json(manifest))
        page = report.render_report(compare, None, base_dir=comparison.parent)
        embedded = re.search(r'id="trace-data">(.*?)</script>', page, re.S)[1]
        self.assertEqual(json.loads(embedded)["pairs"][0]["candidate"]["media"], manifest["media"])
        self.assertNotIn("</script><script>alert", page)
        compare["pairs"][0].update(status="missing", candidate=None, reasons=["missing candidate"])
        compare["coverage"].update(completed=0, missing=1)
        page = report.render_report(compare, None, base_dir=comparison.parent)
        self.assertIn("missing candidate", page)

    def test_invalid_gate_comparison_and_trace_shapes(self) -> None:
        from guided_streaming_policy.lab import report

        comparison, gate_path = report_pipeline(self.root)
        compare, gate = (
            report.load_result(comparison),
            report.load_result(gate_path, gate_result=True),
        )
        gate["exit_code"] = True
        with self.assertRaises(schema.TraceReadError):
            report.render_report(compare, gate, base_dir=comparison.parent)
        gate["exit_code"] = 1
        gate["comparison"]["sha256"] = "different"
        with self.assertRaises(schema.TraceReadError):
            report.render_report(compare, gate, base_dir=comparison.parent)
        path = self.root / "candidate/episode/steps.jsonl"
        steps = [json.loads(line) for line in path.read_text().splitlines()]
        steps[0]["obs"]["agent_pos"] = [1, 2, 3]
        path.write_text(
            "".join(schema.canonical_json(step).replace("\n", "") + "\n" for step in steps)
        )
        with self.assertRaises(schema.TraceReadError):
            report.render_report(compare, None, base_dir=comparison.parent)

    def browser_dom(self, page_text: str) -> str:
        page = self.root / "report.html"
        page.write_text(page_text)
        result = subprocess.run(
            [
                str(_CHROME),
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                f"--user-data-dir={self.root / 'chrome-profile'}",
                "--dump-dom",
                "--virtual-time-budget=2000",
                page.as_uri() + "#pair=static:1&step=3",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return result.stdout

    def test_results_display_two_decimals(self) -> None:
        from guided_streaming_policy.lab import report

        comparison, gate_path = report_pipeline(self.root)
        compare = report.load_result(comparison)
        gate = report.load_result(gate_path, gate_result=True)
        gate["rules"][1].update(baseline_value=44.514561843168565, candidate_value=None)
        page = report.render_report(compare, gate, base_dir=comparison.parent)
        table = re.search(r"<tbody>(.*?)</tbody>", page, re.S)[1]
        self.assertIn("baseline min 44.51; candidate min None; threshold 30.72 ± 0.00 px", table)
        self.assertIn("1.00 → 1.00; delta 0.00", table)
        self.assertIn("rule coverage 1.00", table)
        self.assertNotIn("44.514561843168565", table)

    def test_results_display_counts_as_integers(self) -> None:
        from guided_streaming_policy.lab import report

        comparison, gate_path = report_pipeline(self.root)
        compare = report.load_result(comparison)
        gate = report.load_result(gate_path, gate_result=True)
        gate["rules"][0].update(samples=4, wins=0, losses=1, ties=3)
        page = report.render_report(compare, gate, base_dir=comparison.parent)
        table = re.search(r"<tbody>(.*?)</tbody>", page, re.S)[1]
        self.assertIn("samples 4; wins 0; losses 1; ties 3</td>", table)
        for count in ("samples 4", "wins 0", "losses 1", "ties 3"):
            self.assertNotIn(f"{count}.00", table)
        self.assertIn("step 3</a>", page)
        self.assertNotIn("step 3.00", page)

    def render_toy(self) -> tuple[str, dict[str, float]]:
        """Render the scaffolded one-dimensional rail pack's seeded regression."""
        from guided_streaming_policy.lab import cli_tasks, gate, pairing, report

        pack = cli_tasks.scaffold_pack("rail", self.root / "pack")
        scenario = pack / "rail/fixtures/scenarios/seeded_regression"
        comparison = pairing.build_comparison(
            scenario / "baseline",
            scenario / "candidate",
            schema.load_comparison_config(scenario / "comparison.json"),
            out_dir=self.root,
        )
        result = gate.evaluate_gate(
            comparison, schema.load_gate_config(scenario / "gate.json"), base_dir=self.root
        )
        # browser_dom opens "#pair=static:1&step=3"; an unknown pair selects the first one.
        step, seed = _DEEP_LINK_STEP, comparison["pairs"][0]["key"]["seed"]
        wall = 0.55
        distances = {}
        for side in ("baseline", "candidate"):
            episode = schema.read_episode(scenario / side / f"nominal__seed{seed}")
            distances[side] = abs(episode.steps[step]["obs"]["x"][0] - wall)
        return report.render_report(comparison, result, base_dir=self.root), distances

    @unittest.skipUnless(_CHROME.is_file(), "google-chrome is not installed")
    def test_browser_toy_separation_vectors_and_scalars(self) -> None:
        page, distances = self.render_toy()
        payload = re.search(r'id="trace-data">(.*?)</script>', page, re.S)[1]
        for scalar, extra in ((False, []), (True, []), (False, [0.3, 0.4])):
            with self.subTest(scalar=scalar, extra=extra):
                data = json.loads(payload)
                for pair in data["pairs"]:
                    for side in ("baseline", "candidate"):
                        run = pair[side]
                        run["observation_schema"]["fields"]["x"]["shape"] = (
                            [] if scalar else [1 + len(extra)]
                        )
                        run["obs"]["x"] = [
                            value[0] if scalar else value + [0.0] * len(extra)
                            for value in run["obs"]["x"]
                        ]
                        for obstacle in run["geometry"]["obstacles"]:
                            obstacle["position"] += extra
                dom = self.browser_dom(page.replace(payload, json.dumps(data)))
                text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", dom, flags=re.S)
                text = re.sub(r"<[^>]+>", "", text)
                offset = math.hypot(*extra) if extra else 0.0
                baseline = f"{math.hypot(distances['baseline'], offset):.2f}"
                candidate = f"{math.hypot(distances['candidate'], offset):.2f}"
                self.assertNotIn("not collected", text)
                self.assertIn(
                    f"baseline (solid): {baseline} · candidate (dashed): {candidate}"
                    " · threshold 0.10",
                    text,
                )

    @unittest.skipUnless(_CHROME.is_file(), "google-chrome is not installed")
    def test_browser_deep_link(self) -> None:
        dom = self.browser_dom(self.render())
        self.assertRegex(dom, r'id="step-readout"[^>]*>step 3, t = 0.30 s')
        self.assertIn("baseline (solid): 60.00 · candidate (dashed): 5.00 · threshold 30.72", dom)
        self.assertIn("baseline (solid): 0.30 · candidate (dashed): 0.30", dom)

    @unittest.skipUnless(_CHROME.is_file(), "google-chrome is not installed")
    def test_browser_separation_uses_configured_observation_signal(self) -> None:
        from guided_streaming_policy.lab import report

        comparison, gate_path = report_pipeline(self.root)
        gate = report.load_result(gate_path, gate_result=True)
        gate["rules"][1]["signal"] = "obs.agent_pos"
        page = report.render_report(
            report.load_result(comparison), gate, base_dir=comparison.parent
        )
        dom = self.browser_dom(page)
        self.assertIn("Separation proxy (obs.agent_pos; px; tolerance 0.00)", dom)
        self.assertIn("baseline (solid): 9.22 · candidate (dashed): 9.22 · threshold 30.72", dom)

    @unittest.skipUnless(_CHROME.is_file(), "google-chrome is not installed")
    def test_browser_omits_separation_without_rule_or_gate(self) -> None:
        from guided_streaming_policy.lab import report

        comparison, gate_path = report_pipeline(self.root)
        compare = report.load_result(comparison)
        gate = report.load_result(gate_path, gate_result=True)
        gate["rules"] = gate["rules"][:1]
        for result in (gate, None):
            with self.subTest(gate=result):
                page = report.render_report(compare, result, base_dir=comparison.parent)
                dom = self.browser_dom(page)
                headings = re.findall(r"<h3>(.*?)</h3>", dom)
                self.assertFalse(any("Separation proxy" in heading for heading in headings))
                self.assertIn("Action step size (px)", headings)
