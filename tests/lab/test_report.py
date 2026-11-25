"""Offline report evidence, escaping, replay data, and deterministic output."""

import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from guided_streaming_policy.lab import schema
from tests.lab.helpers import report_pipeline

__all__: list[str] = []

_GOLDEN = Path(__file__).parents[1] / "fixtures/lab/golden/report_regression.html"
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
        self.assertIn("display fallback", pair["separation"]["signal_source"])
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

    @unittest.skipUnless(_CHROME.is_file(), "google-chrome is not installed")
    def test_browser_deep_link(self) -> None:
        page = self.root / "report.html"
        page.write_text(self.render())
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
        self.assertRegex(result.stdout, r'id="step-readout"[^>]*>step 3, t = 0.3 s')
