"""Compare command artifacts, summary, and lab exit conventions."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from guided_streaming_policy.cli import main
from guided_streaming_policy.lab import schema
from tests.lab.helpers import (
    comparison_config,
    gate_config,
    gate_document,
    geometry_rollout,
    rollout_run,
)

__all__: list[str] = []


class CompareCLITests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "comparison.json"
        self.config = comparison_config(self.path, seeds=(0,))
        self.baseline = rollout_run(self.root / "baseline", "baseline", self.config,
                                    [geometry_rollout("a")])
        self.candidate = rollout_run(self.root / "candidate", "candidate", self.config,
                                     [geometry_rollout("a")])
        self.out = self.root / "compare"

    def invoke(self, *, out: bool = True) -> tuple[int, str, str]:
        output, error = io.StringIO(), io.StringIO()
        args = ["lab", "compare", str(self.baseline), str(self.candidate),
                "--comparison", str(self.path)]
        if out:
            args += ["--out", str(self.out)]
        with redirect_stdout(output), redirect_stderr(error):
            code = main(args)
        return code, output.getvalue(), error.getvalue()

    def test_valid_comparison_writes_only_inside_out(self) -> None:
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        code, output, error = self.invoke()
        self.assertEqual(code, 0, error)
        self.assertEqual(error, "")
        self.assertIn("completed=1", output)
        self.assertIn(f"compare.json: {self.out / 'compare.json'}", output)
        result = json.loads((self.out / "compare.json").read_text())
        self.assertTrue(result["valid"])
        self.assertEqual((self.out / "compare.json").read_text(), schema.canonical_json(result))
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(set(p for p in self.root.rglob("*") if p.is_file()) - before.keys(),
                         {self.out / "compare.json"})
        self.assertEqual(self.invoke(), (code, output, error))

    def test_invalid_comparison_still_writes_report_and_stderr_reasons(self) -> None:
        path = self.candidate / "run_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["comparison"]["sha256"] = "wrong"
        path.write_text(schema.canonical_json(manifest))
        code, output, error = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("candidate run comparison hash differs", error)
        self.assertIn("valid: false", output)
        self.assertFalse(json.loads((self.out / "compare.json").read_text())["valid"])

    def test_incomplete_comparison_prints_pair_reasons_and_exits_zero(self) -> None:
        self.config = comparison_config(self.path, seeds=(0, 1, 2))
        self.baseline = rollout_run(
            self.root / "mixed-baseline", "baseline", self.config,
            [geometry_rollout(str(seed), seed=seed) for seed in (0, 1, 2)],
        )
        incompatible = geometry_rollout("1", seed=1)
        incompatible.manifest["reset_state_fingerprint"] = "different"
        self.candidate = rollout_run(
            self.root / "mixed-candidate", "candidate", self.config,
            [geometry_rollout("0"), incompatible],
        )
        code, output, error = self.invoke()
        self.assertEqual(code, 0, error)
        self.assertEqual(error, "")
        self.assertIn("valid: true", output)
        self.assertIn("incompatible [c1 seed=1]: reset state fingerprints differ", output)
        self.assertIn("missing [c1 seed=2]: missing episode in candidate run", output)
        self.assertTrue(json.loads((self.out / "compare.json").read_text())["valid"])

    def test_unintended_config_difference_blocks_compare_and_gate(self) -> None:
        self.candidate = rollout_run(
            self.root / "unintended-candidate", "candidate", self.config,
            [geometry_rollout("a")], resolved_config={
                "policy": "toy/a", "entry_point": "toy:run",
                "settings": {"gain": 2.0, "other": 2},
            },
        )
        reason = "unintended configuration difference: settings.other"
        code, output, error = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("valid: false", output)
        self.assertIn(reason, error)
        comparison = json.loads((self.out / "compare.json").read_text())
        self.assertEqual(comparison["invalid_reasons"], [reason])
        self.assertEqual(comparison["coverage"]["completed"], 1)
        document = gate_document()
        document["min_completed_pairs"] = 1
        path = self.root / "gate-config.json"
        gate_config(path, document)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = main(["lab", "gate", str(self.out / "compare.json"),
                         "--gate", str(path), "--out", str(self.out)])
        self.assertEqual(code, 2)
        result = json.loads((self.out / "gate.json").read_text())
        self.assertEqual(result["blocked_reasons"], [reason])

    def test_unreadable_run_is_two(self) -> None:
        self.candidate = self.root / "absent"
        code, output, error = self.invoke()
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertIn("run manifest", error)
        self.assertFalse(self.out.exists())

    def test_refuses_different_report(self) -> None:
        self.out.mkdir()
        target = self.out / "compare.json"
        target.write_text("earlier result")
        code, _, error = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("compare.json", error)
        self.assertEqual(target.read_text(), "earlier result")

    def test_invalid_config_is_two(self) -> None:
        self.path.write_text("invalid")
        code, _, error = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("comparison", error)

    def test_default_output(self) -> None:
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            code, output, error = self.invoke(out=False)
        finally:
            os.chdir(previous)
        self.assertEqual(code, 0, error)
        expected = Path("out/compare") / self.config.name / "compare.json"
        self.assertIn(f"compare.json: {expected}", output)
        self.assertTrue((self.root / expected).is_file())

    def test_usage_help_and_internal_error(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["lab", "compare"])
        self.assertEqual(caught.exception.code, 3)
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as caught:
            main(["lab", "compare", "--help"])
        self.assertEqual(caught.exception.code, 0)
        text = " ".join(output.getvalue().split())
        for phrase in ("Policy Trace Lab", "Exit codes:", "0 comparable",
                       "2 invalid comparison or unreadable input", "3 usage or internal error"):
            self.assertIn(phrase, text)
        with patch("guided_streaming_policy.lab.cli.pairing.build_comparison",
                   side_effect=RuntimeError("unexpected")):
            code, _, error = self.invoke()
        self.assertEqual(code, 3)
        self.assertIn("unexpected", error)
