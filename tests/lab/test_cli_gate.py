"""Gate CLI with real on-disk comparisons and output preservation."""

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
from tests.lab.helpers import gate_comparison, gate_config, gate_episodes, register_fake_task

__all__: list[str] = []


class GateCLITests(unittest.TestCase):
    def setUp(self) -> None:
        register_fake_task(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.compare_path = self.root / "compare/compare.json"
        self.config_path = self.root / "gate-config.json"
        self.out = self.root / "gate"
        gate_config(self.config_path)

    def write_comparison(self, *, regression: bool = False) -> None:
        episodes = gate_episodes()
        if regression:
            episodes[1].steps[2]["action_executed"]["target"] = [10, 0]
        compare = gate_comparison(self.root, candidate=episodes)
        self.compare_path.parent.mkdir(exist_ok=True)
        self.compare_path.write_text(schema.canonical_json(compare))

    def invoke(
        self, args: list[str] | None = None, *, default_out: bool = False
    ) -> tuple[int, str, str]:
        output, error = io.StringIO(), io.StringIO()
        if args is None:
            args = [str(self.compare_path), "--gate", str(self.config_path)]
            if not default_out:
                args += ["--out", str(self.out)]
        with redirect_stdout(output), redirect_stderr(error):
            try:
                code = main(["lab", "gate", *args])
            except SystemExit as caught:
                code = int(caught.code)
        return code, output.getvalue(), error.getvalue()

    def test_pass_writes_canonical_report_and_repeat_is_identical(self) -> None:
        self.write_comparison()
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = self.invoke()
        self.assertEqual(result[0], 0, result[2])
        self.assertIn("Policy Trace Lab gate: PASS (exit 0)", result[1])
        self.assertIn(f"gate.json: {self.out / 'gate.json'}", result[1])
        self.assertEqual(result[2], "")
        report = self.out / "gate.json"
        self.assertEqual(report.read_text(), schema.canonical_json(json.loads(report.read_text())))
        self.assertEqual(self.invoke(), result)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_regression_returns_one(self) -> None:
        self.write_comparison(regression=True)
        code, output, error = self.invoke()
        self.assertEqual(code, 1, error)
        self.assertIn("Policy Trace Lab gate: REGRESSION (exit 1)", output)
        self.assertEqual(json.loads((self.out / "gate.json").read_text())["exit_code"], 1)

    def test_blocked_and_unreadable_inputs_return_two(self) -> None:
        self.assertEqual(self.invoke()[0], 2)
        self.write_comparison()
        self.compare_path.write_text("{}")
        self.assertEqual(self.invoke()[0], 2)
        self.compare_path.write_text('{"schema_version":"0.1"}')
        code, output, _ = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("BLOCKED", output)
        self.config_path.write_text("{")
        self.assertEqual(self.invoke()[0], 2)

    def test_usage_and_help(self) -> None:
        self.assertEqual(self.invoke([])[0], 3)
        self.assertEqual(self.invoke(["--unknown"])[0], 3)
        code, output, _ = self.invoke(["--help"])
        self.assertEqual(code, 0)
        output = " ".join(output.split())
        for text in (
            "Policy Trace Lab",
            "0 pass",
            "1 regression",
            "2 invalid, incomplete or a required check unknown",
            "3 usage or internal error",
        ):
            self.assertIn(text, output)

    def test_default_out_and_refuse_different_report(self) -> None:
        self.write_comparison()
        with patch.dict(os.environ), tempfile.TemporaryDirectory() as temporary:
            previous = Path.cwd()
            try:
                os.chdir(temporary)
                self.assertEqual(self.invoke(default_out=True)[0], 0)
                self.assertTrue(Path("out/gate/toy-gate/gate.json").is_file())
            finally:
                os.chdir(previous)
        self.out.mkdir()
        path = self.out / "gate.json"
        path.write_text("preserve me\n")
        code, _, error = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("refusing to overwrite", error)
        self.assertEqual(path.read_text(), "preserve me\n")
