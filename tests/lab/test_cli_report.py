"""Report CLI exit conventions and output preservation, using full recorded runs."""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from guided_streaming_policy.cli import main
from tests.lab.helpers import report_pipeline

__all__: list[str] = []


class ReportCLITests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.compare, self.gate = report_pipeline(self.root)
        self.out = self.root / "html"

    def invoke(self, args: list[str]) -> tuple[int, str, str]:
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            try:
                code = main(["lab", "report", *args])
            except SystemExit as caught:
                code = int(caught.code)
        return code, output.getvalue(), error.getvalue()

    def test_report_written_only_inside_out_and_identical_rerun(self) -> None:
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        args = [str(self.compare), "--gate", str(self.gate), "--out", str(self.out)]
        first = self.invoke(args)
        self.assertEqual(first[0], 0, first[2])
        self.assertEqual(first[1], f"report.html: {self.out / 'report.html'}\n")
        self.assertEqual(self.invoke(args), first)
        after = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(set(after) - set(before), {self.out / "report.html"})
        self.assertEqual(before, {path: after[path] for path in before})

    def test_invalid_input_overwrite_refusal_and_usage(self) -> None:
        self.assertEqual(self.invoke([str(self.root / "absent")])[0], 2)
        self.assertEqual(self.invoke([])[0], 3)
        args = [str(self.compare), "--gate", str(self.root / "absent")]
        self.assertEqual(self.invoke(args)[0], 2)
        self.out.mkdir()
        path = self.out / "report.html"
        path.write_text("preserve me")
        result = self.invoke([str(self.compare), "--out", str(self.out)])
        self.assertEqual(result[0], 2, result[2])
        self.assertEqual(path.read_text(), "preserve me")
        self.assertIn("refusing to overwrite", result[2])

    def test_help_and_default_out(self) -> None:
        code, output, _ = self.invoke(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("Policy Trace Lab", output)
        self.assertIn("Exit codes:", output)
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            result = self.invoke([str(self.compare)])
            self.assertEqual(result[0], 0, result[2])
            self.assertTrue(Path("out/report/toy-comparison/report.html").is_file())
        finally:
            os.chdir(previous)
