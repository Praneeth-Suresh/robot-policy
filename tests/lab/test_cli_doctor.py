"""Doctor command input selection, outputs and documented exit codes."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from guided_streaming_policy.cli import main
from guided_streaming_policy.lab import schema
from tests.lab.helpers import demo_episode, write_traces

__all__: list[str] = []

_REAL_DEMOS = Path(__file__).parents[2] / "Push-T/data/pusht_cchi_v7_replay.zarr.zip"


class DoctorCLITests(unittest.TestCase):
    def invoke(self, *args: str) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["lab", "doctor", *args])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_clean_trace_writes_only_in_out_and_is_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            traces, out = root / "traces", root / "reports"
            write_traces(traces, [demo_episode("clean")])
            before = sorted(path.relative_to(traces) for path in traces.rglob("*"))
            code, output, error = self.invoke(str(traces), "--out", str(out))
            self.assertEqual(code, 0, error)
            self.assertIn(f"doctor.json: {out / 'doctor.json'}", output)
            first_bytes = (out / "doctor.json").read_bytes()
            self.assertEqual(json.loads(first_bytes)["summary"]["counts"]["error"], 0)
            self.assertEqual(self.invoke(str(traces), "--out", str(out))[0], 0)
            self.assertEqual((out / "doctor.json").read_bytes(), first_bytes)
            self.assertEqual(sorted(path.relative_to(traces) for path in traces.rglob("*")), before)
            self.assertEqual(list(root.rglob("doctor.json")), [out / "doctor.json"])

    def test_corrupt_trace_exit_one_and_summary_check_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode = demo_episode("corrupt")
            episode.steps[2]["obs"]["agent_pos"] = [float("nan")]
            write_traces(root / "traces", [episode])
            code, output, error = self.invoke(str(root / "traces"), "--out", str(root / "out"))
            self.assertEqual(code, 1, error)
            self.assertIn("shape.mismatch", output)
            self.assertIn("value.non_finite", output)
            self.assertTrue((root / "out/doctor.json").is_file())

    def test_missing_path_exit_two_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            code, _, error = self.invoke(str(root / "missing"), "--out", str(root / "out"))
            self.assertEqual(code, 2)
            self.assertIn("missing", error)
            self.assertFalse((root / "out").exists())

    def test_truncated_step_exit_two_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_traces(root / "traces", [demo_episode("truncated")])
            with (root / "traces/truncated/steps.jsonl").open("a") as stream:
                stream.write('{"step":')
            code, _, error = self.invoke(str(root / "traces"), "--out", str(root / "out"))
            self.assertEqual(code, 2)
            self.assertIn("truncated", error)
            self.assertFalse((root / "out").exists())

    def test_refuses_different_existing_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_traces(root / "traces", [demo_episode("clean")])
            report = root / "out/doctor.json"
            report.parent.mkdir()
            report.write_bytes(b"earlier output\n")
            code, _, error = self.invoke(str(root / "traces"), "--out", str(report.parent))
            self.assertEqual(code, 2)
            self.assertIn("refusing to overwrite", error)
            self.assertEqual(report.read_bytes(), b"earlier output\n")

    def test_usage_and_help(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["lab", "doctor"])
        self.assertEqual(caught.exception.code, 3)
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as caught:
            main(["lab", "doctor", "--help"])
        self.assertEqual(caught.exception.code, 0)
        for expected in ("Policy Trace Lab", "0 clean", "1 errors found", "2 input unreadable",
                         "3 usage or internal error", "trace", "demonstration"):
            self.assertIn(expected, " ".join(output.getvalue().split()))

    def test_unknown_task_and_unexpected_exception_exit_three(self) -> None:
        self.assertEqual(self.invoke("missing", "--task", "unknown")[0], 3)
        with patch("guided_streaming_policy.lab.cli.get_task", side_effect=RuntimeError("oops")):
            code, _, error = self.invoke("missing")
        self.assertEqual(code, 3)
        self.assertIn("oops", error)

    def test_default_out_without_site_packages(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_traces(root / "traces", [demo_episode("clean")])
            # The -c wrapper keeps this checkout importable after changing cwd.
            script = (
                "import os, sys; "
                f"sys.path.insert(0, {str(Path(__file__).parents[2])!r}); "
                f"os.chdir({temp!r}); "
                "from guided_streaming_policy.cli import main; "
                "raise SystemExit(main(['lab', 'doctor', 'traces']))"
            )
            result = subprocess.run([sys.executable, "-S", "-c", script],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "out/doctor/doctor.json").is_file())

    @unittest.skipUnless(_REAL_DEMOS.is_file(), "native demonstration file is absent")
    def test_real_demonstrations(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            code, output, error = self.invoke(str(_REAL_DEMOS), "--out", temp)
            self.assertEqual(code, 0, output + error)
            report = json.loads((Path(temp) / "doctor.json").read_text())
            self.assertEqual(report["schema_version"], schema.SCHEMA_VERSION)
            self.assertTrue(any(f["check"] == "split.not_checked" for f in report["findings"]))
