"""Simulator command wiring, diagnostics and exit codes using a fake task."""

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from guided_streaming_policy.cli import main
from guided_streaming_policy.lab import schema
from guided_streaming_policy.lab.tasks.base import TrialRecord, TrialRequest
from tests.lab.helpers import FakeTask, comparison_document

__all__: list[str] = []


class RunCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = FakeTask(self.root)
        self.config = self.root / "comparison.json"
        self.config.write_text(schema.canonical_json(comparison_document()))

    def invoke(self, *args: str) -> tuple[int, str, str]:
        output, error = io.StringIO(), io.StringIO()
        with patch("guided_streaming_policy.lab.cli.get_task", return_value=self.task) as lookup:
            with redirect_stdout(output), redirect_stderr(error):
                code = main(["lab", "run", *args])
            lookup.assert_called_once_with("toy")
        return code, output.getvalue(), error.getvalue()

    def test_exports_and_logs_with_task_from_comparison(self) -> None:
        out = self.root / "run"
        code, output, error = self.invoke("--comparison", str(self.config), "--variant", "baseline",
                                          "--out", str(out))
        self.assertEqual(code, 0, error)
        self.assertEqual(output, f"run: {out} (2 trials)\n")
        self.assertIn("c1 seed=0", error)
        self.assertTrue((out / "run_manifest.json").is_file())

    def test_setup_error_is_two_and_message_on_stderr(self) -> None:
        self.task.unavailable_reason = "checkpoint absent"
        code, output, error = self.invoke("--comparison", str(self.config), "--variant", "baseline",
                                          "--out", str(self.root / "out"))
        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertIn("checkpoint absent", error)

    def test_crashes_still_exit_zero(self) -> None:
        self.task.crash_seeds = (0,)
        code, output, error = self.invoke(
            "--comparison", str(self.config), "--variant", "candidate",
            "--out", str(self.root / "out"))
        self.assertEqual(code, 0, error)
        self.assertIn("2 trials", output)
        self.assertIn("crashed", error)

    def test_runner_stdout_is_redirected_to_stderr(self) -> None:
        original = self.task.run_trial
        def noisy_runner(request: TrialRequest) -> TrialRecord:
            sys.stdout.write("simulator message\n")
            return original(request)
        with patch.object(self.task, "run_trial", side_effect=noisy_runner):
            code, output, error = self.invoke(
                "--comparison", str(self.config), "--variant", "baseline",
                "--out", str(self.root / "run"))
        self.assertEqual(code, 0, error)
        self.assertNotIn("simulator message", output)
        self.assertEqual(error.count("simulator message"), 2)

    def test_default_output_is_derived_in_exporter(self) -> None:
        out = self.root / "default"
        with patch("guided_streaming_policy.lab.cli.exporter.default_out_dir", return_value=out):
            code, output, error = self.invoke("--comparison", str(self.config),
                                              "--variant", "baseline")
        self.assertEqual(code, 0, error)
        self.assertIn(str(out), output)

    def test_invalid_config_is_two(self) -> None:
        self.config.write_text("invalid")
        error = io.StringIO()
        with redirect_stderr(error):
            code = main(["lab", "run", "--comparison", str(self.config), "--variant", "baseline"])
        self.assertEqual(code, 2)
        self.assertIn("comparison", error.getvalue())

    def test_usage_and_help(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["lab", "run"])
        self.assertEqual(caught.exception.code, 3)
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as caught:
            main(["lab", "run", "--help"])
        self.assertEqual(caught.exception.code, 0)
        help_text = " ".join(output.getvalue().split())
        for phrase in ("Policy Trace Lab", "simulator", "pusht_cchi_v7_replay.zarr.zip",
                       "pusht_state_1000ep_si.ckpt", "CCG_V6_ep200.pth",
                       "pusht_state_100ep_dp.ckpt", "pusht_state_flow_100ep.ckpt",
                       "0", "1", "2", "3"):
            self.assertIn(phrase, help_text)

    def test_internal_error_is_three(self) -> None:
        with patch("guided_streaming_policy.lab.cli.exporter.export_variant",
                   side_effect=RuntimeError("unexpected")):
            code, _, error = self.invoke("--comparison", str(self.config), "--variant", "baseline",
                                          "--out", str(self.root / "out"))
        self.assertEqual(code, 3)
        self.assertIn("unexpected", error)
