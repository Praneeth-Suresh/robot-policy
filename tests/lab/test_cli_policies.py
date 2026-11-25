"""Additive Policy Trace Lab CLI behavior and usage exits."""

import io
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

from guided_streaming_policy.cli import main

__all__: list[str] = []


class PoliciesCLITests(unittest.TestCase):
    def test_policy_listing(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["lab", "policies"]), 0)
        for name in ("pusht/naive", "pusht/ccg", "pusht/steg", "pusht/dp", "pusht/flow"):
            self.assertIn(name, output.getvalue())
        self.assertRegex(output.getvalue(), "runnable|unavailable")

    def test_lab_usage_errors(self) -> None:
        for argv in (
            ["lab", "bogus"],
            ["lab"],
            ["lab", "policies", "--bogus"],
            ["lab", "policies", "--task"],
        ):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main(argv)
                self.assertEqual(caught.exception.code, 3)

    def test_unknown_task(self) -> None:
        error = io.StringIO()
        with redirect_stderr(error):
            self.assertEqual(main(["lab", "policies", "--task", "missing"]), 3)
        self.assertIn("pusht", error.getvalue())

    def test_help_and_exit_code_documentation(self) -> None:
        for argv in (["lab", "--help"], ["lab", "policies", "--help"]):
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaises(SystemExit) as caught:
                main(argv)
            self.assertEqual(caught.exception.code, 0)
            self.assertIn("Policy Trace Lab", output.getvalue())
            if "policies" in argv:
                self.assertIn("0 listed, 3 usage error", output.getvalue())

    def test_legacy_list_tasks(self) -> None:
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["list-tasks"]), 0)

    def test_module_entry_point_without_site_packages(self) -> None:
        result = subprocess.run(
            [sys.executable, "-S", "-m", "guided_streaming_policy", "lab", "policies"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("pusht/naive", result.stdout)
        self.assertIn("unavailable", result.stdout)
