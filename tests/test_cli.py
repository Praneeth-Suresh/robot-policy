import subprocess
import sys
import unittest


class CLITests(unittest.TestCase):
    def test_cli_module_entrypoint_lists_tasks(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "guided_streaming_policy.cli", "list-tasks"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("train-policy", completed.stdout)

    def test_package_module_entrypoint_lists_tasks(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "guided_streaming_policy", "list-tasks"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("infer-dynamic-ccg", completed.stdout)
