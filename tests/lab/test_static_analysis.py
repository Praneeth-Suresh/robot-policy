"""Run the configured developer gates with the interpreter running the tests."""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

__all__: list[str] = []

ROOT = Path(__file__).resolve().parents[2]


class StaticAnalysisTests(unittest.TestCase):
    def run_tool(self, tool: str, *arguments: str) -> None:
        if importlib.util.find_spec(tool) is None:
            self.skipTest(f"{tool} unavailable; CI installs the dev extra")
        result = subprocess.run(
            [sys.executable, "-m", tool, *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_ruff(self) -> None:
        self.run_tool("ruff", "check", "guided_streaming_policy/lab", "tests/lab")

    def test_mypy(self) -> None:
        self.run_tool("mypy")

    def test_vulture(self) -> None:
        self.run_tool("vulture")
