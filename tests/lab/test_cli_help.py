"""Exact exit-code and output-preservation wording in formatted CLI help."""

import io
import unittest
from contextlib import redirect_stdout

from guided_streaming_policy.cli import main

__all__: list[str] = []

_EXIT_HELP = {
    "": (
        "Exit codes: 0 success; 1 findings or regression (doctor, gate, demo); "
        "2 invalid, unreadable or incomplete input; 3 usage or internal error."
    ),
    "policies": "Exit codes: 0 listed, 3 usage error.",
    "doctor": (
        "Exit codes: 0 clean (no errors), 1 errors found, "
        "2 input unreadable or output would overwrite, 3 usage or internal error."
    ),
    "run": (
        "Exit codes: 0 all trials recorded (crashed trials are recorded and logged), "
        "2 setup invalid, 3 usage or internal error."
    ),
    "compare": (
        "Exit codes: 0 comparable, 2 invalid comparison or unreadable input, "
        "3 usage or internal error."
    ),
    "gate": (
        "Exit codes: 0 pass, 1 regression, 2 invalid, incomplete or a required check unknown, "
        "3 usage or internal error."
    ),
    "report": (
        "Exit codes: 0 written, 2 unreadable input or overwrite refused, "
        "3 usage or internal error."
    ),
    "demo": (
        "Exit codes: same as gate: 0 pass, 1 regression, 2 invalid or incomplete, "
        "3 usage or internal error. The bundled fixture exits 1."
    ),
}
_OUT_HELP = {
    "doctor": "Report directory (default: out/doctor).",
    "run": "Output directory (default: out/runs/<comparison name>/<variant>); must be empty.",
    "compare": "Output directory (default: out/compare/<comparison name>).",
    "gate": "Report directory (default: out/gate/<gate name>).",
    "report": "Output directory (default: out/report/<comparison name>).",
    "demo": (
        "Output directory. Identical existing files are kept; "
        "different files are never overwritten."
    ),
}


class CLIHelpTests(unittest.TestCase):
    def formatted_help(self, command: str) -> str:
        output = io.StringIO()
        args = ["lab", *([command] if command else []), "--help"]
        with redirect_stdout(output), self.assertRaises(SystemExit) as caught:
            main(args)
        self.assertEqual(caught.exception.code, 0)
        return " ".join(output.getvalue().split())

    def test_exact_exit_code_epilogs(self) -> None:
        for command, expected in _EXIT_HELP.items():
            with self.subTest(command=command or "lab"):
                text = self.formatted_help(command)
                self.assertEqual(text.count("Exit codes:"), 1)
                self.assertEqual("Exit codes:" + text.split("Exit codes:", 1)[1], expected)

    def test_output_preservation_help(self) -> None:
        for command, expected in _OUT_HELP.items():
            with self.subTest(command=command):
                if command != "demo":
                    expected += " Existing files with different content are never overwritten."
                text = self.formatted_help(command)
                self.assertIn("--out OUT " + expected, text)
