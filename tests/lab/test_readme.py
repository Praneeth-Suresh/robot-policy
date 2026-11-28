"""Keep the README command reference and offline transcripts executable."""

import argparse
import re
import shlex
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from guided_streaming_policy.cli import build_parser

__all__: list[str] = []

_ROOT = Path(__file__).resolve().parents[2]
_README = _ROOT / "README.md"
_COMMAND_TIMEOUT_SECONDS = 60
_FENCES = re.compile(r"^```([^\n]*)\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
_COMMAND_HEADING = re.compile(r"^### `gsp lab ([\w -]+)`$", re.MULTILINE)
_OPTION_TABLE = re.compile(
    r"^\| (?:Arguments and options|Argument or option|Option) \|[^\n]*\n"
    r"\|[- |:]+\|\n((?:\|[^\n]*\n)+)",
    re.MULTILINE,
)


def _subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    return next(
        action.choices
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )


def _leaf_commands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """`gsp lab` commands by name, with a command group expanded into "group leaf" names."""
    leaves: dict[str, argparse.ArgumentParser] = {}
    for name, command in _subcommands(parser).items():
        groups = [a for a in command._actions if isinstance(a, argparse._SubParsersAction)]
        if groups:
            leaves.update({f"{name} {leaf}": item for leaf, item in groups[0].choices.items()})
        else:
            leaves[name] = command
    return leaves


def _section(text: str, heading: str) -> str:
    content = text.split(f"## {heading}\n", 1)[1]
    return re.split(r"^## ", content, maxsplit=1, flags=re.MULTILINE)[0]


def _command_sections(text: str) -> dict[str, str]:
    reference = _section(text, "Command reference")
    headings = list(_COMMAND_HEADING.finditer(reference))
    return {
        match[1]: reference[
            match.end() : headings[index + 1].start()
            if index + 1 < len(headings)
            else len(reference)
        ]
        for index, match in enumerate(headings)
    }


def _normalized_output(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).rstrip("\n")


def _argument_names(parser: argparse.ArgumentParser) -> set[str]:
    return {
        name
        for action in parser._actions
        for name in (action.option_strings or [action.dest])
        if name != "--help" and (not name.startswith("-") or name.startswith("--"))
    }


def _exit_codes(text: str) -> set[int]:
    return {int(value) for value in re.findall(r"\b\d+\b", " ".join(text.split()))}


class ReadmeTests(unittest.TestCase):
    def test_command_list_matches_parser_and_chooser(self) -> None:
        text = _README.read_text(encoding="utf-8")
        documented = _COMMAND_HEADING.findall(_section(text, "Command reference"))
        commands = _leaf_commands(_subcommands(build_parser())["lab"])
        self.assertEqual(set(documented), set(commands))
        self.assertEqual(documented, list(commands))
        chooser = _section(text, "Which command do I run?")
        for command in commands:
            with self.subTest(command=command):
                self.assertRegex(chooser, rf"\|.*`gsp lab {re.escape(command)}(?:\s|`)")

    def _clear_documented_outputs(self, words: list[str]) -> None:
        outputs = [word.partition("=")[2] for word in words if word.startswith("--out=")]
        for index, word in enumerate(words):
            if word == "--out":
                self.assertLess(index + 1, len(words), "README --out needs a directory")
                outputs.append(words[index + 1])
        for name in outputs:
            if not name.startswith("out/"):
                continue
            target = _ROOT / name
            self.assertNotIn("..", Path(name).parts, "Unsafe README output traversal")
            self.assertEqual(target.resolve(), target, "README output must not follow symlinks")
            self.assertTrue(target.is_relative_to(_ROOT / "out"), "Only README out/ paths")
            self.assertNotEqual(target, _ROOT / "out", "Do not remove the whole out/ directory")
            if target.exists():
                self.assertTrue(target.is_dir(), "README --out must name a directory")
                shutil.rmtree(target)

    def _assert_policy_table(self, expected: str, actual: str) -> None:
        documented = _normalized_output(expected).splitlines()
        rows = [
            line for line in _normalized_output(actual).splitlines() if not line.startswith("  -")
        ]
        self.assertEqual(rows[0].partition("PAPER")[0].rstrip(), documented[0])
        self.assertEqual(len(rows), len(documented), "README policy inventory changed")
        for shown, full in zip(documented[1:], rows[1:], strict=True):
            shown_tokens, full_tokens = shown.split(), full.split()
            self.assertEqual(shown_tokens[0], full_tokens[0])
            self.assertIn(full_tokens[1], ("runnable", "unavailable"))
            self.assertEqual(shown_tokens[2], "ckpt")
            self.assertEqual(full_tokens[2 : len(shown_tokens)], shown_tokens[2:])

    def _assert_pusht_rows(self, expected: str, actual: str) -> None:
        """Other packs may be installed; compare the header and the built-in pack's row."""
        def wanted(text: str) -> list[list[str]]:
            return [line.split() for line in _normalized_output(text).splitlines()
                    if line.startswith(("NAME", "pusht"))]

        self.assertEqual(wanted(actual), wanted(expected))
        self.assertEqual(len(wanted(expected)), 2)

    def test_offline_examples_in_readme_order(self) -> None:
        text = _README.read_text(encoding="utf-8")
        blocks = list(_FENCES.finditer(text))
        examples = 0
        for index, block in enumerate(blocks):
            code = re.search(r"\bexit=(\d+)\b", block[1])
            if not block[1].startswith("bash offline") or code is None:
                continue
            examples += 1
            line = text.count("\n", 0, block.start()) + 1
            with self.subTest(location=f"README.md:{line}", command=block[2].strip()):
                words = shlex.split(block[2])
                self.assertTrue(words, "Empty README offline example")
                self.assertEqual(words[0], "gsp", "Offline examples must start with gsp")
                self._clear_documented_outputs(words)
                result = subprocess.run(
                    [sys.executable, "-m", "guided_streaming_policy", *words[1:]],
                    cwd=_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=_COMMAND_TIMEOUT_SECONDS,
                )
                self.assertEqual(result.returncode, int(code[1]), result.stdout + result.stderr)
                self._assert_example_output(words, blocks[index + 1 : index + 2], result.stdout)
        self.assertGreater(examples, 0, "No executable README examples found")

    def _assert_example_output(
        self,
        words: list[str],
        following: list[re.Match[str]],
        stdout: str,
    ) -> None:
        if words[1:] in (["list-tasks"], ["run", "infer-static-naive", "--dry-run"]):
            return
        if not following or following[0][1].strip() != "text":
            return
        expected = following[0][2]
        if words[1:3] == ["lab", "policies"]:
            self._assert_policy_table(expected, stdout)
        elif words[1:4] == ["lab", "tasks", "list"]:
            self._assert_pusht_rows(expected, stdout)
        else:
            self.assertEqual(_normalized_output(stdout), _normalized_output(expected))

    def test_help_arguments_and_exit_codes_match_readme(self) -> None:
        sections = _command_sections(_README.read_text(encoding="utf-8"))
        commands = _leaf_commands(_subcommands(build_parser())["lab"])
        for name, parser in commands.items():
            with self.subTest(command=name):
                section = sections[name]
                table = _OPTION_TABLE.search(section)
                self.assertIsNotNone(table, f"README {name}: missing argument table")
                assert table is not None
                documented = {
                    name.strip().strip("`")
                    for row in table[1].splitlines()
                    for name in row.split("|")[1].strip().split(", ")
                }
                self.assertEqual(documented, _argument_names(parser))
                command = [sys.executable, "-m", "guided_streaming_policy", "lab"]
                result = subprocess.run(
                    [*command, *name.split(), "--help"],
                    cwd=_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=_COMMAND_TIMEOUT_SECONDS,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                for argument in documented:
                    self.assertIn(argument, result.stdout)
                bullet = re.search(r"^- \*\*Exit codes:\*\* (.+)$", section, re.MULTILINE)
                self.assertIsNotNone(bullet, f"README {name}: missing exit-code bullet")
                assert bullet is not None
                help_text = " ".join(result.stdout.split())
                self.assertIn("Exit codes:", help_text)
                self.assertEqual(
                    _exit_codes(help_text.split("Exit codes:", 1)[1]), _exit_codes(bullet[1])
                )

    def test_no_placeholders(self) -> None:
        text = _README.read_text(encoding="utf-8")
        for marker in ("@@", "TO" + "DO", "TBD"):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, text)
