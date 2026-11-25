"""Offline demo pipeline, repeatability and preservation through the real CLI."""

import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from guided_streaming_policy.cli import main
from guided_streaming_policy.lab import adapters
from guided_streaming_policy.lab.tasks.base import FixtureSet
from guided_streaming_policy.lab.tasks.pusht.adapter import PushTTask

__all__: list[str] = []

_OUTPUTS = ("doctor/clean/doctor.json", "doctor/corrupt/doctor.json", "compare/compare.json",
            "gate/gate.json", "report/report.html")
_HEADERS = ("== Doctor: clean fixture (synthetic) ==",
            "== Doctor: corrupt fixture (synthetic) ==",
            "== Gate (synthetic fixture, not a benchmark) ==")


def _invoke(args: list[str]) -> tuple[int, str, str]:
    output, error = io.StringIO(), io.StringIO()
    with redirect_stdout(output), redirect_stderr(error):
        try:
            code = main(["lab", "demo", *args])
        except SystemExit as caught:
            code = int(caught.code)
    return code, output.getvalue(), error.getvalue()


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


class DemoCLITests(unittest.TestCase):
    def test_offline_demo_and_identical_rerun(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("socket.socket", side_effect=AssertionError("network")),
            patch.object(subprocess, "Popen", side_effect=AssertionError("subprocess")),
        ):
            root = Path(temporary)
            args = ["--out", str(root)]
            code, output, error = _invoke(args)
            self.assertEqual(code, 1, error)
            self.assertEqual(error, "")
            positions = [output.index(header) for header in _HEADERS]
            self.assertEqual(positions, sorted(positions))
            for header in _HEADERS[1:]:
                self.assertIn("\n\n" + header, output)
            for text in ("REGRESSION", "exit 1", "separation_proxy", "report.html",
                         "value.non_finite", "shape.mismatch", "time.non_monotonic",
                         "demo exit code: 1"):
                self.assertIn(text, output)
            self.assertTrue(all((root / name).is_file() for name in _OUTPUTS))
            self.assertIn("Synthetic fixture data. Not a benchmark.",
                          (root / "report/report.html").read_text())
            result = json.loads((root / "gate/gate.json").read_text())["regressions"][0]
            key = result["key"]
            self.assertIn(f"open: {root / 'report/report.html'}#pair="
                          f"{key['condition']}:{key['seed']}&step={result['replay_step']}", output)
            before = _tree(root)
            self.assertEqual(_invoke(args), (code, output, error))
            self.assertEqual(_tree(root), before)

    def test_tampered_outputs_and_copies_are_preserved(self) -> None:
        for name in (*_OUTPUTS, "fixtures/scenario/baseline/resolved_config.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                args = ["--out", str(root)]
                self.assertEqual(_invoke(args)[0], 1)
                target = root / name
                target.write_bytes(b"preserve me")
                code, _, error = _invoke(args)
                self.assertEqual(code, 2)
                self.assertIn("refusing to overwrite", error)
                self.assertEqual(target.read_bytes(), b"preserve me")

    def test_usage_help_and_missing_fixtures(self) -> None:
        self.assertEqual(_invoke([])[0], 3)
        self.assertEqual(_invoke(["--out", "unused", "--task", "bogus"])[0], 3)
        code, output, _ = _invoke(["--help"])
        self.assertEqual(code, 0)
        for text in ("Policy Trace Lab", "CPU", "default install", "0 pass", "1 regression",
                     "2 invalid or incomplete", "3 usage or internal error"):
            self.assertIn(text, " ".join(output.split()))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = FixtureSet(root / "clean", root / "corrupt", root / "scenario")
            with patch.object(PushTTask, "fixtures", return_value=fixture):
                code, _, error = _invoke(["--out", str(root / "out")])
            self.assertEqual(code, 2)
            self.assertIn("fixtures not found; run from a source checkout of the repository", error)
            self.assertFalse((root / "out").exists())

    def test_partial_fixture_files_are_reported_before_output_is_created(self) -> None:
        source = PushTTask().fixtures()
        missing = ("scenario/baseline/static__2/manifest.json",
                   "scenario/candidate/static__1/steps.jsonl", "clean/clean_0/manifest.json",
                   "scenario/baseline/static__2", "scenario/gate.json")
        for name in missing:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = FixtureSet(root / "clean", root / "corrupt", root / "scenario")
                for original, copied in ((source.demo_clean, fixture.demo_clean),
                                         (source.demo_corrupt, fixture.demo_corrupt),
                                         (source.scenario, fixture.scenario)):
                    adapters.copy_tree_once(original, copied)
                path = root / name
                if path.is_dir():
                    for file in path.iterdir():
                        file.unlink()
                    path.rmdir()
                else:
                    path.unlink()
                with patch.object(PushTTask, "fixtures", return_value=fixture):
                    code, _, error = _invoke(["--out", str(root / "out")])
                self.assertEqual(code, 2)
                self.assertIn("fixtures not found; run from a source checkout of the repository",
                              error)
                self.assertFalse((root / "out").exists())

    def test_copy_tree_once_keeps_identical_binary_files_and_never_deletes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / "source", root / "target"
            (source / "nested/empty").mkdir(parents=True)
            (source / "nested/blob").write_bytes(bytes(range(256)))
            adapters.copy_tree_once(source, target)
            self.assertTrue((target / "nested/empty").is_dir())
            (target / "extra").write_text("keep")
            before = _tree(target)
            adapters.copy_tree_once(source, target)
            self.assertEqual(_tree(target), before)
            (source / "nested/blob").write_bytes(b"changed")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                adapters.copy_tree_once(source, target)
            self.assertEqual(_tree(target), before)
