"""Comparison conditions are checked against the task pack in run, compare and gate."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from guided_streaming_policy.cli import main
from guided_streaming_policy.lab import schema
from tests.lab.helpers import comparison_document, gate_comparison, gate_config, register_fake_task


def _lab(*arguments: str) -> tuple[int, str]:
    out, error = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(error):
        code = main(["lab", *arguments])
    return code, error.getvalue()


def _pusht_comparison(root: Path, condition: str) -> Path:
    document = comparison_document((0,))
    document.update(task="pusht", simulator_version="7.2.0", variants={
        "baseline": {"policy": "pusht/dp"},
        "candidate": {"policy": "pusht/dp", "overrides": {"guidance_scale": 0.0}}},
        intended_differences=["overrides.guidance_scale"])
    document["matrix"] = {"conditions": [condition], "seeds": [0]}
    document["variants"]["baseline"] = {"policy": "pusht/dp"}
    path = root / f"{condition}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class ConditionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        register_fake_task(self)

    def test_compare_refuses_an_unknown_condition_and_lists_the_supported_names(self) -> None:
        path = _pusht_comparison(self.root, "sideways")
        code, error = _lab("compare", str(self.root / "a"), str(self.root / "b"),
                           "--comparison", str(path), "--out", str(self.root / "out"))
        self.assertEqual(code, 2)
        self.assertIn("Unknown condition 'sideways'; supported: static, dynamic", error)
        self.assertFalse((self.root / "out").exists())

    def test_run_refuses_an_unknown_condition_before_running_anything(self) -> None:
        path = _pusht_comparison(self.root, "sideways")
        code, error = _lab("run", "--comparison", str(path), "--variant", "baseline",
                           "--out", str(self.root / "run"))
        self.assertEqual(code, 2)
        self.assertIn("supported: static, dynamic", error)
        self.assertFalse((self.root / "run").exists())

    def test_run_refuses_a_declared_condition_that_has_no_run_hook(self) -> None:
        path = _pusht_comparison(self.root, "dynamic")
        code, error = _lab("run", "--comparison", str(path), "--variant", "baseline",
                           "--out", str(self.root / "run"))
        self.assertEqual(code, 2)
        self.assertIn("cannot run through gsp lab run", error)
        self.assertIn("research scripts", error)

    def test_gate_refuses_a_comparison_with_a_condition_the_pack_does_not_declare(self) -> None:
        gate_path = self.root / "gate-config.json"
        gate_config(gate_path)
        compare_path = self.root / "compare/compare.json"
        compare_path.parent.mkdir()
        compare_path.write_text(schema.canonical_json(gate_comparison(self.root)))
        _, accepted = _lab("gate", str(compare_path), "--gate", str(gate_path),
                           "--out", str(self.root / "ok"))
        self.assertNotIn("Unknown condition", accepted)
        document = json.loads(compare_path.read_text())
        for pair in document["pairs"]:
            pair["key"]["condition"] = "sideways"
        bad = self.root / "bad.json"
        bad.write_text(json.dumps(document), encoding="utf-8")
        code, error = _lab("gate", str(bad), "--gate", str(gate_path),
                           "--out", str(self.root / "blocked"))
        self.assertEqual(code, 2)
        self.assertIn("Unknown condition 'sideways'; supported: c1", error)
        self.assertFalse((self.root / "blocked").exists())

    def test_an_unloadable_task_is_refused_with_its_reason(self) -> None:
        document = comparison_document((0,))
        document["task"] = "nowhere"
        path = self.root / "nowhere.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        code, error = _lab("compare", str(self.root / "a"), str(self.root / "b"),
                           "--comparison", str(path), "--out", str(self.root / "out"))
        self.assertEqual(code, 2)
        self.assertIn("Unknown task 'nowhere'", error)


if __name__ == "__main__":
    unittest.main()
