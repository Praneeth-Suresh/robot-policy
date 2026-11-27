"""A scaffolded one-dimensional task pack passes the conformance kit and the whole lab.

The rail simulator lives once, in the scaffold templates. These tests generate it into a
temporary directory, import it and register it with one line, so any Push-T assumption
left in the core makes them fail.
"""

import importlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from guided_streaming_policy.cli import main
from guided_streaming_policy.lab import cli_tasks, testing
from guided_streaming_policy.lab.tasks import registry
from tests.lab.packs import PACK, ScaffoldedPack

ROOT = Path(__file__).resolve().parents[2]
_BLOCKED = ("numpy", "torch", "zarr", "pymunk")


class RailPackConformanceTests(testing.TaskContractTests):
    """Every conformance check runs against the freshly scaffolded pack."""

    pack: ScaffoldedPack

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.pack = ScaffoldedPack()
        cls.addClassCleanup(cls.pack.close)

    def make_adapter(self) -> Any:
        return self.pack.adapter_module.RailTask()

    def test_no_check_is_skipped_for_a_pure_python_pack(self) -> None:
        results = testing.run_conformance(self.make_adapter())
        self.assertEqual([r.name for r in results if r.status != testing.PASS], [])
        self.assertEqual(len(results), 5 + len(testing.FIXTURE_CASES) + 2)


class RailPipelineTests(unittest.TestCase):
    pack: ScaffoldedPack

    @classmethod
    def setUpClass(cls) -> None:
        cls.pack = ScaffoldedPack()
        cls.addClassCleanup(cls.pack.close)
        # The sole wiring a new simulator needs: one registration.
        registry.register(PACK, cls.pack.adapter_module.RailTask)
        cls.addClassCleanup(registry.unregister, PACK)

    def invoke(self, expected: int, *arguments: str) -> str:
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            actual = main(["lab", *arguments])
        message = f"{arguments}:\n{stdout.getvalue()}\n{stderr.getvalue()}"
        self.assertEqual(actual, expected, message)
        return stdout.getvalue()

    def test_whole_offline_pipeline(self) -> None:
        pack = self.pack.root
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            self.assertIn("rail/steady", self.invoke(0, "policies", "--task", PACK))
            self.assertIn("rail", self.invoke(0, "tasks", "list"))
            fixtures = self.pack.adapter_module.RailTask().fixtures()
            self.invoke(0, "doctor", str(fixtures.demo_clean), "--task", PACK,
                        "--out", str(out / "doctor-clean"))
            self.invoke(1, "doctor", str(fixtures.demo_corrupt), "--task", PACK,
                        "--out", str(out / "doctor-corrupt"))
            comparison, gate = pack / "configs/comparison.json", pack / "configs/gate.json"
            for variant in ("baseline", "candidate"):
                self.invoke(0, "run", "--comparison", str(comparison), "--variant", variant,
                            "--out", str(out / variant))
            self.invoke(0, "doctor", str(out / "candidate"), "--task", PACK,
                        "--out", str(out / "doctor-run"))
            self.invoke(0, "compare", str(out / "baseline"), str(out / "candidate"),
                        "--comparison", str(comparison), "--out", str(out / "compare"))
            self.invoke(1, "gate", str(out / "compare/compare.json"), "--gate", str(gate),
                        "--out", str(out / "gate"))
            result = json.loads((out / "gate/gate.json").read_text())
            self.assertEqual((result["verdict"], result["exit_code"]), ("REGRESSION", 1))
            separation = {rule["id"]: rule for rule in result["rules"]}["separation_proxy"]
            self.assertEqual((separation["units"], separation["verdict"]), ("m", "fail"))
            regression = result["regressions"][0]
            self.assertEqual((regression["key"]["task"], regression["category"]),
                             (PACK, "constraint_fail"))
            self.invoke(0, "report", str(out / "compare/compare.json"), "--gate",
                        str(out / "gate/gate.json"), "--out", str(out / "report"))
            self.assert_report((out / "report/report.html").read_text())
            self.invoke(1, "demo", "--task", PACK, "--out", str(out / "demo"))
            self.validate_documents(out, pack)

    def validate_documents(self, out: Path, pack: Path) -> None:
        if importlib.util.find_spec("jsonschema") is None:
            return
        import jsonschema

        examples = {
            "comparison_config.schema.json": pack / "configs/comparison.json",
            "gate_config.schema.json": pack / "configs/gate.json",
            "run_manifest.schema.json": out / "candidate/run_manifest.json",
            "episode_manifest.schema.json": out / "candidate/nominal__seed0/manifest.json",
            "doctor_report.schema.json": out / "doctor-clean/doctor.json",
            "compare_result.schema.json": out / "compare/compare.json",
            "gate_result.schema.json": out / "gate/gate.json",
        }
        for filename, path in examples.items():
            with self.subTest(schema=filename):
                definition = json.loads((ROOT / "schemas" / filename).read_text())
                jsonschema.validate(json.loads(path.read_text()), definition)
        definition = json.loads((ROOT / "schemas/episode_step.schema.json").read_text())
        lines = (out / "candidate/nominal__seed0/steps.jsonl").read_text().splitlines()
        for line in lines:
            jsonschema.validate(json.loads(line), definition)

    def assert_report(self, page: str) -> None:
        payload = page.split('<script type="application/json" id="trace-data">', 1)[1]
        data = json.loads(payload.split("</script>", 1)[0])
        pair = data["pairs"][0]
        self.assertEqual(pair["baseline"]["observation_schema"]["fields"]["x"]["shape"], [1])
        self.assertEqual((pair["separation"]["signal"], pair["separation"]["units"]),
                         ("obs.x", "m"))
        self.assertIn("Separation proxy", page)
        self.assertIn("REGRESSION", page)

    def test_pipeline_imports_no_optional_runtime_packages(self) -> None:
        script = f"""
import importlib.abc, importlib.util, sys, unittest
class Block(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {set(_BLOCKED)!r}:
            return importlib.util.spec_from_loader(fullname, self)
        return None
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        raise AssertionError('the rail pipeline imported ' + module.__name__)
sys.meta_path.insert(0, Block())
suite = unittest.defaultTestLoader.loadTestsFromName(
    'tests.lab.test_toy_task_contract.RailPipelineTests.test_whole_offline_pipeline')
raise SystemExit(not unittest.TextTestRunner().run(suite).wasSuccessful())
"""
        done = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True,
                              text=True, check=False)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)


class ScaffoldRefusalTests(unittest.TestCase):
    def test_refuses_a_registered_name_a_bad_name_and_a_non_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "pack"
            for name in ("pusht", "Bad Name", "1rail", "json"):
                with self.subTest(name=name), self.assertRaises(cli_tasks.ScaffoldError):
                    cli_tasks.scaffold_pack(name, out)
            self.assertFalse(out.exists())
            (Path(temporary) / "full").mkdir()
            (Path(temporary) / "full/keep.txt").write_text("keep")
            with self.assertRaisesRegex(cli_tasks.ScaffoldError, "not empty"):
                cli_tasks.scaffold_pack(PACK, Path(temporary) / "full")
            self.assertEqual([p.name for p in (Path(temporary) / "full").iterdir()], ["keep.txt"])


if __name__ == "__main__":
    unittest.main()
