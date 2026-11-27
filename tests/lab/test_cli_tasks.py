"""`gsp lab tasks list|new|check`: output, refusals, broken packs and exit codes."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from guided_streaming_policy.cli import main
from guided_streaming_policy.lab.tasks import registry
from tests.lab.packs import PACK, PackSpec, ScaffoldedPack, installed_packs


def _lab(*arguments: str) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            code = main(["lab", *arguments])
        except SystemExit as error:
            code = int(error.code or 0)
    return code, stdout.getvalue(), stderr.getvalue()


class ListTests(unittest.TestCase):
    def test_lists_pusht_ready_with_contract_policies_and_conditions(self) -> None:
        code, out, _ = _lab("tasks", "list")
        self.assertEqual(code, 0)
        header, row = out.splitlines()[:2]
        self.assertEqual(header.split(), ["NAME", "DISTRIBUTION", "CONTRACT", "STATUS",
                                          "POLICIES", "CONDITIONS"])
        self.assertEqual(row.split(), ["pusht", "guided-streaming-policy", "0.1.0", "1",
                                       "ready", "5", "static,", "dynamic"])

    def test_verbose_adds_each_condition_with_its_description(self) -> None:
        code, out, _ = _lab("tasks", "list", "--verbose")
        self.assertEqual(code, 0)
        self.assertIn("    static: Fixed obstacle", out)
        self.assertRegex(out, r"dynamic: .*\(declared; not runnable\)")

    def test_a_broken_pack_is_listed_as_broken_and_other_commands_still_run(self) -> None:
        broken = PackSpec("ghost", target="no_such_module_anywhere:Pack")
        mismatch = PackSpec("future", source="class Pack:\n    name = 'future'\n"
                                             "    contract_version = 7\n")
        with installed_packs(broken, mismatch):
            code, out, _ = _lab("tasks", "list")
            self.assertEqual(code, 0)
            rows = {line.split()[0]: line for line in out.splitlines()[1:] if line.strip()}
            self.assertIn("broken: ModuleNotFoundError", rows["ghost"])
            self.assertIn("broken: task 'future' implements contract version 7", rows["future"])
            self.assertIn("ready", rows["pusht"])
            fixtures = Path(__file__).resolve().parents[2] / (
                "guided_streaming_policy/lab/tasks/pusht/fixtures")
            with tempfile.TemporaryDirectory() as temporary:
                self.assertEqual(_lab("doctor", str(fixtures / "demos/clean"), "--out",
                                      str(Path(temporary) / "d"))[0], 0)
                self.assertEqual(_lab("demo", "--out", str(Path(temporary) / "demo"))[0], 1)
                code, _, error = _lab("doctor", str(fixtures / "demos/clean"), "--task", "ghost",
                                      "--out", str(Path(temporary) / "x"))
                self.assertEqual(code, 2)
                self.assertIn("Cannot load task 'ghost'", error)

    def test_a_doubly_claimed_name_is_reported_with_both_distributions(self) -> None:
        source = "class Pack:\n    name = 'twin'\n    contract_version = 1\n"
        first = PackSpec("twin", distribution="alpha", source=source, module="twin_a")
        second = PackSpec("twin", distribution="beta", version="2.0", source=source,
                          module="twin_b")
        with installed_packs(first, second):
            _, out, _ = _lab("tasks", "list")
        self.assertRegex(out, r"broken: task 'twin' is claimed by alpha 1.0 and beta 2.0")


class NewAndCheckTests(unittest.TestCase):
    def test_new_writes_a_standalone_pack_and_refuses_what_it_should(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "pack"
            code, stdout, _ = _lab("tasks", "new", PACK, "--out", str(out))
            self.assertEqual(code, 0)
            self.assertIn(f"gsp lab tasks check {PACK}", stdout)
            for name in ("pyproject.toml", "README.md", "rail/adapter.py", "rail/simulator.py",
                         "rail/pins.json", "rail/fixtures/cases.json", "configs/comparison.json",
                         "configs/gate.json", "tests/test_contract.py",
                         "tools/build_fixtures.py"):
                self.assertTrue((out / name).is_file(), name)
            pyproject = (out / "pyproject.toml").read_text()
            self.assertIn('[project.entry-points."gsp_lab.tasks"]', pyproject)
            self.assertIn('rail = "rail.adapter:RailTask"', pyproject)
            for path in out.rglob("*"):
                if path.is_file() and path.suffix in (".py", ".toml", ".md", ".json"):
                    self.assertNotIn("${", path.read_text(), path.name)
            code, _, error = _lab("tasks", "new", PACK, "--out", str(out))
            self.assertEqual(code, 2)
            self.assertIn("not empty", error)
            self.assertEqual(_lab("tasks", "new", "pusht", "--out", str(Path(temporary) / "x"))[0],
                             2)
            self.assertEqual(_lab("tasks", "new", "Bad Name", "--out",
                                  str(Path(temporary) / "y"))[0], 2)
            self.assertFalse((Path(temporary) / "x").exists())

    def test_usage_errors_exit_three(self) -> None:
        for arguments in (("tasks",), ("tasks", "new", "x"), ("tasks", "check"),
                          ("tasks", "list", "--bogus")):
            with self.subTest(arguments=arguments):
                self.assertEqual(_lab(*arguments)[0], 3)

    def test_check_passes_a_conforming_pack_and_prints_each_result(self) -> None:
        pack = ScaffoldedPack()
        self.addCleanup(pack.close)
        registry.register(PACK, pack.adapter_module.RailTask)
        self.addCleanup(registry.unregister, PACK)
        code, out, _ = _lab("tasks", "check", PACK)
        self.assertEqual(code, 0, out)
        self.assertIn("pass contract_version: version 1", out)
        self.assertIn("pass fixture:seeded_regression", out)
        self.assertTrue(out.rstrip().endswith("17 passed, 0 failed, 0 skipped"))

    def test_check_exits_one_and_names_the_failing_case(self) -> None:
        pack = ScaffoldedPack()
        self.addCleanup(pack.close)
        path = pack.root / "rail/fixtures/cases.json"
        cases = json.loads(path.read_text())
        cases["cases"]["seeded_regression"]["gate_exit"] = 0
        path.write_text(json.dumps(cases))
        registry.register(PACK, pack.adapter_module.RailTask)
        self.addCleanup(registry.unregister, PACK)
        code, out, _ = _lab("tasks", "check", PACK)
        self.assertEqual(code, 1)
        self.assertRegex(out, r"fail fixture:seeded_regression: .*gate_exit")
        self.assertIn("1 failed", out)

    def test_check_exits_two_when_the_pack_cannot_be_loaded(self) -> None:
        self.assertEqual(_lab("tasks", "check", "missing")[0], 2)
        with installed_packs(PackSpec("ghost", target="no_such_module_anywhere:Pack")):
            code, _, error = _lab("tasks", "check", "ghost")
        self.assertEqual(code, 2)
        self.assertIn("Cannot load task 'ghost'", error)


if __name__ == "__main__":
    unittest.main()
