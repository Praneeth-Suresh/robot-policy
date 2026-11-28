"""Entry-point discovery: metadata-only listing, isolation of broken packs and conflicts."""

import sys
import unittest

from guided_streaming_policy.lab.tasks import registry
from guided_streaming_policy.lab.tasks.base import CONTRACT_VERSION
from tests.lab.helpers import FakeTask
from tests.lab.packs import PackSpec, installed_packs

_GOOD = """
from guided_streaming_policy.lab.tasks.base import CONTRACT_VERSION


class Pack:
    name = "{task}"
    contract_version = {version}
"""


def _good(task: str, version: str = "CONTRACT_VERSION") -> PackSpec:
    return PackSpec(task, source=_GOOD.format(task=task, version=version))


class RegistryTests(unittest.TestCase):
    def test_pusht_is_registered_through_the_same_entry_point_path(self) -> None:
        entry = {item.name: item for item in registry.entries()}["pusht"]
        self.assertEqual(entry.target, "guided_streaming_policy.lab.tasks.pusht.adapter:PushTTask")
        self.assertEqual(entry.distribution, "guided-streaming-policy")
        self.assertEqual(registry.get_task("pusht").name, "pusht")

    def test_listing_reads_metadata_and_never_imports_the_pack(self) -> None:
        with installed_packs(_good("rail")):
            self.assertIn("rail", registry.task_names())
            self.assertNotIn("fake_pack_rail", sys.modules)
            self.assertEqual(registry.get_task("rail").name, "rail")
            self.assertIn("fake_pack_rail", sys.modules)

    def test_unknown_task_lists_the_known_names(self) -> None:
        with self.assertRaisesRegex(KeyError, "known tasks: .*pusht"):
            registry.get_task("missing")

    def test_a_pack_that_cannot_be_imported_breaks_only_itself(self) -> None:
        broken = PackSpec("ghost", target="no_such_module_anywhere:Pack")
        with installed_packs(broken, _good("rail")):
            with self.assertRaises(registry.TaskLoadError) as caught:
                registry.get_task("ghost")
            self.assertIn("ModuleNotFoundError", str(caught.exception))
            self.assertEqual(registry.get_task("rail").name, "rail")
            self.assertEqual(registry.get_task("pusht").name, "pusht")

    def test_a_pack_that_raises_while_constructing_is_reported(self) -> None:
        source = "class Pack:\n    def __init__(self):\n        raise RuntimeError('no sim')\n"
        with installed_packs(PackSpec("angry", source=source)):
            with self.assertRaisesRegex(registry.TaskLoadError, "RuntimeError: no sim"):
                registry.get_task("angry")

    def test_contract_version_mismatch_names_both_numbers(self) -> None:
        with installed_packs(_good("future", version="7")):
            with self.assertRaises(registry.TaskLoadError) as caught:
                registry.get_task("future")
        message = str(caught.exception)
        self.assertIn("contract version 7", message)
        self.assertIn(f"requires version {CONTRACT_VERSION}", message)

    def test_a_pack_without_a_contract_version_is_refused(self) -> None:
        source = "class Pack:\n    name = 'old'\n"
        with installed_packs(PackSpec("old", source=source)):
            with self.assertRaisesRegex(registry.TaskLoadError, "contract version None"):
                registry.get_task("old")

    def test_a_pack_must_declare_the_name_it_registers(self) -> None:
        source = _GOOD.format(task="other", version="CONTRACT_VERSION")
        with installed_packs(PackSpec("liar", source=source)):
            with self.assertRaisesRegex(registry.TaskLoadError, "declares the name 'other'"):
                registry.get_task("liar")

    def test_two_distributions_claiming_one_name_are_named_in_the_error(self) -> None:
        first = PackSpec("twin", distribution="alpha", source=_GOOD.format(
            task="twin", version="CONTRACT_VERSION"), module="twin_a")
        second = PackSpec("twin", distribution="beta", version="2.0", source=_GOOD.format(
            task="twin", version="CONTRACT_VERSION"), module="twin_b")
        with installed_packs(first, second):
            conflicted = {e.name: e for e in registry.entries()}["twin"]
            self.assertIn("alpha 1.0", conflicted.conflict or "")
            self.assertIn("beta 2.0", conflicted.conflict or "")
            with self.assertRaisesRegex(registry.TaskLoadError, "claimed by"):
                registry.get_task("twin")
            self.assertEqual(registry.get_task("pusht").name, "pusht")

    def test_register_adds_an_in_process_task_and_unregister_removes_it(self) -> None:
        registry.register("inproc", lambda: _named(FakeTask, "inproc"))
        try:
            self.assertIn("inproc", registry.task_names())
            self.assertEqual(registry.get_task("inproc").name, "inproc")
            with self.assertRaisesRegex(ValueError, "already registered"):
                registry.register("inproc", lambda: _named(FakeTask, "inproc"))
            with self.assertRaisesRegex(ValueError, "already registered"):
                registry.register("pusht", lambda: _named(FakeTask, "pusht"))
        finally:
            registry.unregister("inproc")
        self.assertNotIn("inproc", registry.task_names())


def _named(factory: type[FakeTask], name: str) -> FakeTask:
    import tempfile
    from pathlib import Path

    task = factory(Path(tempfile.gettempdir()))
    task.name = name
    return task


if __name__ == "__main__":
    unittest.main()
