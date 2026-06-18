import unittest

from guided_streaming_policy.paths import pusht_root
from guided_streaming_policy.pusht_legacy import build_task_command, get_task, task_names


class PushTLegacyTests(unittest.TestCase):
    def test_known_task_registry_entry(self) -> None:
        task = get_task("train-policy")
        self.assertEqual(task.module, "train.train_policy")

    def test_task_names_include_expected_entries(self) -> None:
        names = task_names()
        self.assertIn("train-policy", names)
        self.assertIn("infer-dynamic-ccg", names)

    def test_build_task_command_uses_module_execution(self) -> None:
        task = get_task("train-dp")
        command = build_task_command(task, python_executable="python3")
        self.assertEqual(command, ["python3", "-m", "train.train_dp"])

    def test_pusht_root_points_to_legacy_workspace(self) -> None:
        self.assertTrue(pusht_root().name == "Push-T")


if __name__ == "__main__":
    unittest.main()
