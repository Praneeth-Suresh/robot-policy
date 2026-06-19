from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class ReleaseStructureTests(unittest.TestCase):
    def test_public_docs_do_not_link_to_local_machine_paths(self) -> None:
        docs = [
            REPO_ROOT / "README.md",
            REPO_ROOT / "Push-T" / "README.md",
            REPO_ROOT / "docs" / "reproducibility.md",
        ]

        for path in docs:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/home/users/", text, msg=str(path))
            self.assertNotIn("/Users/", text, msg=str(path))

    def test_experiment_scripts_require_explicit_module_execution(self) -> None:
        scripts = list((REPO_ROOT / "Push-T" / "inference" / "static").glob("inference_*.py"))
        scripts += list((REPO_ROOT / "Push-T" / "inference" / "dynamic").glob("inference_*.py"))
        scripts += [
            REPO_ROOT / "Push-T" / "train" / "train_dp.py",
            REPO_ROOT / "Push-T" / "train" / "train_policy.py",
            REPO_ROOT / "Push-T" / "train" / "train_critic.py",
        ]

        for path in scripts:
            text = path.read_text(encoding="utf-8")
            self.assertIn('if __name__ == "__main__":', text, msg=str(path))

    def test_legacy_numerical_behavior_is_documented(self) -> None:
        notes = (REPO_ROOT / "docs" / "reproducibility.md").read_text(encoding="utf-8")

        self.assertIn("benchmark-compatible guidance calibration", notes.lower())
        self.assertIn("checkpoint", notes.lower())
        self.assertIn("preserve", notes.lower())


if __name__ == "__main__":
    unittest.main()
