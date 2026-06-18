import unittest

from guided_streaming_policy.doctor import collect_environment_report, render_environment_report


class DoctorTests(unittest.TestCase):
    def test_environment_report_contains_core_entries(self) -> None:
        report = collect_environment_report()
        labels = {item.label for item in report}

        self.assertIn("python", labels)
        self.assertIn("repo-root", labels)
        self.assertIn("pusht-root", labels)
        self.assertIn("pusht/core", labels)
        self.assertIn("artifact-dir:data", labels)
        self.assertIn("python-dep:torch", labels)

    def test_render_environment_report_returns_text(self) -> None:
        rendered = render_environment_report()
        self.assertIsInstance(rendered, str)
        self.assertIn("python", rendered)


if __name__ == "__main__":
    unittest.main()
