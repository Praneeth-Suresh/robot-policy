import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TrialScriptValidation(unittest.TestCase):
    def test_invalid_choices_are_rejected_before_loading_models(self):
        for script, variables in [
            ('Push-T/run_trial.py', {'METHOD': 'invalid'}),
            ('Robomimic/run_trial.py', {'TASK': 'invalid'}),
        ]:
            with self.subTest(script=script):
                env = os.environ.copy()
                env.update(variables)
                result = subprocess.run([sys.executable, str(ROOT / script)], env=env,
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('invalid', result.stderr.lower())
