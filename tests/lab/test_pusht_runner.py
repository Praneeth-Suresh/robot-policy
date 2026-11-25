"""Recording proxy tests and optional native reset / explicit opt-in model trial."""

import importlib
import importlib.util
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

from guided_streaming_policy.lab.tasks.base import TrialCrashed, TrialRequest
from guided_streaming_policy.lab.tasks.pusht import runner
from guided_streaming_policy.lab.tasks.pusht.adapter import PushTTask

__all__: list[str] = []
_ROOT = Path(__file__).resolve().parents[2]
_DT_SECONDS = 0.1
_MIN_DISTANCE_TOLERANCE_PX = 1e-3
_NATIVE_KEYS = {"guided_success", "guided_collision", "guided_reward", "guided_min_dist"}
_ENV_DEPS = ("pymunk", "pygame", "gym", "cv2", "shapely", "skimage")
_HAS_ENV_DEPS = all(importlib.util.find_spec(name) is not None for name in _ENV_DEPS)


class FakeEnv:
    window_size = 512

    def __init__(self) -> None:
        self.obs = [1.0, 2.0, 3.0, 4.0, 0.5]

    def reset(self) -> tuple[list[float], dict[str, Any]]:
        self.obs = [1.0, 2.0, 3.0, 4.0, 0.5]
        return self.obs, {"reset": True}

    def step(self, _action: Any) -> tuple[list[float], float, bool, bool, dict[str, Any]]:
        self.obs[0] += 10.0
        return self.obs, 0.75, True, True, {"unused": True}


class RecordingEnvTests(unittest.TestCase):
    def test_failed_reset_has_no_segment_successful_reset_does(self) -> None:
        proxy = runner.RecordingEnv(FakeEnv(), _DT_SECONDS)
        with self.assertRaises(TypeError):
            proxy.reset(seed=42)
        self.assertEqual(len(proxy.segments), 0)
        obs, info = proxy.reset()
        self.assertEqual(obs, [1.0, 2.0, 3.0, 4.0, 0.5])
        self.assertEqual(info, {"reset": True})
        self.assertEqual(len(proxy.segments), 1)
        proxy.reset()
        self.assertEqual(len(proxy.segments), 2)
        self.assertEqual(proxy.window_size, 512)

    def test_steps_capture_before_observation_action_reward_and_flags(self) -> None:
        proxy = runner.RecordingEnv(FakeEnv(), _DT_SECONDS)
        proxy.reset()
        action = [9.0, 8.0]
        result = proxy.step(action)
        action[0] = -100.0
        proxy.step([7.0, 6.0])
        segment = proxy.segments[0]
        self.assertEqual(segment.reset_obs, (1.0, 2.0, 3.0, 4.0, 0.5))
        self.assertEqual(segment.steps[0], {
            "step": 0, "t": 0.0, "obs": {"agent_pos": [1.0, 2.0],
            "block_pos": [3.0, 4.0], "block_angle": 0.5},
            "action_executed": {"target": [9.0, 8.0]}, "reward": 0.75})
        self.assertEqual(segment.steps[1]["obs"]["agent_pos"], [11.0, 2.0])
        self.assertEqual(segment.steps[1]["t"], _DT_SECONDS)
        self.assertEqual(result[1:], (0.75, True, True, {"unused": True}))
        self.assertTrue(segment.terminated)
        self.assertTrue(segment.truncated)

    def test_fingerprint_rounding_stability_and_sensitivity(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 0.5]
        fingerprint = runner.reset_fingerprint(values)
        self.assertEqual(fingerprint, runner.reset_fingerprint(tuple(values)))
        values[0] += 0.00000001
        self.assertEqual(fingerprint, runner.reset_fingerprint(values))
        values[-1] += 0.001
        self.assertNotEqual(fingerprint, runner.reset_fingerprint(values))

    def test_import_works_without_third_party_packages(self) -> None:
        code = "import guided_streaming_policy.lab.tasks.pusht.runner"
        result = subprocess.run([sys.executable, "-S", "-c", code], cwd=_ROOT,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


class RunnerHookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = ModuleType("fake_inference")
        self.original = FakeEnv()
        self.module.env = self.original
        self.module.stats = {"action": {}}
        self.module.unnormalize_data = lambda _pos, _stats: [100.0, 200.0]
        self.request = TrialRequest("pusht/naive", {"max_guidance_scale": 6.0}, "static", 0)

    def invoke(self) -> Any:
        # Bypass the research module's expensive import, retaining the real hook code.
        with patch.object(runner.importlib, "import_module", return_value=self.module):
            return runner.run_pusht_trial(self.request, _ROOT)

    def simulate(self, count: int, *, crash: bool = False) -> Any:
        def run(**kwargs: Any) -> tuple[dict[str, Any], list[Any]]:
            self.assertEqual(Path.cwd(), _ROOT / "Push-T")
            self.assertEqual(kwargs, {"seed": 0, "fixed_obstacle_pos": [-0.3, -0.5],
                                      "save_gif": False, "max_guidance_scale": 6.0})
            for index in range(count):
                try:
                    self.module.env.reset(seed=0)
                except TypeError:
                    pass
                self.module.env.reset()
                self.module.env.step([float(index), 2.0])
            if crash:
                raise RuntimeError("inference failed")
            return {"guided_success": True, "guided_collision": False,
                    "guided_reward": 0.75, "guided_min_dist": 42.0}, []
        return run

    def test_third_segment_metrics_geometry_and_restoration(self) -> None:
        self.module.run_naive_inference = self.simulate(3)
        cwd = Path.cwd()
        record = self.invoke()
        self.assertEqual(Path.cwd(), cwd)
        self.assertIs(self.module.env, self.original)
        self.assertEqual(record.steps[0]["action_executed"]["target"], [2.0, 2.0])
        self.assertEqual(set(record.native), _NATIVE_KEYS)
        self.assertTrue(record.native_success)
        self.assertTrue(record.terminated)
        self.assertFalse(record.truncated)
        self.assertEqual(record.reason, "coverage_reached")
        self.assertEqual(record.geometry["obstacles"][0]["position"], [100.0, 200.0])
        self.assertEqual(record.geometry["obstacles"][0]["radius"], 15.36)
        self.assertEqual(record.geometry["block"]["pose_fields"], ["block_pos", "block_angle"])
        self.assertEqual(record.geometry["block"]["shapes"][0],
                         [[-60.0, 30.0], [60.0, 30.0], [60.0, 0.0], [-60.0, 0.0]])
        self.assertIsNotNone(record.reset_state_fingerprint)

    def test_crash_and_short_call_retain_only_third_segment(self) -> None:
        cwd = Path.cwd()
        for count, crash in ((2, False), (2, True), (3, True)):
            with self.subTest(count=count, crash=crash):
                self.module.run_naive_inference = self.simulate(count, crash=crash)
                with self.assertRaises(TrialCrashed) as caught:
                    self.invoke()
                partial = caught.exception.record
                if count == 3:
                    self.assertEqual(len(partial.steps), 1)
                    self.assertIsNone(partial.native_success)
                else:
                    self.assertIsNone(partial)
                self.assertIs(self.module.env, self.original)
                self.assertEqual(Path.cwd(), cwd)

    def test_adapter_delegates_and_only_supports_static_condition(self) -> None:
        task = PushTTask(_ROOT)
        self.assertEqual(task.conditions(), ("static",))
        with patch.object(runner, "run_pusht_trial") as run:
            self.assertIs(task.run_trial(self.request), run.return_value)
        run.assert_called_once_with(self.request, _ROOT)

    def test_step_cap_is_truncation_and_unfinished_call_has_explicit_reason(self) -> None:
        for count, reason in ((runner.MAX_STEPS, "max_steps"), (1, "ended_without_termination")):
            def run(*, step_count: int = count, **_kwargs: Any) -> tuple[dict[str, Any], list[Any]]:
                for _ in range(3):
                    self.module.env.reset()
                    for _ in range(step_count):
                        self.module.env.step([1.0, 2.0])
                    self.module.env.segments[-1].terminated = False
                return {"guided_success": False, "guided_collision": True,
                        "guided_reward": 0.0, "guided_min_dist": 0.0}, []
            self.module.run_naive_inference = run
            record = self.invoke()
            self.assertEqual(record.reason, reason)
            self.assertEqual(record.truncated, count == runner.MAX_STEPS)

    def test_interrupt_restores_environment_and_directory(self) -> None:
        def interrupt(**_kwargs: Any) -> Any:
            raise KeyboardInterrupt
        self.module.run_naive_inference = interrupt
        cwd = Path.cwd()
        with self.assertRaises(KeyboardInterrupt):
            self.invoke()
        self.assertEqual(Path.cwd(), cwd)
        self.assertIs(self.module.env, self.original)


@unittest.skipUnless(_HAS_ENV_DEPS, "native simulator dependencies absent")
class NativeResetTests(unittest.TestCase):
    def test_seeded_reset_fingerprints(self) -> None:
        sys.path.insert(0, str(_ROOT / "Push-T"))
        self.addCleanup(sys.path.remove, str(_ROOT / "Push-T"))
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        env = importlib.import_module("core.pusht_env").PushTEnv()
        self.addCleanup(env.close)
        fingerprints = []
        for seed in (0, 0, 1):
            env.seed(seed)
            obs, _info = env.reset()
            fingerprints.append(runner.reset_fingerprint(obs))
        self.assertEqual(fingerprints[0], fingerprints[1])
        self.assertNotEqual(fingerprints[0], fingerprints[2])


@unittest.skipUnless(os.environ.get("GSP_LAB_REAL_RUN") == "1", "explicit model trial opt-in only")
class RealTrialTests(unittest.TestCase):
    def test_naive_seed_zero(self) -> None:
        record = PushTTask(_ROOT).run_trial(TrialRequest(
            "pusht/naive", {"max_guidance_scale": 6.0}, "static", 0))
        self.assertGreater(len(record.steps), 0)
        self.assertLessEqual(len(record.steps), runner.MAX_STEPS)
        self.assertEqual(set(record.native), _NATIVE_KEYS)
        self.assertIsNotNone(record.reset_state_fingerprint)
        obstacle = record.geometry["obstacles"][0]["position"]
        minimum = min(math.dist(step["action_executed"]["target"], obstacle)
                      for step in record.steps)
        self.assertAlmostEqual(record.native["guided_min_dist"], minimum,
                               delta=_MIN_DISTANCE_TOLERANCE_PX)
