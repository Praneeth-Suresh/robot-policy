"""Pure separation measurements and independent native outcomes."""

import copy
import dataclasses
import unittest

from guided_streaming_policy.lab import checks, schema
from tests.lab.helpers import rollout_episode

__all__: list[str] = []

_SIGNAL = "action_executed.target"


def _episode() -> schema.Episode:
    episode = rollout_episode("geometry", n_steps=3, obstacle={
        "id": "point", "kind": "point", "position": [0.0, 0.0], "radius": 0.0,
        "units": "px", "frame": "window", "source": "fixture"})
    for step, position in zip(episode.steps, ([3.0, 4.0], [0.0, 2.0], [0.0, 1.0]), strict=True):
        step["action_executed"]["target"] = position
    return episode


class CheckTests(unittest.TestCase):
    def test_distances_and_no_mutation(self) -> None:
        episode = _episode()
        before = copy.deepcopy(episode)
        series = checks.separation_series(episode, _SIGNAL)
        self.assertEqual(series, checks.SeparationSeries((0.0, 0.1, 0.2), (5.0, 2.0, 1.0), None))
        self.assertEqual(checks.measurement_coverage(series), 1.0)
        self.assertEqual(episode, before)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            series.values = ()

    def test_multiple_obstacles_nd_positions_and_generic_signal(self) -> None:
        episode = _episode()
        episode.manifest["geometry"]["obstacles"] = [
            {"position": [0, 0, 0]}, {"position": [1, 2, 1]}]
        for step in episode.steps:
            step["obs"]["arbitrary"] = [1, 2, 3]
        self.assertEqual(checks.separation_series(episode, "obs.arbitrary").values, (2, 2, 2))

    def test_missing_nonfinite_dimension_mismatch_and_recorded_times(self) -> None:
        for invalid in (None, [float("nan"), 0], [1], [True, 0], [[1], [2]]):
            with self.subTest(invalid=invalid):
                episode = _episode()
                episode.steps[1]["action_executed"]["target"] = invalid
                episode.steps[1]["t"] = float("inf")
                del episode.steps[2]["action_executed"]["target"]
                del episode.steps[2]["t"]
                series = checks.separation_series(episode, _SIGNAL)
                self.assertEqual(series.values, (5, None, None))
                self.assertEqual(series.t, (0, None, None))
                self.assertEqual(checks.measurement_coverage(series), 1 / 3)

    def test_unknown_reasons(self) -> None:
        for change, reason in (
            (lambda e: e.manifest.pop("geometry"), "no obstacle geometry recorded"),
            (lambda e: e.manifest["fields_present"].remove("geometry"),
             "no obstacle geometry recorded"),
            (lambda e: e.manifest["fields_present"].remove("action_executed"),
             f"signal {_SIGNAL} was not collected"),
            (lambda e: e.manifest["geometry"].update(obstacles=[]), "no obstacles in geometry"),
        ):
            episode = _episode()
            change(episode)
            series = checks.separation_series(episode, _SIGNAL)
            self.assertEqual(series.unknown_reason, reason)
            self.assertEqual(series.values, (None, None, None))
        episode = _episode()
        series = checks.separation_series(schema.Episode(episode.manifest, ()), _SIGNAL)
        self.assertEqual(series.unknown_reason, "episode has no steps")
        self.assertEqual(checks.measurement_coverage(series), 0.0)
        result = checks.evaluate_separation(series, threshold=3, tolerance=0, min_coverage=0)
        self.assertEqual((result.verdict, result.reason), ("unknown", "episode has no steps"))

    def test_pass_fail_and_first_violation(self) -> None:
        series = checks.separation_series(_episode(), _SIGNAL)
        result = checks.evaluate_separation(series, threshold=3, tolerance=0, min_coverage=1)
        self.assertEqual(result.as_dict(), {
            "verdict": "fail", "minimum": 1.0, "minimum_step": 2,
            "first_violation_step": 1, "first_violation_time": 0.1,
            "coverage": 1.0, "steps": 3, "reason": None})
        self.assertEqual(checks.evaluate_separation(
            series, threshold=1, tolerance=0, min_coverage=1).verdict, "pass")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.verdict = "pass"

    def test_tolerance_boundary(self) -> None:
        for value, verdict in ((4.5, "pass"), (4.499, "fail")):
            series = checks.SeparationSeries((0.0,), (value,), None)
            self.assertEqual(checks.evaluate_separation(
                series, threshold=5, tolerance=0.5, min_coverage=1).verdict, verdict)

    def test_unknown_coverage_retains_violation_and_missing_time(self) -> None:
        series = checks.SeparationSeries((None, 0.1), (1.0, None), None)
        result = checks.evaluate_separation(series, threshold=3, tolerance=0, min_coverage=0.75)
        self.assertEqual(result.verdict, "unknown")
        self.assertIn("coverage", result.reason)
        self.assertEqual((result.minimum, result.minimum_step, result.first_violation_step,
                          result.first_violation_time, result.coverage), (1, 0, 0, None, 0.5))
        empty = checks.SeparationSeries((), (), None)
        result = checks.evaluate_separation(empty, threshold=3, tolerance=0, min_coverage=1)
        self.assertEqual((result.verdict, result.minimum, result.minimum_step),
                         ("unknown", None, None))

    def test_success_and_constraint_are_independent(self) -> None:
        episode = _episode()
        episode.manifest["outcome"].update(native_success=True, truncated=False)
        self.assertIs(checks.native_success(episode), True)
        self.assertIsNone(checks.failure_category(episode))
        self.assertEqual(checks.evaluate_separation(checks.separation_series(episode, _SIGNAL),
                         threshold=3, tolerance=0, min_coverage=1).verdict, "fail")

    def test_failure_categories_in_priority_order(self) -> None:
        for outcome, category in (
            ({"reason": "crashed", "native_success": None}, "crashed"),
            ({"reason": "complete", "native_success": False, "truncated": True}, "truncated"),
            ({"native_success": False, "truncated": False}, "native_fail"),
            ({"native_success": True, "truncated": False}, None),
            ({"native_success": None, "truncated": False}, None),
        ):
            episode = rollout_episode("outcome", outcome=outcome)
            self.assertEqual(checks.failure_category(episode), category)
            self.assertIs(checks.native_success(episode), outcome["native_success"])
