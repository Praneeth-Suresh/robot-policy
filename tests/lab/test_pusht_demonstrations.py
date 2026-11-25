"""Native demonstration mapping, malformed inputs, laziness and source integrity."""

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from guided_streaming_policy.lab import schema
from guided_streaming_policy.lab.tasks.pusht.adapter import PushTTask

__all__: list[str] = []

ROOT = Path(__file__).resolve().parents[2]
HAS_NATIVE_DEPS = all(importlib.util.find_spec(name) is not None for name in ("zarr", "numpy"))
HAS_JSONSCHEMA = importlib.util.find_spec("jsonschema") is not None
CREATED_AT = "2026-01-01T00:00:00Z"
NATIVE_DATA = ROOT / "Push-T/data/pusht_cchi_v7_replay.zarr.zip"


def _fixture(path: Path, ends: tuple[int, ...] = (3, 5, 9), **changes: Any) -> None:
    import numpy as np
    import zarr

    arrays = {
        "data/state": np.arange(45, dtype=float).reshape(9, 5),
        "data/action": np.arange(18, dtype=float).reshape(9, 2),
        "meta/episode_ends": np.array(ends),
        "data/img": np.zeros((9, 2, 2, 3), dtype=np.uint8),
    }
    arrays["data/state"][1, 0] = float("nan")
    arrays["data/state"][2, 4] = float("inf")
    arrays.update(changes)
    with zarr.ZipStore(str(path), mode="w") as store:
        group = zarr.group(store=store)
        for name, array in arrays.items():
            if array is not None:
                group.create_dataset(name, data=array)


@unittest.skipUnless(HAS_NATIVE_DEPS, "zarr/numpy unavailable")
class DemonstrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "fixture.zarr.zip"
        _fixture(self.path)

    def test_adapter_maps_episodes_and_preserves_source_and_nonfinite_values(self) -> None:
        checksum = schema.sha256_file(self.path)
        episodes = list(PushTTask().read_demonstrations(self.path))
        self.assertEqual(
            [e.manifest["episode_id"] for e in episodes],
            ["demo_000000", "demo_000001", "demo_000002"],
        )
        self.assertEqual([len(e.steps) for e in episodes], [3, 2, 4])
        self.assertEqual([step["t"] for step in episodes[2].steps], [0.0, 0.1, 0.2, 0.3])
        self.assertEqual([step["step"] for step in episodes[2].steps], [0, 1, 2, 3])
        self.assertEqual(
            episodes[0].steps[0],
            {
                "step": 0,
                "t": 0.0,
                "obs": {"agent_pos": [0.0, 1.0], "block_pos": [2.0, 3.0], "block_angle": 4.0},
                "action_executed": {"target": [0.0, 1.0]},
            },
        )
        self.assertEqual(episodes[1].steps[0]["obs"]["agent_pos"], [15.0, 16.0])
        self.assertEqual(episodes[1].steps[0]["action_executed"]["target"], [6.0, 7.0])
        self.assertTrue(math.isnan(episodes[0].steps[1]["obs"]["agent_pos"][0]))
        self.assertTrue(math.isinf(episodes[0].steps[2]["obs"]["block_angle"]))
        for index, episode in enumerate(episodes):
            self._assert_manifest(episode.manifest, index, checksum)
            for step in episode.steps:
                json.dumps(step)
                self.assertIs(type(step["obs"]["block_angle"]), float)
        self.assertEqual(schema.sha256_file(self.path), checksum)

    def _assert_manifest(self, manifest: dict[str, Any], index: int, checksum: str) -> None:
        self.assertTrue(set(schema.required_manifest_fields("demonstration")) <= manifest.keys())
        self.assertEqual(manifest["source_episode_id"], str(index))
        self.assertEqual(manifest["schema_version"], "0.1")
        self.assertEqual(manifest["kind"], "demonstration")
        self.assertEqual(manifest["task"], "pusht")
        self.assertEqual(
            manifest["dataset"],
            {
                "name": "fixture",
                "revision": "unversioned (no revision metadata in the source)",
                "checksum": checksum,
            },
        )
        self.assertIsNone(manifest["split"])
        self.assertEqual(manifest["split_source"], "none")
        self.assertEqual(manifest["fields_present"], ["obs", "action_executed"])
        adapter = PushTTask()
        self.assertEqual(manifest["timebase"], adapter.timebase())
        self.assertEqual(manifest["observation_schema"], adapter.observation_schema())
        self.assertEqual(manifest["action_schema"], adapter.action_schema())
        self.assertEqual(manifest["provenance"]["command"], f"gsp lab doctor {self.path}")
        self.assertNotIn("media", manifest)
        self.assertNotIn("synthetic", manifest)

    @unittest.skipUnless(HAS_JSONSCHEMA, "jsonschema unavailable")
    def test_generated_manifests_and_steps_validate_against_committed_schemas(self) -> None:
        import jsonschema

        manifest_schema = json.loads((ROOT / "schemas/episode_manifest.schema.json").read_text())
        step_schema = json.loads((ROOT / "schemas/episode_step.schema.json").read_text())
        for episode in PushTTask().read_demonstrations(self.path):
            jsonschema.validate(episode.manifest, manifest_schema)
            for step in episode.steps:
                jsonschema.validate(step, step_schema)

    def test_reads_only_first_episode_until_iterator_advanced(self) -> None:
        import zarr

        from guided_streaming_policy.lab.tasks.pusht.demonstrations import read_demonstrations

        original = zarr.core.Array.__getitem__
        slices = []

        def track(array: Any, key: Any, *args: Any, **kwargs: Any) -> Any:
            if array.path in ("data/state", "data/action"):
                slices.append(key)
            return original(array, key, *args, **kwargs)

        with patch.object(zarr.core.Array, "__getitem__", track):
            iterator = read_demonstrations(self.path, created_at=CREATED_AT)
            self.assertEqual(slices, [])
            first = next(iterator)
            self.assertEqual(len(first.steps), 3)
            self.assertEqual(slices, [slice(0, 3), slice(0, 3)])
            iterator.close()
        self.assertEqual(first.manifest["provenance"]["created_at"], CREATED_AT)

    def test_hash_and_provenance_computed_once(self) -> None:
        from guided_streaming_policy.lab.tasks.pusht.demonstrations import read_demonstrations

        with (
            patch.object(schema, "sha256_file", wraps=schema.sha256_file) as digest,
            patch.object(schema, "make_provenance", wraps=schema.make_provenance) as provenance,
        ):
            episodes = list(read_demonstrations(self.path, created_at=CREATED_AT))
        digest.assert_called_once_with(self.path)
        provenance.assert_called_once_with(f"gsp lab doctor {self.path}", CREATED_AT)
        self.assertEqual(len(episodes), 3)

    def test_unreadable_paths(self) -> None:
        invalid = self.path.parent / "invalid.zarr.zip"
        invalid.write_text("not a zarr file")
        for path in (self.path.parent / "missing.zarr.zip", invalid):
            with self.subTest(path=path), self.assertRaisesRegex(schema.TraceReadError, str(path)):
                list(PushTTask().read_demonstrations(path))

    def test_invalid_arrays_and_boundaries(self) -> None:
        import numpy as np

        cases = [
            {"meta/episode_ends": np.array(ends)}
            for ends in (
                (3, 3, 9),
                (3, 2, 9),
                (0, 5, 9),
                (3, 5, 8),
                (3, 5, 10),
                (),
                (3.0, 5.0, 9.0),
                ((3, 5, 9),),
            )
        ]
        cases += [
            {"data/state": None},
            {"data/action": None},
            {"meta/episode_ends": None},
            {"data/state": np.zeros((9, 4))},
            {"data/action": np.zeros((9, 3))},
            {"data/action": np.zeros((8, 2))},
            {"data/state": np.zeros(45)},
        ]
        for index, changes in enumerate(cases):
            path = self.path.parent / f"invalid_{index}.zarr.zip"
            _fixture(path, **changes)
            with self.subTest(changes=changes), self.assertRaises(schema.TraceReadError):
                list(PushTTask().read_demonstrations(path))

    @unittest.skipUnless(NATIVE_DATA.is_file(), "native demonstration archive absent")
    def test_native_archive(self) -> None:
        before = schema.sha256_file(NATIVE_DATA)
        episodes = list(PushTTask().read_demonstrations(NATIVE_DATA))
        self.assertEqual(len(episodes), 206)
        self.assertEqual([len(e.steps) for e in episodes[:2]], [161, 118])
        self.assertEqual(sum(len(e.steps) for e in episodes), 25650)
        for actual, expected in zip(
            episodes[0].steps[0]["obs"]["agent_pos"],
            [222.0, 97.0],
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(episodes[0].steps[0]["action_executed"]["target"], [233.0, 71.0])
        self.assertEqual(schema.sha256_file(NATIVE_DATA), before)
