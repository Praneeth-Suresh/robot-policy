"""Task contracts, policy configuration, readiness and deliberate pin review."""

import importlib.metadata
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from guided_streaming_policy.lab import adapters, schema
from guided_streaming_policy.lab.tasks.base import PolicySpec, PolicyStatus, TunableSpec
from guided_streaming_policy.lab.tasks.pusht.adapter import PushTTask
from guided_streaming_policy.lab.tasks.pusht.pins import compute_pins, pin_file_document
from guided_streaming_policy.lab.tasks.registry import get_task, task_names

__all__: list[str] = []

ROOT = Path(__file__).resolve().parents[2]
POLICY_NAMES = {"pusht/naive", "pusht/ccg", "pusht/steg", "pusht/dp", "pusht/flow"}
DEPENDENCIES = ("torch", "numpy", "pymunk", "zarr", "gym", "diffusers")
PIN_REVIEW = "pin changed: review lab/pins.json deliberately"


class TaskTests(unittest.TestCase):
    def test_registered_policies_and_unknown_names(self) -> None:
        adapter = get_task("pusht")
        self.assertEqual(task_names(), ("pusht",))
        self.assertEqual({spec.name for spec in adapter.policies()}, POLICY_NAMES)
        with self.assertRaisesRegex(KeyError, "pusht"):
            get_task("missing")
        with self.assertRaisesRegex(KeyError, "pusht/naive"):
            adapters.find_policy(adapter, "missing")
        self.assertFalse(any(name.startswith("inference.static.") for name in sys.modules))

    def test_config_defaults_overrides_and_hash(self) -> None:
        adapter = get_task("pusht")
        expected = {
            "pusht/naive": {"max_guidance_scale": 6.0},
            "pusht/ccg": {"max_guidance_scale": 4.0},
            "pusht/steg": {"guidance_scale": 1.5, "N_ensemble": 64},
            "pusht/dp": {"guidance_scale": 2.0, "guidance_sign": 1.0, "grad_clip_norm": 1.0},
            "pusht/flow": {"guidance_scale": 5.0},
        }
        for spec in adapter.policies():
            with self.subTest(policy=spec.name):
                resolved = adapters.resolve_policy_config(spec, {})
                self.assertEqual(
                    resolved,
                    {
                        "policy": spec.name,
                        "entry_point": spec.entry_point,
                        "settings": expected[spec.name],
                    },
                )
                self.assertEqual(adapters.config_sha256(resolved), schema.sha256_json(resolved))
        spec = adapters.find_policy(adapter, "pusht/steg")
        resolved = adapters.resolve_policy_config(spec, {"guidance_scale": 3, "N_ensemble": 8})
        self.assertEqual(resolved["settings"], {"guidance_scale": 3, "N_ensemble": 8})
        self.assertEqual(adapters.resolve_policy_config(spec, {})["settings"], expected[spec.name])

    def test_rejected_override_names_and_types(self) -> None:
        spec = adapters.find_policy(get_task("pusht"), "pusht/steg")
        with self.assertRaisesRegex(ValueError, "unknown.*N_ensemble.*guidance_scale"):
            adapters.resolve_policy_config(spec, {"unknown": 1})
        for key, value in (
            ("N_ensemble", 1.5),
            ("N_ensemble", True),
            ("guidance_scale", True),
            ("guidance_scale", "2"),
        ):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                adapters.resolve_policy_config(spec, {key: value})
        other = PolicySpec(
            "fixture",
            "summary",
            "paper",
            "module:run",
            (),
            (TunableSpec("enabled", False, "bool"), TunableSpec("mode", "a", "string")),
        )
        self.assertEqual(
            adapters.resolve_policy_config(other, {"enabled": True, "mode": "b"})["settings"],
            {"enabled": True, "mode": "b"},
        )
        for key, value in (("enabled", 1), ("mode", False)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                adapters.resolve_policy_config(other, {key: value})

    def test_checkpoint_and_config_pins(self) -> None:
        adapter = get_task("pusht")
        document = adapters.load_pins(adapter.pins_path())
        self.assertEqual(document["schema_version"], schema.SCHEMA_VERSION, PIN_REVIEW)
        self.assertEqual(set(document["tasks"]), {"pusht"}, PIN_REVIEW)
        pins = document["tasks"]["pusht"]["policies"]
        self.assertEqual(set(pins), POLICY_NAMES, PIN_REVIEW)
        for spec in adapter.policies():
            with self.subTest(policy=spec.name):
                self.assertEqual(set(pins[spec.name]), {"checkpoints", "config_sha256"}, PIN_REVIEW)
                self.assertEqual(
                    set(pins[spec.name]["checkpoints"]), set(spec.checkpoints), PIN_REVIEW
                )
                self.assertEqual(
                    adapters.config_sha256(adapters.resolve_policy_config(spec, {})),
                    pins[spec.name]["config_sha256"],
                    PIN_REVIEW,
                )
            for checkpoint in spec.checkpoints:
                with self.subTest(policy=spec.name, checkpoint=checkpoint):
                    if not (ROOT / checkpoint).is_file():
                        self.skipTest(f"checkpoint absent: {checkpoint}")
                    self.assertEqual(
                        schema.sha256_file(ROOT / checkpoint),
                        pins[spec.name]["checkpoints"][checkpoint],
                        PIN_REVIEW,
                    )

    def test_status_reports_all_missing_causes_and_runnable(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("importlib.util.find_spec", return_value=None),
        ):
            root = Path(temporary)
            adapter = PushTTask(root)
            for status in adapter.policy_statuses():
                expected = [f"missing checkpoint {path}" for path in status.spec.checkpoints]
                expected += [
                    f"missing Python package {name} (install the pusht extra)"
                    for name in DEPENDENCIES
                ]
                self.assertEqual(status.status, "unavailable")
                self.assertEqual(status.reason, "; ".join(expected))
            for spec in adapter.policies():
                for checkpoint in spec.checkpoints:
                    path = root / checkpoint
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"fixture")
            with patch("importlib.util.find_spec", return_value=object()):
                for status in adapter.policy_statuses():
                    self.assertEqual((status.status, status.reason), ("runnable", None))

    def test_probe_schemas_and_missing_asset_pins(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("importlib.util.find_spec", return_value=None),
            patch(
                "importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError
            ),
        ):
            root = Path(temporary)
            adapter = PushTTask(root)
            probe = adapter.probe()
            self.assertEqual((probe.simulator_name, probe.simulator_version), ("pymunk", None))
            self.assertEqual(probe.dependencies, dict.fromkeys(DEPENDENCIES, False))
            self.assertEqual(len(probe.assets), 5)
            self.assertFalse(any(probe.assets.values()))
            pins = adapter.current_pins()
            self.assertEqual(pins, compute_pins(root))
            self.assertEqual(
                pin_file_document(root), {"schema_version": "0.1", "tasks": {"pusht": pins}}
            )
            self.assertEqual(pins["simulator"], {"name": "pymunk", "version": None})
            for policy in pins["policies"].values():
                self.assertTrue(all(value is None for value in policy["checkpoints"].values()))
        self.assertEqual(adapter.timebase(), {"unit": "s", "mode": "fixed_dt", "dt": 0.1})
        observations = adapter.observation_schema()["fields"]
        self.assertEqual(set(observations), {"agent_pos", "block_pos", "block_angle"})
        self.assertEqual(
            observations["agent_pos"],
            {"shape": [2], "dtype": "float64", "units": "px", "frame": "window_512_y_down"},
        )
        self.assertEqual(observations["block_pos"], observations["agent_pos"])
        self.assertEqual(
            observations["block_angle"],
            {"shape": [], "dtype": "float64", "units": "rad", "frame": "block_mod_2pi"},
        )
        self.assertEqual(
            adapter.action_schema(),
            {"convention": "absolute", "fields": {"target": observations["agent_pos"]}},
        )

    def test_policy_table_is_deterministic_and_displays_pins_and_reasons(self) -> None:
        spec = adapters.find_policy(get_task("pusht"), "pusht/naive")
        status = PolicyStatus(spec, "unavailable", "missing checkpoint fixture")
        pins = {
            "policies": {
                spec.name: {
                    "checkpoints": {spec.checkpoints[0]: "a" * 64},
                    "config_sha256": "b" * 64,
                }
            }
        }
        lines = adapters.format_policy_rows((status,), pins)
        self.assertEqual(lines[0].split(), ["NAME", "STATUS", "PIN", "PAPER"])
        self.assertIn("ckpt aaaaaaaaaaaa cfg bbbbbbbbbbbb", lines[1])
        self.assertEqual(lines[2], "  - missing checkpoint fixture")
        self.assertIn("no pin", adapters.format_policy_rows((status,), {})[1])
        self.assertEqual(lines, adapters.format_policy_rows((status,), pins))

    def test_load_pins_rejects_non_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pins.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                adapters.load_pins(path)
