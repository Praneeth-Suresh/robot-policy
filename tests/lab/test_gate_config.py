"""Strict gate configuration validation and byte provenance."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

from guided_streaming_policy.lab import schema
from tests.lab.helpers import gate_config, gate_document

__all__: list[str] = []


class GateConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "gate.json"

    def test_valid_config_and_frozen_byte_digest(self) -> None:
        config = gate_config(self.path)
        self.assertEqual(config.name, "toy-gate")
        self.assertEqual(config.sha256, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(config.rules[1].signal, "action_executed.target")
        with self.assertRaises(FrozenInstanceError):
            config.name = "changed"
        with self.assertRaises(FrozenInstanceError):
            config.rules[0].threshold = 0
        self.path.write_text(self.path.read_text() + "\n")
        self.assertNotEqual(schema.load_gate_config(self.path).sha256, config.sha256)

    def test_invalid_root_keys_and_values(self) -> None:
        cases: list[tuple[str, Any]] = [
            ("schema_version", "0.2"),
            ("name", ""),
            ("name", "  "),
            ("name", 2),
            ("min_completed_pairs", 0),
            ("min_completed_pairs", True),
            ("min_completed_pairs", 1.5),
            ("max_missing_pairs", -1),
            ("max_missing_pairs", False),
            ("max_incompatible_pairs", -1),
            ("max_incompatible_pairs", 1.5),
            ("rules", []),
            ("rules", {}),
            ("extra", 1),
        ]
        for key, value in cases:
            document = gate_document()
            document[key] = value
            with (
                self.subTest(key=key, value=value),
                self.assertRaisesRegex(schema.ConfigError, key),
            ):
                gate_config(self.path, document)
        for key in gate_document():
            document = gate_document()
            del document[key]
            with self.subTest(missing=key), self.assertRaisesRegex(schema.ConfigError, key):
                gate_config(self.path, document)

    def test_invalid_rules_name_the_key_path(self) -> None:
        cases: list[tuple[int, str, Any]] = [
            (0, "id", ""),
            (1, "id", "success_not_worse"),
            (0, "required", 1),
            (0, "compare", "baseline_minus_candidate"),
            (0, "threshold", True),
            (0, "threshold", float("inf")),
            (0, "threshold", float("nan")),
            (0, "units", 7),
            (0, "extra", 1),
            (0, "signal", "obs.position"),
            (1, "signal", "target"),
            (1, "signal", ".target"),
            (1, "signal", "action_executed."),
            (1, "signal", "a.b.c"),
            (1, "signal", 7),
            (1, "threshold", -1),
            (1, "tolerance", -1),
            (1, "tolerance", True),
            (1, "min_coverage", 0),
            (1, "min_coverage", 1.01),
            (1, "min_coverage", True),
            (1, "min_coverage", float("inf")),
            (1, "required", None),
            (1, "compare", "candidate_minus_baseline"),
        ]
        for index, key, value in cases:
            document = gate_document()
            document["rules"][index][key] = value
            self.path.write_text(json.dumps(document))
            with self.subTest(key=key, value=value), self.assertRaises(schema.ConfigError) as error:
                schema.load_gate_config(self.path)
            self.assertIn(f"rules[{index}].{key}", str(error.exception))

    def test_missing_rule_fields_and_unsupported_metric(self) -> None:
        for index, original in enumerate(gate_document()["rules"]):
            for key in original:
                document = gate_document()
                del document["rules"][index][key]
                with (
                    self.subTest(index=index, key=key),
                    self.assertRaises(schema.ConfigError) as err,
                ):
                    gate_config(self.path, document)
                self.assertIn(f"rules[{index}].{key}", str(err.exception))
        document = gate_document()
        document["rules"][0]["metric"] = "unavailable"
        with self.assertRaises(schema.ConfigError) as err:
            gate_config(self.path, document)
        for text in ("rules[0].metric", "native_success_rate", "obstacle_separation_min"):
            self.assertIn(text, str(err.exception))

    def test_bad_files_and_non_objects(self) -> None:
        with self.assertRaises(schema.ConfigError):
            schema.load_gate_config(self.path)
        for raw in (b"{", b"[]", b"\xff", b'{"rules":[false]}'):
            self.path.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(schema.ConfigError):
                schema.load_gate_config(self.path)
        document = gate_document()
        document["rules"][0] = None
        with self.assertRaisesRegex(schema.ConfigError, r"rules\[0\]"):
            gate_config(self.path, document)

    @unittest.skipUnless(importlib.util.find_spec("jsonschema"), "jsonschema is optional")
    def test_schema_rejects_unknown_and_wrong_metric_fields(self) -> None:
        import jsonschema

        definition = schema.gate_config_json_schema()
        jsonschema.Draft202012Validator.check_schema(definition)
        jsonschema.validate(gate_document(), definition)
        for index, key in ((0, "signal"), (1, "compare")):
            document = copy.deepcopy(gate_document())
            document["rules"][index][key] = "unexpected"
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(document, definition)
