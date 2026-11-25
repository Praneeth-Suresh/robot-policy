"""Comparison configuration validation and raw-file identity."""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from unittest.mock import patch

from guided_streaming_policy.lab import schema

__all__: list[str] = []

HAS_JSONSCHEMA = importlib.util.find_spec("jsonschema") is not None


def _document() -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "name": "fixture_comparison",
        "task": "fixture",
        "simulator_version": "7.2.0",
        "matrix": {"conditions": ["static"], "seeds": [0, 1, 2]},
        "variants": {
            "baseline": {"policy": "fixture/dp"},
            "candidate": {"policy": "fixture/dp", "overrides": {"guidance_scale": 0.0}},
        },
        "intended_differences": ["overrides.guidance_scale"],
    }


class ComparisonConfigTests(unittest.TestCase):
    def _load(self, document: Any) -> schema.ComparisonConfig:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "comparison.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return schema.load_comparison_config(path)

    def _assert_invalid(self, document: Any, key: str) -> None:
        with self.assertRaises(schema.ConfigError) as caught:
            self._load(document)
        self.assertIn(key, str(caught.exception))

    def test_valid_config_and_raw_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "comparison.json"
            path.write_text(json.dumps(_document(), indent=4) + "\n", encoding="utf-8")
            config = schema.load_comparison_config(path)
            self.assertEqual(config.sha256, schema.sha256_file(path))
            self.assertNotEqual(config.sha256, schema.sha256_json(_document()))
        self.assertEqual(config.name, "fixture_comparison")
        self.assertEqual(config.task, "fixture")
        self.assertEqual(config.simulator_version, "7.2.0")
        self.assertEqual(config.conditions, ("static",))
        self.assertEqual(config.seeds, (0, 1, 2))
        self.assertEqual(config.variants["baseline"], schema.VariantSpec("fixture/dp", {}))
        self.assertEqual(config.variants["candidate"].overrides, {"guidance_scale": 0.0})
        self.assertEqual(config.intended_differences, ("overrides.guidance_scale",))
        with self.assertRaises(FrozenInstanceError):
            config.name = "changed"
        with self.assertRaises(FrozenInstanceError):
            config.variants["baseline"].policy = "changed"

    def test_top_level_keys_and_version(self) -> None:
        for key in _document():
            document = _document()
            del document[key]
            with self.subTest(missing=key):
                self._assert_invalid(document, key)
        self._assert_invalid({**_document(), "unknown": 0}, "unknown")
        self._assert_invalid({**_document(), "schema_version": "0.2"}, "schema_version")
        self._assert_invalid([], "comparison")
        for key in ("name", "task", "simulator_version"):
            for value in ("", "  ", None, 1):
                with self.subTest(key=key, value=value):
                    self._assert_invalid({**_document(), key: value}, key)

    def test_matrix_structure(self) -> None:
        for matrix, key in (
            ([], "matrix"),
            ({"seeds": [0]}, "matrix.conditions"),
            ({"conditions": ["static"]}, "matrix.seeds"),
            ({"conditions": ["static"], "seeds": [0], "extra": 1}, "matrix.extra"),
        ):
            with self.subTest(matrix=matrix):
                self._assert_invalid({**_document(), "matrix": matrix}, key)

    def test_invalid_conditions(self) -> None:
        for value, key in (
            ([], "matrix.conditions"),
            ("static", "matrix.conditions"),
            (["static", "static"], "matrix.conditions[1]"),
            (["static", ""], "matrix.conditions[1]"),
            ([False], "matrix.conditions[0]"),
        ):
            document = _document()
            document["matrix"]["conditions"] = value
            with self.subTest(value=value):
                self._assert_invalid(document, key)

    def test_invalid_seeds(self) -> None:
        for value, key in (
            ([], "matrix.seeds"),
            ("range(3)", "matrix.seeds"),
            ({"start": 0, "stop": 3}, "matrix.seeds"),
            ([0, 0], "matrix.seeds[1]"),
            ([0, -1], "matrix.seeds[1]"),
            ([0, True], "matrix.seeds[1]"),
            ([0, 1.0], "matrix.seeds[1]"),
            (["0"], "matrix.seeds[0]"),
        ):
            document = _document()
            document["matrix"]["seeds"] = value
            with self.subTest(value=value):
                self._assert_invalid(document, key)

    def test_variant_keys(self) -> None:
        self._assert_invalid({**_document(), "variants": []}, "variants")
        for name in ("baseline", "candidate"):
            document = _document()
            del document["variants"][name]
            self._assert_invalid(document, f"variants.{name}")
        document = _document()
        document["variants"]["other"] = {"policy": "fixture"}
        self._assert_invalid(document, "variants.other")

    def test_invalid_variant(self) -> None:
        for value, suffix in (
            ([], ""),
            ({}, ".policy"),
            ({"policy": ""}, ".policy"),
            ({"policy": False}, ".policy"),
            ({"policy": "fixture", "extra": 0}, ".extra"),
            ({"policy": "fixture", "overrides": []}, ".overrides"),
        ):
            document = _document()
            document["variants"]["baseline"] = value
            with self.subTest(value=value):
                self._assert_invalid(document, "variants.baseline" + suffix)

    def test_override_scalars(self) -> None:
        document = _document()
        document["variants"]["baseline"]["overrides"] = {
            "flag": True,
            "count": 3,
            "scale": 0.5,
            "label": "",
        }
        self.assertEqual(
            self._load(document).variants["baseline"].overrides,
            document["variants"]["baseline"]["overrides"],
        )
        for value in (None, [], {}, float("nan"), float("inf")):
            invalid = copy.deepcopy(document)
            invalid["variants"]["baseline"]["overrides"]["bad"] = value
            with self.subTest(value=value):
                self._assert_invalid(invalid, "variants.baseline.overrides.bad")

    def test_intended_differences(self) -> None:
        for value, key in (
            ("policy", "intended_differences"),
            (["policy", "policy"], "intended_differences[1]"),
            ([False], "intended_differences[0]"),
            (["unknown"], "intended_differences[0]"),
            (["overrides."], "intended_differences[0]"),
            (["overrides.missing"], "intended_differences[0]"),
        ):
            with self.subTest(value=value):
                self._assert_invalid({**_document(), "intended_differences": value}, key)
        for value in ([], ["policy"], ["policy", "overrides.guidance_scale"]):
            self.assertEqual(
                self._load({**_document(), "intended_differences": value}).intended_differences,
                tuple(value),
            )
        document = _document()
        document["variants"]["baseline"]["overrides"] = {"gain": 1}
        document["intended_differences"] = ["overrides.gain"]
        self.assertEqual(self._load(document).intended_differences, ("overrides.gain",))

    def test_unreadable_and_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "comparison.json"
            with self.assertRaises(schema.ConfigError):
                schema.load_comparison_config(path)
            with patch.object(Path, "read_bytes", side_effect=PermissionError("denied")):
                with self.assertRaises(schema.ConfigError):
                    schema.load_comparison_config(path)
            for content in (b"{", b"\xff", b"null"):
                path.write_bytes(content)
                with self.subTest(content=content), self.assertRaises(schema.ConfigError):
                    schema.load_comparison_config(path)

    @unittest.skipUnless(HAS_JSONSCHEMA, "jsonschema is optional")
    def test_json_schema(self) -> None:
        import jsonschema

        definition = schema.comparison_config_json_schema()
        jsonschema.Draft202012Validator.check_schema(definition)
        jsonschema.validate(_document(), definition)
        for document in (
            {**_document(), "unknown": 0},
            {**_document(), "matrix": {"conditions": ["static"], "seeds": [True]}},
        ):
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(document, definition)
