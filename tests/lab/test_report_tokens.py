"""Brand token drift, contrast, accessible status marks, and copied asset contracts."""

import json
import re
import unittest
from pathlib import Path

__all__: list[str] = []

_ROOT = Path(__file__).parents[2]
_ASSETS = _ROOT / "guided_streaming_policy/lab/assets"
_BRAND = _ROOT / "docs/brand"
_AA_RATIO = 4.5
_SRGB_BREAKPOINT = 0.04045
_SRGB_DIVISOR = 12.92
_SRGB_OFFSET = 0.055
_SRGB_SCALE = 1.055
_SRGB_EXPONENT = 2.4
_CHANNEL_MAX = 255
_LUMINANCE_OFFSET = 0.05
_LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)


def _luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / _CHANNEL_MAX for index in (1, 3, 5)]
    linear = [
        channel / _SRGB_DIVISOR
        if channel <= _SRGB_BREAKPOINT
        else ((channel + _SRGB_OFFSET) / _SRGB_SCALE) ** _SRGB_EXPONENT
        for channel in channels
    ]
    return sum(channel * weight for channel, weight in zip(linear, _LUMINANCE_WEIGHTS, strict=True))


class ReportTokenTests(unittest.TestCase):
    def test_color_tokens_and_variables(self) -> None:
        tokens = json.loads((_BRAND / "tokens.json").read_text())
        css = (_ASSETS / "report.css").read_text()
        roots = re.findall(r":root\s*\{([^}]+)\}", css)
        self.assertEqual(len(roots), 2)
        for mode, root in zip(("light", "dark"), roots, strict=True):
            properties = dict(re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", root))
            for key, value in tokens["color"][mode].items():
                self.assertEqual(properties[f"--c-{key}"], value)
        defined = set(re.findall(r"(--c-[\w-]+)\s*:", css))
        self.assertLessEqual(set(re.findall(r"var\((--c-[\w-]+)\)", css)), defined)
        self.assertNotRegex("".join(css.split("}")[2:]), r"#[0-9a-fA-F]{3,8}\b")

    def test_contrast(self) -> None:
        tokens = json.loads((_BRAND / "tokens.json").read_text())
        for mode, colors in tokens["color"].items():
            for foreground, background in tokens["contrast_pairs"]:
                first, second = sorted(
                    (_luminance(colors[foreground]), _luminance(colors[background]))
                )
                with self.subTest(mode=mode, foreground=foreground, background=background):
                    ratio = (second + _LUMINANCE_OFFSET) / (first + _LUMINANCE_OFFSET)
                    self.assertGreaterEqual(ratio, _AA_RATIO)

    def test_svg_copies_and_status_marks(self) -> None:
        for filename in ("logo-wordmark.svg", "logo-wordmark-dark.svg", "favicon.svg"):
            self.assertEqual((_ASSETS / filename).read_bytes(), (_BRAND / filename).read_bytes())
        source = (_ASSETS / "report.js").read_text() + (
            _ROOT / "guided_streaming_policy/lab/report.py"
        ).read_text()
        for mark in json.loads((_BRAND / "tokens.json").read_text())["status_marks"].values():
            self.assertIn(mark["word"], source)
            self.assertIn(mark["icon"], source)
