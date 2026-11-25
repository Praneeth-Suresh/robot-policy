"""Check documentation links, schema coverage and committed visual assets offline."""

import json
import re
import struct
import unittest
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

__all__: list[str] = []

_ROOT = Path(__file__).resolve().parents[2]
_README = _ROOT / "README.md"
_LAB_DOCS = _ROOT / "docs/lab"
_BRAND = _ROOT / "docs/brand"
_EXPECTED_PAGES = {
    "adding-a-task.md",
    "architecture.md",
    "baseline.md",
    "decisions.md",
    "limitations.md",
    "real-run.md",
    "reproducibility.md",
    "schema.md",
}
_LINK = re.compile(r'(!?)\[([^\]\n]*)\]\((<[^>\n]+>|[^\s)]+)(?:\s+"[^"\n]*")?\)')
_HEADING = re.compile(r"^#{1,6}\s+(.+?)(?:\s+#+)?\s*$")
_IMAGE_LIMIT_BYTES = 300 * 1024
_IMAGE_TOTAL_LIMIT_BYTES = 2 * 1024 * 1024
_SOCIAL_LIMIT_BYTES = 200 * 1024
_SOCIAL_SIZE_PIXELS = (1280, 640)
_CAPTION_NONEMPTY_LINES = 3
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_HEADER_BYTES = 24
_PNG_CHUNK_TYPE_OFFSET_BYTES = 12
_PNG_DIMENSIONS_OFFSET_BYTES = 16
_MANIFEST_OBJECTS = (
    "outcome",
    "geometry",
    "timebase",
    "provenance",
    "policy",
    "dataset",
    "resolved_config",
    "simulator",
)


def _prose_lines(path: Path) -> Iterator[tuple[int, str]]:
    fence = ""
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if not fence:
                fence = marker[1]
            elif line.strip().startswith(fence) and not line.strip().strip(fence[0]):
                fence = ""
            continue
        if not fence:
            yield number, line


def _links(path: Path) -> Iterator[tuple[int, re.Match[str]]]:
    for number, line in _prose_lines(path):
        for match in _LINK.finditer(line):
            yield number, match


def _headings(path: Path) -> list[str]:
    return [match[1] for _, line in _prose_lines(path) if (match := _HEADING.match(line))]


def _heading_slugs(path: Path) -> set[str]:
    slugs: set[str] = set()
    for heading in _headings(path):
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
        text = re.sub(r"<[^>]*>", "", text).lower()
        base = "".join(char for char in text if char.isalnum() or char in " -")
        base = base.replace(" ", "-")
        slug, suffix = base, 0
        while slug in slugs:
            suffix += 1
            slug = f"{base}-{suffix}"
        slugs.add(slug)
    return slugs


def _nested_properties(node: dict[str, Any], prefix: str) -> Iterator[tuple[str, str]]:
    for name, child in node.get("properties", {}).items():
        location = f"{prefix}.{name}"
        yield location, name
        yield from _nested_properties(child, location)
    if "items" in node:
        yield from _nested_properties(node["items"], prefix + "[]")


class DocumentationTests(unittest.TestCase):
    def test_relative_links_and_anchors_resolve(self) -> None:
        for source in (_README, *sorted(_LAB_DOCS.glob("*.md"))):
            for line, link in _links(source):
                target = urlsplit(link[3].strip("<>"))
                if target.scheme in ("http", "https", "mailto"):
                    continue
                location = f"{source.relative_to(_ROOT)}:{line}: {link[3]}"
                with self.subTest(location=location):
                    path = source.parent / unquote(target.path) if target.path else source
                    self.assertTrue(path.exists(), f"{location}: missing link target")
                    if target.fragment and path.suffix.lower() == ".md":
                        self.assertIn(
                            unquote(target.fragment),
                            _heading_slugs(path),
                            f"{location}: missing heading anchor",
                        )

    def test_documentation_inventory_and_readme_links(self) -> None:
        pages = {path.name for path in _LAB_DOCS.glob("*.md")}
        self.assertEqual(pages, _EXPECTED_PAGES)
        linked = {
            (_README.parent / unquote(urlsplit(link[3]).path)).resolve()
            for _, link in _links(_README)
            if not link[1]
        }
        for name in sorted(pages):
            with self.subTest(page=name):
                self.assertIn(
                    (_LAB_DOCS / name).resolve(),
                    linked,
                    f"README.md: missing link to docs/lab/{name}",
                )

    def test_png_inventory_and_size_budgets(self) -> None:
        images = sorted((_ROOT / "docs/img").glob("*.png"))
        self.assertTrue(images, "docs/img: no PNG images found")
        references = {
            (_ROOT / unquote(urlsplit(link[3]).path)).resolve()
            for _, link in _links(_README)
            if link[1]
        }
        for image in images:
            with self.subTest(image=image.name):
                self.assertIn(
                    image.resolve(),
                    references,
                    f"README.md: unreferenced image {image.relative_to(_ROOT)}",
                )
                self.assertLess(
                    image.stat().st_size,
                    _IMAGE_LIMIT_BYTES,
                    f"{image.relative_to(_ROOT)}: PNG must be < 300 KB",
                )
        self.assertLess(
            sum(image.stat().st_size for image in images),
            _IMAGE_TOTAL_LIMIT_BYTES,
            "docs/img: PNG total must be < 2 MB",
        )

    def test_readme_image_alt_text_and_captions(self) -> None:
        lines = _README.read_text(encoding="utf-8").splitlines()
        for number, image in _links(_README):
            if not image[1]:
                continue
            location = f"README.md:{number}: {image[3]}"
            with self.subTest(location=location):
                self.assertTrue(image[2].strip(), f"{location}: empty image alt text")
                following = [line.strip() for line in lines[number:] if line.strip()]
                captions = [
                    line
                    for line in following[:_CAPTION_NONEMPTY_LINES]
                    if line.startswith("*") and line.endswith("*") and not line.startswith("**")
                ]
                self.assertTrue(
                    captions,
                    f"{location}: missing italic caption within "
                    f"{_CAPTION_NONEMPTY_LINES} non-empty lines",
                )
                caption = " ".join(captions)
                filename = Path(urlsplit(image[3]).path).name
                if filename.startswith("quickstart-report"):
                    self.assertIn("ynthetic", caption, f"{location}: missing synthetic label")
                if filename.startswith("real-"):
                    self.assertRegex(
                        caption, r"commit\s+`[0-9a-fA-F]{7,}`", f"{location}: missing commit hash"
                    )
                    self.assertIn("Verified in simulation", caption, location)

    def test_schema_reference_covers_all_properties(self) -> None:
        page = _LAB_DOCS / "schema.md"
        prose = "\n".join(line for _, line in _prose_lines(page))
        documented = set(re.findall(r"`([^`\n]+)`", prose))
        schemas = sorted((_ROOT / "schemas").glob("*.schema.json"))
        self.assertTrue(schemas, "schemas: no JSON schemas found")
        for path in schemas:
            properties = json.loads(path.read_text(encoding="utf-8"))["properties"]
            required = [(name, name) for name in properties]
            if path.name == "episode_manifest.schema.json":
                for name in _MANIFEST_OBJECTS:
                    required.extend(_nested_properties(properties[name], name))
            missing = [location for location, name in required if name not in documented]
            with self.subTest(schema=path.name):
                self.assertEqual(
                    len(missing),
                    0,
                    f"docs/lab/schema.md: missing backticked "
                    f"properties from {path.name}: {', '.join(missing)}",
                )

    def test_brand_file_inventory(self) -> None:
        guide = (_BRAND / "BRAND.md").read_text(encoding="utf-8")
        for path in sorted(_BRAND.iterdir()):
            if path.is_file() and path.name != "BRAND.md":
                with self.subTest(file=path.name):
                    self.assertIn(
                        f"`{path.name}`", guide, f"docs/brand/BRAND.md: missing file {path.name}"
                    )

    def test_brand_svgs_are_xml_with_titles(self) -> None:
        svgs = sorted(_BRAND.glob("*.svg"))
        self.assertTrue(svgs, "docs/brand: no SVG files found")
        for path in svgs:
            with self.subTest(file=path.name):
                root = ET.parse(path).getroot()
                self.assertEqual(root.tag.rsplit("}", 1)[-1], "svg")
                titles = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "title"]
                self.assertTrue(titles, f"docs/brand/{path.name}: missing SVG title")
                self.assertTrue(
                    any("".join(title.itertext()).strip() for title in titles),
                    f"docs/brand/{path.name}: empty SVG title",
                )

    def test_social_preview_png_dimensions_and_size(self) -> None:
        path = _BRAND / "social-preview.png"
        with path.open("rb") as stream:
            header = stream.read(_PNG_HEADER_BYTES)
        self.assertEqual(header[: len(_PNG_SIGNATURE)], _PNG_SIGNATURE)
        self.assertEqual(header[_PNG_CHUNK_TYPE_OFFSET_BYTES:_PNG_DIMENSIONS_OFFSET_BYTES], b"IHDR")
        size = struct.unpack(">II", header[_PNG_DIMENSIONS_OFFSET_BYTES:_PNG_HEADER_BYTES])
        self.assertEqual(size, _SOCIAL_SIZE_PIXELS)
        self.assertLess(
            path.stat().st_size,
            _SOCIAL_LIMIT_BYTES,
            "docs/brand/social-preview.png: must be < 200 KB",
        )

    def test_brand_color_keys_match_between_themes(self) -> None:
        tokens = json.loads((_BRAND / "tokens.json").read_text(encoding="utf-8"))
        self.assertEqual(set(tokens["color"]["light"]), set(tokens["color"]["dark"]))

    def test_required_decision_and_real_run_headings(self) -> None:
        required = {
            "decisions.md": ("Validation run choice", "Brand rationale", "Removal manifest"),
            "real-run.md": ("Setup", "Results", "Every registered policy, one seed"),
        }
        for name, titles in required.items():
            actual = _headings(_LAB_DOCS / name)
            for title in titles:
                with self.subTest(page=name, heading=title):
                    self.assertTrue(
                        any(
                            heading == title or heading.startswith(title + " (")
                            for heading in actual
                        ),
                        f"docs/lab/{name}: missing heading {title!r}",
                    )
