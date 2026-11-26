"""Trace discovery, content hashes and immutable report output."""

import tempfile
import unittest
from pathlib import Path

from guided_streaming_policy.lab import adapters, schema
from tests.lab.helpers import demo_episode, write_traces

__all__: list[str] = []


class TraceAdapterTests(unittest.TestCase):
    def test_read_traces_single_and_sorted_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_traces(root, [demo_episode("b"), demo_episode("a")])
            self.assertTrue(adapters.is_trace_dir(root))
            self.assertTrue(adapters.is_trace_dir(root / "a"))
            self.assertEqual([e.manifest["episode_id"] for e in adapters.read_traces(root)],
                             ["a", "b"])
            self.assertEqual(list(adapters.read_traces(root / "a")), [demo_episode("a")])

    def test_missing_empty_and_file_are_not_traces(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            file = root / "file"
            file.write_text("text")
            for path in (root, root / "missing", file):
                with self.subTest(path=path):
                    self.assertFalse(adapters.is_trace_dir(path))
                    with self.assertRaises(schema.TraceReadError):
                        list(adapters.read_traces(path))

    def test_directory_checksum_is_relative_sorted_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "nested").mkdir()
            (root / "z").write_bytes(b"z")
            (root / "nested" / "a").write_bytes(b"a")
            expected = schema.sha256_json([
                ["nested/a", schema.sha256_bytes(b"a")], ["z", schema.sha256_bytes(b"z")]])
            self.assertEqual(adapters.path_checksum(root), expected)
            self.assertEqual(adapters.path_checksum(root / "z"), schema.sha256_bytes(b"z"))
            (root / "z").write_bytes(b"changed")
            self.assertNotEqual(adapters.path_checksum(root), expected)

    def test_report_write_creates_parents_and_preserves_existing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nested" / "doctor.json"
            adapters.write_report_file(path, "é\n")
            before = path.stat().st_mtime_ns
            adapters.write_report_file(path, "é\n")
            self.assertEqual(path.stat().st_mtime_ns, before)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                adapters.write_report_file(path, "different\n")
            self.assertEqual(path.read_bytes(), "é\n".encode())
