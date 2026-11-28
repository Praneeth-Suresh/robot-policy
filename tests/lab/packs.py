"""Throwaway task packs: fake distributions on sys.path and a scaffolded one-dimensional pack."""

import importlib
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

from guided_streaming_policy.lab import cli_tasks

__all__ = ["PACK", "PackSpec", "ScaffoldedPack", "installed_packs"]

PACK = "rail"


class PackSpec:
    """One fake distribution that registers one task through the gsp_lab.tasks group."""

    def __init__(
        self, task: str, *, distribution: str | None = None, version: str = "1.0",
        module: str | None = None, source: str | None = None, target: str | None = None,
    ) -> None:
        self.task = task
        self.distribution = distribution or f"fake-{task}"
        self.version = version
        self.module = module or f"fake_pack_{task}"
        self.source = source
        self.target = target or f"{self.module}:Pack"


def _write(root: Path, spec: PackSpec) -> None:
    info = root / f"{spec.distribution.replace('-', '_')}-{spec.version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {spec.distribution}\nVersion: {spec.version}\n",
        encoding="utf-8",
    )
    (info / "entry_points.txt").write_text(
        f"[gsp_lab.tasks]\n{spec.task} = {spec.target}\n", encoding="utf-8")
    if spec.source is not None:
        (root / f"{spec.module}.py").write_text(spec.source, encoding="utf-8")


@contextmanager
def installed_packs(*specs: PackSpec) -> Iterator[Path]:
    """Make the packs discoverable on sys.path for the duration of the block."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for spec in specs:
            _write(root, spec)
        sys.path.insert(0, temporary)
        importlib.invalidate_caches()
        try:
            yield root
        finally:
            sys.path.remove(temporary)
            for spec in specs:
                sys.modules.pop(spec.module, None)
            importlib.invalidate_caches()


class ScaffoldedPack:
    """Generate the rail pack into a temporary directory and make it importable."""

    def __init__(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name) / "pack"
        cli_tasks.scaffold_pack(PACK, self.root)
        sys.path.insert(0, str(self.root))
        importlib.invalidate_caches()
        self.adapter_module: ModuleType = importlib.import_module(f"{PACK}.adapter")

    def close(self) -> None:
        sys.path.remove(str(self.root))
        for name in [name for name in sys.modules if name == PACK or name.startswith(PACK + ".")]:
            del sys.modules[name]
        self._directory.cleanup()
