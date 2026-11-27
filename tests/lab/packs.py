"""Install throwaway task packs by writing package metadata and modules to a temp directory."""

import importlib
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = ["PackSpec", "installed_packs"]


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
