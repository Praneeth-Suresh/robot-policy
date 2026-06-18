from dataclasses import dataclass
import importlib.util
import platform
from pathlib import Path

from guided_streaming_policy.paths import pusht_root, repo_root


@dataclass(frozen=True)
class CheckResult:
    label: str
    ok: bool
    detail: str


DEPENDENCIES = (
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("diffusers", "diffusers"),
    ("tqdm", "tqdm"),
    ("zarr", "zarr"),
    ("gym", "gym"),
    ("cv2", "opencv-python"),
    ("imageio", "imageio"),
    ("pandas", "pandas"),
    ("IPython", "ipython"),
    ("pygame", "pygame"),
    ("pymunk", "pymunk"),
    ("shapely", "shapely"),
    ("skimage", "scikit-image"),
)


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _exists(path: Path) -> bool:
    return path.exists()


def collect_environment_report() -> list[CheckResult]:
    repo = repo_root()
    pusht = pusht_root()

    results = [
        CheckResult("python", True, platform.python_version()),
        CheckResult("repo-root", _exists(repo), str(repo)),
        CheckResult("pusht-root", _exists(pusht), str(pusht)),
    ]

    for relative in ("core", "train", "inference"):
        path = pusht / relative
        results.append(CheckResult(f"pusht/{relative}", _exists(path), str(path)))

    for relative in ("data", "checkpoints", "outputs"):
        path = pusht / relative
        results.append(CheckResult(f"artifact-dir:{relative}", _exists(path), str(path)))

    dataset_path = pusht / "data" / "pusht_cchi_v7_replay.zarr.zip"
    results.append(CheckResult("dataset:pusht", _exists(dataset_path), str(dataset_path)))

    for checkpoint_name in (
        "pusht_state_1000ep_si.ckpt",
        "pusht_state_100ep_dp.ckpt",
        "pusht_state_flow_100ep.ckpt",
        "CCG_V6_ep200.pth",
    ):
        checkpoint_path = pusht / "checkpoints" / checkpoint_name
        results.append(
            CheckResult(f"checkpoint:{checkpoint_name}", _exists(checkpoint_path), str(checkpoint_path))
        )

    for module_name, package_name in DEPENDENCIES:
        available = _module_available(module_name)
        detail = "importable" if available else f"missing, install `{package_name}`"
        results.append(CheckResult(f"python-dep:{module_name}", available, detail))

    return results


def render_environment_report() -> str:
    lines = []
    for result in collect_environment_report():
        status = "OK" if result.ok else "MISSING"
        lines.append(f"[{status}] {result.label}: {result.detail}")
    return "\n".join(lines)
