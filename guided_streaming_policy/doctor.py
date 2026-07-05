from dataclasses import dataclass
import importlib.util
import platform
from pathlib import Path

from guided_streaming_policy.paths import pusht_root, repo_root, robomimic_root


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


ROBOMIMIC_DEPENDENCIES = (
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("diffusers", "diffusers"),
    ("tqdm", "tqdm"),
    ("h5py", "h5py"),
    ("cv2", "opencv-python"),
    ("imageio", "imageio"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("mujoco", "mujoco"),
    ("robosuite", "robosuite"),
    ("robomimic", "robomimic"),
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


def collect_robomimic_report() -> list[CheckResult]:
    root = robomimic_root()

    results = [CheckResult("robomimic-root", _exists(root), str(root))]

    for relative in ("core", "train", "inference"):
        path = root / relative
        results.append(CheckResult(f"robomimic/{relative}", _exists(path), str(path)))

    checkpoint_dir = root / "checkpoint"
    results.append(CheckResult("artifact-dir:checkpoint", _exists(checkpoint_dir), str(checkpoint_dir)))

    # Matches Robomimic/core/config.py: <root>/datasets/<task>/ph/low_dim_abs.hdf5
    for task in ("can", "square", "lift"):
        dataset_path = root / "datasets" / task / "ph" / "low_dim_abs.hdf5"
        results.append(CheckResult(f"dataset:{task}", _exists(dataset_path), str(dataset_path)))

    for checkpoint_name in (
        "can_state_1200_ep_si_abs.ckpt",
        "square_state_1200_ep_si_abs.ckpt",
        "lift_state_1200_ep_si_abs.ckpt",
        "Robust_critic_can.pth",
    ):
        checkpoint_path = checkpoint_dir / checkpoint_name
        results.append(
            CheckResult(f"checkpoint:{checkpoint_name}", _exists(checkpoint_path), str(checkpoint_path))
        )

    for module_name, package_name in ROBOMIMIC_DEPENDENCIES:
        available = _module_available(module_name)
        detail = "importable" if available else f"missing, install `{package_name}`"
        results.append(CheckResult(f"python-dep:{module_name}", available, detail))

    return results


def _render_section(title: str, results: list[CheckResult]) -> list[str]:
    lines = [f"== {title} =="]
    for result in results:
        status = "OK" if result.ok else "MISSING"
        lines.append(f"[{status}] {result.label}: {result.detail}")
    return lines


def render_environment_report() -> str:
    lines = _render_section("Push-T", collect_environment_report())
    lines.append("")
    lines.extend(_render_section("Robomimic", collect_robomimic_report()))
    return "\n".join(lines)
