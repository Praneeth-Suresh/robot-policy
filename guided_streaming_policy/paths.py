from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def pusht_root() -> Path:
    return repo_root() / "Push-T"


def robomimic_root() -> Path:
    return repo_root() / "Robotic_mimic"
