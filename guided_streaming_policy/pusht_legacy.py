from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import sys

from guided_streaming_policy.paths import pusht_root, repo_root


@dataclass(frozen=True)
class LegacyTask:
    name: str
    module: str
    description: str


TASKS = (
    LegacyTask("train-dp", "train.train_dp", "Train the diffusion-policy baseline."),
    LegacyTask("train-policy", "train.train_policy", "Train the streaming SI policy."),
    LegacyTask("train-critic", "train.train_critic", "Train the Push-T collision critic."),
    LegacyTask("infer-static-ccg", "inference.static.inference_ccg", "Run static guided CCG inference."),
    LegacyTask("infer-static-dp", "inference.static.inference_dp_gconva", "Run static diffusion-policy inference."),
    LegacyTask("infer-static-flow", "inference.static.inference_fp_gconva", "Run static flow-policy inference."),
    LegacyTask("infer-static-naive", "inference.static.inference_naive", "Run static unguided streaming inference."),
    LegacyTask("infer-static-steg", "inference.static.inference_steg", "Run static STEG-style inference."),
    LegacyTask("infer-dynamic-ccg", "inference.dynamic.inference_ccg_dynamic", "Run dynamic guided CCG inference."),
    LegacyTask("infer-dynamic-dp", "inference.dynamic.inference_dp_gconva_dynamic", "Run dynamic diffusion-policy inference."),
    LegacyTask("infer-dynamic-flow", "inference.dynamic.inference_fp_gconva_dynamic", "Run dynamic flow-policy inference."),
    LegacyTask("infer-dynamic-naive", "inference.dynamic.inference_naive_dynamic", "Run dynamic unguided streaming inference."),
    LegacyTask("infer-dynamic-steg", "inference.dynamic.inference_steg_dynamic", "Run dynamic STEG-style inference."),
)


def get_task(name: str) -> LegacyTask:
    for task in TASKS:
        if task.name == name:
            return task
    raise KeyError(f"Unknown Push-T task: {name}")


def task_names() -> list[str]:
    return [task.name for task in TASKS]


def build_task_command(task: LegacyTask, python_executable: str | None = None) -> list[str]:
    return [python_executable or sys.executable, "-m", task.module]


def run_task(
    task_name: str,
    python_executable: str | None = None,
    dry_run: bool = False,
) -> int:
    task = get_task(task_name)
    cwd = pusht_root()
    if not cwd.exists():
        raise FileNotFoundError(f"Push-T workspace not found: {cwd}")

    command = build_task_command(task, python_executable=python_executable)
    if dry_run:
        print(f"cwd={cwd}")
        print(" ".join(command))
        return 0

    env = os.environ.copy()
    root = str(repo_root())
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not existing_pythonpath else f"{root}{os.pathsep}{existing_pythonpath}"

    completed = subprocess.run(command, cwd=str(cwd), check=False, env=env)
    return completed.returncode


def describe_workspace() -> Path:
    return pusht_root()
