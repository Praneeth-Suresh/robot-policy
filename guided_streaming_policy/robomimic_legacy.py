from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import sys

from guided_streaming_policy.paths import repo_root, robomimic_root


TASK_CHOICES = ("can", "square", "lift")
DEFAULT_TASK = "can"


@dataclass(frozen=True)
class LegacyTask:
    name: str
    module: str
    description: str


TASKS = (
    LegacyTask("mimic-train-policy", "train.train_policy", "Train the streaming SI policy (robomimic)."),
    LegacyTask("mimic-train-critic", "train.train_critic", "Train the robust collision critic (robomimic)."),
    LegacyTask("mimic-infer-si", "inference.inference_si", "Run streaming SI inference (robomimic)."),
    LegacyTask("mimic-infer-ccg", "inference.inference_ccg", "Run guided CCG inference (robomimic)."),
    LegacyTask("mimic-infer-steg", "inference.inference_steg", "Run STEG-style inference (robomimic)."),
)


def get_task(name: str) -> LegacyTask:
    for task in TASKS:
        if task.name == name:
            return task
    raise KeyError(f"Unknown Robomimic task: {name}")


def task_names() -> list[str]:
    return [task.name for task in TASKS]


def build_task_command(task: LegacyTask, python_executable: str | None = None) -> list[str]:
    return [python_executable or sys.executable, "-m", task.module]


def run_task(
    task_name: str,
    task: str = DEFAULT_TASK,
    python_executable: str | None = None,
    dry_run: bool = False,
) -> int:
    legacy_task = get_task(task_name)
    if task not in TASK_CHOICES:
        raise ValueError(f"Unknown robomimic task variant: {task!r} (choose from {TASK_CHOICES})")

    cwd = robomimic_root()
    if not cwd.exists():
        raise FileNotFoundError(f"Robomimic workspace not found: {cwd}")

    command = build_task_command(legacy_task, python_executable=python_executable)
    if dry_run:
        print(f"cwd={cwd}")
        print(f"TASK={task}")
        print(" ".join(command))
        return 0

    env = os.environ.copy()
    # The original Robomimic scripts select can/square/lift via the TASK
    # environment variable (see Robomimic/core/config.py); injecting it here
    # reproduces that mechanism without touching the original code.
    env["TASK"] = task
    root = str(repo_root())
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not existing_pythonpath else f"{root}{os.pathsep}{existing_pythonpath}"

    completed = subprocess.run(command, cwd=str(cwd), check=False, env=env)
    return completed.returncode


def describe_workspace() -> Path:
    return robomimic_root()
