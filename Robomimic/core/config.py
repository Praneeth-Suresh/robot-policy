import os
import sys
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = PROJECT_ROOT 


if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TASK = os.environ.get("TASK", "can").strip().lower()

_DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets")
MODEL_ROOT = os.path.join(PROJECT_ROOT, "checkpoint")
_MODEL_ROOT = MODEL_ROOT


_TASK_REGISTRY = {
    "can": {
        "ckpt_epoch": 1200,
        "ccg":  {"guidance_scale": 25.0, "power_k": 2.0, "activation_dist": 0.2},
        "steg": {"guidance_scale": 10.0, "activation_dist": 0.3,
                 "N_ensemble": 32, "update_interval": 1, "K_horizon": 4},
    },
    "square": {
        "ckpt_epoch": 1200,
        "ccg":  {"guidance_scale": 70.0, "power_k": 2.0, "activation_dist": 0.2},
        "steg": {"guidance_scale": 10.0, "activation_dist": 0.3,
                 "N_ensemble": 32, "update_interval": 1, "K_horizon": 4},
    },
    "lift": {
        "ckpt_epoch": 1200,
        "ccg":  {"guidance_scale": 100.0, "power_k": 2.0, "activation_dist": 0.2},
        "steg": {"guidance_scale":10.0, "activation_dist": 0.3,
                 "N_ensemble": 32, "update_interval": 1, "K_horizon": 4},
    },
}
assert TASK in _TASK_REGISTRY, f"Unknown TASK={TASK!r}; options: {list(_TASK_REGISTRY)}"
_TASK_HP = _TASK_REGISTRY[TASK]

DATASET_PATH = os.path.join(_DATASET_ROOT, TASK, "ph", "low_dim_abs.hdf5")
CKPT_PATH_SI = os.path.join(MODEL_ROOT, f"{TASK}_state_{_TASK_HP['ckpt_epoch']}_ep_si_abs.ckpt")


CCG_HP = dict(_TASK_HP["ccg"])
STEG_HP = dict(_TASK_HP["steg"])

CCG_RESULTS_DIR = os.path.join(PROJECT_ROOT, f"ccg_{TASK}_results")
STEG_RESULTS_DIR = os.path.join(PROJECT_ROOT, f"steg_{TASK}_results")
SI_RESULTS_DIR = os.path.join(PROJECT_ROOT, f"si_{TASK}_results")


obs_keys = [
    "object",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
]
pred_horizon = 16     # predict 16 future steps
obs_horizon = 2       # observe past 2 steps
action_horizon = 8    # execute 8 steps per chunk
action_dim = 10       # 3 pos + 6 rot(6d) + 1 gripper
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_config():
    print(f"[TASK] PROJECT_ROOT={PROJECT_ROOT}")
    print(f"[TASK] TASK={TASK} | dataset={DATASET_PATH}")
    print(f"[TASK] ckpt={CKPT_PATH_SI}")
    print(f"[TASK] CCG_HP={CCG_HP} | results={CCG_RESULTS_DIR}")
    print(f"[TASK] STEG_HP={STEG_HP} | results={STEG_RESULTS_DIR}")
    print(f"[TASK] device={device}")