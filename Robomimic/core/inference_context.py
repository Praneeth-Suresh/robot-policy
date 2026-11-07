"""Shared inference normalization state for one process."""
from core.config import DATASET_PATH, pred_horizon, obs_horizon
from core.inference_stats import load_inference_stats

dataset = load_inference_stats(
    DATASET_PATH, ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    pred_horizon, obs_horizon,
)
