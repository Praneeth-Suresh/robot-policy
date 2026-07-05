
import os
import sys
import glob
import ctypes
import traceback

os.environ.setdefault("MUJOCO_PY_FORCE_CPU", "1")
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "swrast")

# Ensure LD_LIBRARY_PATH includes mujoco210/bin (otherwise mujoco_py raises Missing path)
default_mujoco_bin = os.path.expanduser("~/.mujoco/mujoco210/bin")
if os.path.isdir(default_mujoco_bin):
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_paths = [p for p in current_ld.split(":") if p]
    if default_mujoco_bin not in ld_paths:
        os.environ["LD_LIBRARY_PATH"] = default_mujoco_bin + (":" + current_ld if current_ld else "")

# Dependency injection: preload OSMesa / libffi / GL with RTLD_GLOBAL (must run before import mujoco_py)
_system_lib_path = "/usr/lib/x86_64-linux-gnu"

def _dlopen_first(glob_pattern: str, label: str) -> None:
    candidates = glob.glob(glob_pattern)
    if not candidates:
        print(f"{label} not found: {glob_pattern}")
        return
    target = candidates[0]
    try:
        rtld_now = getattr(os, "RTLD_NOW", 0)  # some systems lack os.RTLD_NOW
        ctypes.CDLL(target, mode=ctypes.RTLD_GLOBAL | rtld_now)
        print(f"{label} injected successfully: {target}")
    except Exception as e:
        print(f"{label} injection failed: {e} ({target})")

_dlopen_first(os.path.join(_system_lib_path, "libffi.so.*"), "libffi")        # .7 / .8 depends on system
_dlopen_first(os.path.join(_system_lib_path, "libOSMesa.so*"), "OSMesa")
_dlopen_first(os.path.join(_system_lib_path, "libGL.so.1"), "libGL")
_dlopen_first(os.path.join(default_mujoco_bin, "libglewosmesa.so*"), "libglewosmesa (mujoco210)")

# ================= 1. Application imports (OSMesa backend locked at this point) =================
from typing import List, Dict, Optional

import numpy as np
import h5py
import gym
from gym.spaces import Box

import mujoco_py
from mujoco_py.builder import MujocoException  # used by wrapper.step(); was in removed EGL block
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_robosuite import EnvRobosuite

# DATASET_PATH / obs_keys / PROJECT_ROOT (on sys.path) from core.config
from core.config import *  # noqa: F401,F403

import collections                                          # noqa: F401
from dataclasses import dataclass                           # noqa: F401
import math                                                 # noqa: F401
import random
import torch                                                # noqa: F401
from torch import Tensor                                    # noqa: F401
import torch.nn as nn                                       # noqa: F401
from tqdm.auto import tqdm                                  # noqa: F401

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_random_seed(0)

# Default observation keys (consistent with original build call)
DEFAULT_OBS_KEYS: List[str] = [
    'object',
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
]


# ================= 2. Wrapper (original logic preserved line-by-line) =================
class RobomimicLowdimWrapper(gym.Env):
    def __init__(self,
        env: EnvRobosuite,
        obs_keys: List[str]=[
            'object',
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos'],
        init_state: Optional[np.ndarray]=None,
        render_hw=(256,256),
        render_camera_name='agentview',
        clip_action: bool = True
        ):

        self.env = env
        self.obs_keys = obs_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.clip_action = clip_action
        self.seed_state_map = dict()
        self._seed = None

        # setup spaces
        if self.clip_action:
            # delta control (or bounded control): inputs typically in [-1, 1]
            low = np.full(env.action_dimension, fill_value=-1.0, dtype=np.float32)
            high = np.full(env.action_dimension, fill_value=1.0, dtype=np.float32)
        else:
            # absolute control: real actions may exceed [-1,1] (e.g. z≈1.02, rotvec up to 2~3)
            # do not clip range here; only sanitize NaN/Inf
            low = np.full(env.action_dimension, fill_value=-np.inf, dtype=np.float32)
            high = np.full(env.action_dimension, fill_value=np.inf, dtype=np.float32)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )

    # [Added] expose wrapper.sim for external access
    @property
    def sim(self):
        # Traverse two layers: wrapper.env -> EnvRobosuite.env -> RobosuiteRawEnv
        return self.env.env.sim

    def get_observation(self):
        raw_obs = self.env.get_observation()
        obs = np.concatenate([
            raw_obs[key] for key in self.obs_keys
        ], axis=0)
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self):
        if self.init_state is not None:
            # always reset to the same state
            self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            self.env.reset()

        obs = self.get_observation()
        return obs

    def step(self, action):
        # --- Safety: MuJoCo/robosuite is sensitive to NaN / Inf / out-of-range actions ---
        action = np.asarray(action, dtype=np.float32)
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        # delta mode: clip to [-1,1]; absolute mode: no clip (preserve real action scale)
        if getattr(self, "clip_action", True):
            action = np.clip(action, -1.0, 1.0)

        try:
            raw_obs, reward, done, info = self.env.step(action)
        except MujocoException as e:
            print(f"[MuJoCoException] step() failed: {e}")
            obs = self.reset()
            info = {"mujoco_exception": str(e)}
            return obs, 0.0, True, info
        obs = np.concatenate([
            raw_obs[key] for key in self.obs_keys
        ], axis=0)
        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode,
            height=h, width=w,
            camera_name=self.render_camera_name)


# ================= 3. control_delta self-calibration (original logic preserved) =================
def _infer_control_delta_by_replay(dataset_path: str, env_meta: dict, demo_key: str = "demo_0") -> bool:
    """Replay demo raw actions; pick control_delta=True/False with higher reward."""
    import h5py
    import numpy as np
    import robomimic.utils.env_utils as EnvUtils

    with h5py.File(dataset_path, "r") as f:
        actions = f[f"data/{demo_key}/actions"][:].astype(np.float32)
        init_state = f[f"data/{demo_key}/states"][0]

    def _rollout(cd: bool) -> float:
        # clone env_meta
        meta = dict(env_meta)
        env_kwargs = dict(meta.get("env_kwargs", {}))
        ccfg = dict(env_kwargs.get("controller_configs", {}))
        ccfg["control_delta"] = bool(cd)
        env_kwargs["controller_configs"] = ccfg
        meta["env_kwargs"] = env_kwargs

        env = EnvUtils.create_env_from_metadata(
            env_meta=meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        env.reset_to({"states": init_state})
        max_r = 0.0
        for a in actions:
            _, r, d, _ = env.step(a)
            if r > max_r:
                max_r = float(r)
            if d:
                break
        return float(max_r)

    max_r_true = _rollout(True)
    max_r_false = _rollout(False)
    best = False if (max_r_false >= max_r_true) else True
    print(f"[control_delta replay] max_r(True)={max_r_true} | max_r(False)={max_r_false} -> choose {best}")
    return bool(best)


def resolve_control_delta(dataset_path: str, env_meta: dict) -> bool:
    """Three-level calibration: meta -> stats -> replay; write back to env_meta in place."""
    ccfg = env_meta.get('env_kwargs', {}).get('controller_configs', {})
    print(f"[env_meta] controller_configs.control_delta = {ccfg.get('control_delta', None)}")
    print(f"[env_meta] reward_shaping = {env_meta.get('env_kwargs', {}).get('reward_shaping', None)}")

    # Stats fallback: delta action[:3] mean abs near 0; absolute typically deviates from 0
    _inferred_control_delta = None
    try:
        with h5py.File(dataset_path, 'r') as f:
            a = f['data/demo_0/actions'][:]
            a_xyz = a[: min(2000, a.shape[0]), :3].astype(np.float32)
            mean_abs = float(np.mean(np.abs(a_xyz)))
            mean = a_xyz.mean(axis=0)
            std = a_xyz.std(axis=0)
            print(f"[dataset action stats] mean_abs(|xyz|)={mean_abs:.4f} mean={mean} std={std}")
            _inferred_control_delta = bool(mean_abs < 0.10)
    except Exception as e:
        print(f"[warn] failed to infer control_delta from dataset actions: {e}")

    # Prefer metadata; use stats only when missing
    control_delta = ccfg.get('control_delta', None)
    if control_delta is None and _inferred_control_delta is not None:
        control_delta = _inferred_control_delta
    print(f"[control_delta] initial CONTROL_DELTA (from meta/stats) = {control_delta}")

    # Replay result overrides
    try:
        control_delta = bool(_infer_control_delta_by_replay(
            dataset_path=dataset_path, env_meta=env_meta, demo_key="demo_0"))
    except Exception as e:
        print(f"[warn] control_delta replay inference failed, keep initial CONTROL_DELTA={control_delta}. err={e}")

    # Write back env_meta so env/controller config matches inference
    env_meta["env_kwargs"]["controller_configs"]["control_delta"] = bool(control_delta)
    print(f"[env_meta] OVERRIDE: controller_configs.control_delta = "
          f"{env_meta['env_kwargs']['controller_configs']['control_delta']}")
    print(f"[control_delta] final CONTROL_DELTA = {control_delta}")
    return bool(control_delta)


# ================= 4. Build env + wrapper =================
def build_env_and_wrapper(dataset_path: str, obs_keys: Optional[List[str]] = None):
    """Read meta -> init ObsUtils -> calibrate control_delta -> build env+wrapper."""
    obs_keys = list(obs_keys) if obs_keys else list(DEFAULT_OBS_KEYS)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    print(f"Environment name: {env_meta['env_name']}")
    print(f"Environment type: {env_meta['type']}")

    # Tell robomimic these keys are low_dim (required before building env)
    ObsUtils.initialize_obs_modality_mapping_from_dict({'low_dim': obs_keys})

    control_delta = resolve_control_delta(dataset_path, env_meta)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,  # offscreen rendering required
        use_image_obs=False,
    )
    wrapper = RobomimicLowdimWrapper(
        env=env,
        obs_keys=obs_keys,
        render_hw=(256, 256),
        render_camera_name='agentview',
        # delta: clip to [-1,1]; absolute: no clip to avoid truncating workspace coordinates
        clip_action=bool(control_delta) if control_delta is not None else True,
    )
    print("Environment created and wrapped successfully!")
    print(f"Action Space: {wrapper.action_space}")
    print(f"Observation Space: {wrapper.observation_space}")
    return wrapper, env_meta


# ================= 5. Optional self-check (__main__ only; not on import) =================
def inspect_action_dim(dataset_path: str, demo_key: str = "demo_0") -> None:
    try:
        with h5py.File(dataset_path, 'r') as f:
            action_shape = f[f'data/{demo_key}/actions'].shape
        print(f"Dataset loaded successfully! Action shape: {action_shape}")
        dim = action_shape[1]
        if dim == 10:
            print("10D absolute action.")
        elif dim == 7:
            print("7D action (OSC_POSE: 6+gripper).")
        else:
            print(f"Unknown dimension: {dim}")
    except Exception as e:
        print(f"Failed to read dataset: {e}")


def sanity_check(wrapper) -> None:
    import matplotlib.pyplot as plt

    obs = wrapper.reset()
    print(f"1. Reset OK | initial obs shape: {obs.shape}")

    action = np.zeros(wrapper.action_space.shape)
    next_obs, reward, done, info = wrapper.step(action)
    print(f"2. Step OK  | obs shape after step: {next_obs.shape}, Reward: {reward}")

    try:
        img = wrapper.render()
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title("Environment Sanity Check")
        plt.axis('off')
        plt.show()
        print("3. Render OK | you should see the arm and red block.")
    except Exception as e:
        print("\n[Error] Rendering failed!")
        print(f"Error message: {e}")
        print("Hint: on headless servers, ensure OSMesa libraries are ready (see file header).")



from types import SimpleNamespace

_ENV_SINGLETON: Optional[SimpleNamespace] = None


def get_env(dataset_path: Optional[str] = None,
            obs_keys: Optional[List[str]] = None) -> SimpleNamespace:
    """Build and cache environment singleton on demand. First call builds env+wrapper and runs control_delta calibration."""
    global _ENV_SINGLETON
    if _ENV_SINGLETON is not None:
        return _ENV_SINGLETON

    dp = dataset_path or DATASET_PATH
    wrapper, env_meta = build_env_and_wrapper(dp, obs_keys=obs_keys or DEFAULT_OBS_KEYS)

    from core.myrotation_transformer_final import RotationTransformer
    rotation_transformer = RotationTransformer('rotation_6d', 'axis_angle')

    _ENV_SINGLETON = SimpleNamespace(
        wrapper=wrapper,
        env=wrapper.env,
        env_meta=env_meta,
        dataset_path=dp,
        control_delta=bool(wrapper.clip_action),  # == control_delta at build time
        rotation_transformer=rotation_transformer,
        action_dim=10,  # note: can/ph datasets are 7D raw; script uses 10D latent — verify as needed
    )
    return _ENV_SINGLETON


def main():
    print(f"Dataset: {DATASET_PATH}")
    inspect_action_dim(DATASET_PATH)
    e = get_env()
    sanity_check(e.wrapper)
    print(f"RotationTransformer initialized, Action Dim = {e.action_dim}")
    return e


if __name__ == "__main__":
    main()
