"""Cached normalization data used by Robomimic inference."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
import torch

from core.dataset import MinMaxNormalizer, RobomimicDataset
from core.myrotation_transformer_final import RotationTransformer


class InferenceStats:
    get_unnormalized_action = RobomimicDataset.get_unnormalized_action

    def __init__(self, values, obs_horizon, from_cache):
        self.obs_normalizer = MinMaxNormalizer(min_val=values['obs_min'], max_val=values['obs_max'])
        self.action_normalizer = MinMaxNormalizer(min_val=values['action_min'], max_val=values['action_max'])
        self.rotation_transformer = RotationTransformer(from_rep='axis_angle', to_rep='rotation_6d')
        self.obs_horizon = obs_horizon
        self.from_cache = from_cache
        self.sample_count = int(values['sample_count'])

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return {'obs': torch.zeros((self.obs_horizon, len(self.obs_normalizer.min_val))),
                'action': torch.zeros((1, len(self.action_normalizer.min_val)))}


def load_inference_stats(dataset_path, obs_keys, pred_horizon, obs_horizon, cache_path=None):
    path = Path(dataset_path).resolve()
    stat = path.stat()
    identity = json.dumps([str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                           list(obs_keys), pred_horizon, 1])
    digest = hashlib.sha256(identity.encode()).hexdigest()
    if cache_path is None:
        cache_path = path.parent / '.gsp-cache' / (digest + '.npz')
    cache_path = Path(cache_path)
    try:
        with np.load(cache_path, allow_pickle=False) as saved:
            if str(saved['identity']) != identity:
                raise ValueError('stale cache')
            values = {k: saved[k].copy() for k in ('obs_min', 'obs_max', 'action_min', 'action_max', 'sample_count')}
            if (values['obs_min'].ndim != 1 or values['obs_min'].size == 0 or
                values['obs_min'].shape != values['obs_max'].shape or
                values['action_min'].shape != (10,) or values['action_max'].shape != (10,) or
                int(values['sample_count']) <= 0 or
                any(not np.all(np.isfinite(values[k])) for k in ('obs_min', 'obs_max', 'action_min', 'action_max')) or
                np.any(values['obs_min'] > values['obs_max']) or np.any(values['action_min'] > values['action_max'])):
                raise ValueError('invalid cache')
        print(f'Loaded inference statistics: {cache_path}')
        return InferenceStats(values, obs_horizon, True)
    except (OSError, ValueError, KeyError, EOFError):
        pass
    print(f'Computing inference statistics: {path}')
    transformer = RotationTransformer(from_rep='axis_angle', to_rep='rotation_6d')
    obs_min = obs_max = action_min = action_max = None
    count = 0
    with h5py.File(path, 'r') as source:
        for demo in source['data'].values():
            obs = np.concatenate([demo['obs'][key][:] for key in obs_keys], axis=-1)
            raw = demo['actions'][:].astype(np.float32)
            rot = transformer.forward(torch.from_numpy(raw[:, 3:6])).numpy()
            actions = np.concatenate([raw[:, :3], rot, raw[:, 6:]], axis=-1)
            low, high = np.min(obs, axis=0), np.max(obs, axis=0)
            alow, ahigh = np.min(actions, axis=0), np.max(actions, axis=0)
            obs_min = low if obs_min is None else np.minimum(obs_min, low)
            obs_max = high if obs_max is None else np.maximum(obs_max, high)
            action_min = alow if action_min is None else np.minimum(action_min, alow)
            action_max = ahigh if action_max is None else np.maximum(action_max, ahigh)
            count += max(0, len(obs) - pred_horizon)
    values = dict(obs_min=obs_min, obs_max=obs_max, action_min=action_min,
                  action_max=action_max, sample_count=count)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix='.npz', delete=False) as temp:
        temp_path = temp.name
    try:
        np.savez(temp_path, identity=identity, **values)
        os.replace(temp_path, cache_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    print(f'Saved inference statistics: {cache_path}')
    return InferenceStats(values, obs_horizon, False)
