import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class InferenceCacheTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("h5py"), "Robomimic dependencies unavailable")
    def test_cache_matches_loader_and_rebuilds(self):
        code = '''
import os, tempfile
import h5py
import numpy as np
from core.dataset import RobomimicDataset
from core.inference_stats import load_inference_stats
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "data.hdf5")
    cache = os.path.join(d, "cache.npz")
    def write(offset):
        with h5py.File(path, "w") as f:
            demo = f.create_group("data/demo_0")
            obs = demo.create_group("obs")
            obs["x"] = np.arange(40, dtype=np.float32).reshape(20, 2) + offset
            demo["actions"] = np.arange(140, dtype=np.float32).reshape(20, 7) / 100
    write(0)
    full = RobomimicDataset(path, ["x"], 4, 2, 2)
    cached = load_inference_stats(path, ["x"], 4, 2, cache)
    np.testing.assert_array_equal(full.obs_normalizer.min_val, cached.obs_normalizer.min_val)
    np.testing.assert_array_equal(full.action_normalizer.max_val, cached.action_normalizer.max_val)
    np.testing.assert_allclose(full.get_unnormalized_action(np.zeros((1, 10))), cached.get_unnormalized_action(np.zeros((1, 10))))
    again = load_inference_stats(path, ["x"], 4, 2, cache)
    assert again.from_cache
    write(10)
    os.utime(path, None)
    changed = load_inference_stats(path, ["x"], 4, 2, cache)
    assert not changed.from_cache
    assert changed.obs_normalizer.min_val[0] == 10
'''
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'Robomimic'))
        result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
