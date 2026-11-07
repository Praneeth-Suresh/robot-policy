"""Run selected Robomimic trials using environment variables."""
import os
from pathlib import Path
import runpy

method = os.getenv('METHOD', 'si')
task = os.getenv('TASK', 'can')
if method not in ('si', 'ccg', 'steg') or task not in ('can', 'square', 'lift'):
    raise SystemExit('Invalid METHOD or TASK')
try:
    count, seed = int(os.getenv('COUNT', '1')), int(os.getenv('SEED', '0'))
    if count < 1 or seed < 0 or os.getenv('RENDER_GIFS', '1') not in ('0', '1'):
        raise ValueError
except ValueError:
    raise SystemExit('Invalid COUNT, SEED or RENDER_GIFS')
os.chdir(Path(__file__).resolve().parent)
os.environ['TASK'] = task
os.environ.setdefault('MUJOCO_GL', 'osmesa')
if method != 'si':
    os.environ['TOTAL_NEEDED'] = str(count)
    os.environ.setdefault('RENDER_GIFS', '1')
    runpy.run_module('inference.inference_' + method, run_name='__main__')
else:
    import random
    import numpy as np
    import torch
    from inference.inference_si import wrapper, run_si_inference
    for current_seed in range(seed, seed + count):
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)
        wrapper.seed(current_seed)
        run_si_inference(0.0, f'SI seed {current_seed}',
                         save_path=f'si_{task}_results/si_seed_{current_seed}.gif')
