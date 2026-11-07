"""Run selected Push-T trials using environment variables."""
import importlib
import os
from pathlib import Path

method = os.getenv('METHOD', 'naive')
condition = os.getenv('CONDITION', 'static')
if method not in ('naive', 'ccg', 'steg', 'dp', 'flow') or condition not in ('static', 'dynamic'):
    raise SystemExit('Invalid METHOD or CONDITION')
try:
    count, seed = int(os.getenv('COUNT', '1')), int(os.getenv('SEED', '0'))
    if count < 1 or seed < 0:
        raise ValueError
except ValueError:
    raise SystemExit('Invalid COUNT or SEED: COUNT must be positive and SEED nonnegative')
os.chdir(Path(__file__).resolve().parent)
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import imageio

static = {
    'naive': ('inference_naive', 'run_naive_inference', {'max_guidance_scale': 6.0}),
    'ccg': ('inference_ccg', 'run_ccg_inference', {'max_guidance_scale': 4.0}),
    'steg': ('inference_steg', 'run_steg_sde_inference', {}),
    'dp': ('inference_dp_gconva', 'run_gconv_a_dp_inference', {}),
    'flow': ('inference_fp_gconva', 'run_gcov_a_fp_inference', {}),
}
dynamic = {'naive': ('inference_naive_dynamic', 7.0), 'ccg': ('inference_ccg_dynamic', 4.0),
           'steg': ('inference_steg_dynamic', 7.0), 'dp': ('inference_dp_gconva_dynamic', 15.0),
           'flow': ('inference_fp_gconva_dynamic', 0.7)}
out = Path('outputs/one-trial') / condition / method
out.mkdir(parents=True, exist_ok=True)
for current_seed in range(seed, seed + count):
    if condition == 'static':
        module, name, options = static[method]
        run = getattr(importlib.import_module('inference.static.' + module), name)
        metrics, _ = run(seed=current_seed, fixed_obstacle_pos=[-0.3, -0.5],
                         save_gif=True, output_dir=out, **options)
    else:
        module, scale = dynamic[method]
        run = importlib.import_module('inference.dynamic.' + module).run_guided_inference_visualized
        images, metrics = run(obstacle_mode=1, start_obs_pos_norm=[0.5, 0.5],
                              max_guidance_scale=scale, seed=current_seed)
        if images:
            imageio.mimsave(out / f'seed_{current_seed}_mode_1.gif', images, fps=20, loop=0)
    print('RESULT', condition, method, current_seed, metrics, flush=True)
