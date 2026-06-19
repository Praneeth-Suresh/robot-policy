import os
from pathlib import Path
import collections

import numpy as np
import torch
from tqdm.auto import tqdm

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from core.pusht_env import PushTEnv
from core.dataset import PushTDataset, normalize_data, unnormalize_data
from core.models import ConditionalUnet1D

import cv2
import math
import imageio
import pandas as pd
from core.my_utils import *


PROJECT_ROOT = Path(__file__).resolve().parents[2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

obs_horizon = 2
obs_dim = 5
action_dim = 2
pred_horizon = 16
action_horizon = 8

dataset_path = PROJECT_ROOT / "data" / "pusht_cchi_v7_replay.zarr.zip"

dataset = PushTDataset(
    dataset_path=str(dataset_path),
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon,
)

stats = dataset.stats

env = PushTEnv()
#—————————Create PyTorch model for diffusion policy——————
# Create network object
dp_noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim * obs_horizon,
    updownsample_type="Conv",
    sin_embedding_scale=1,
).to(device)

num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)



#————Loading pretrained checkpoint———————
ckpt_path_dp = PROJECT_ROOT / "checkpoints" / "pusht_state_100ep_dp.ckpt"

if not os.path.isfile(ckpt_path_dp):
    raise FileNotFoundError(f"DP checkpoint not found: {ckpt_path_dp}")

state_dict_dp = torch.load(ckpt_path_dp, map_location=device)

ema_noise_pred_net_dp = dp_noise_pred_net
ema_noise_pred_net_dp.load_state_dict(state_dict_dp)
ema_noise_pred_net_dp.eval()

print("Pretrained weights loaded for diffusion policy.")



def run_gconv_a_dp_inference(obstacle_percentage=0.5, 
                              fixed_obstacle_pos=None,
                              guidance_scale=2.0,  # Diffusion Guidance Scale typically lower than flow
                              guidance_sign=+1.0,
                              grad_clip_norm=1.0,
                              debug_guidance=False,
                              seed=42, 
                              save_gif=True, 
                              output_dir=PROJECT_ROOT / "outputs" / "gconv_a_dp_results",
                              show_baseline=True,
                              show_axes=True):
    
    result_metrics = {
        "seed": seed,
        "obs_pct": obstacle_percentage,
        # Baseline Metrics
        "base_collision": False,
        "base_reward": 0.0,
        "base_success": False,
        "base_min_dist": 0.0,
        # GCOV-A Metrics
        "guided_collision": False,
        "guided_reward": 0.0,
        "guided_success": False,
        "guided_min_dist": 0.0
    }

    latency_tracker = LatencyMeasurer(enabled=False) 
    def run_episode(obstacle_pos_norm=None, record_traj=False, baseline_pixel_traj=None, force_no_guidance=False):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            import random
            random.seed(seed)
        
        try:
            env.reset(seed=seed)
        except:
            env.seed(seed)
            
        obs, info = env.reset()
        obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
        
        start_img = env.render()
        img_h, img_w = start_img.shape[:2]
        vis_scale = img_w / getattr(env, 'window_size', 512)
        
        TRAIN_NORM_R = 0.06 
        PHYSICAL_RADIUS = TRAIN_NORM_R * 256.0      # ~15.36 px
        COLLISION_THRESHOLD = PHYSICAL_RADIUS * 2   # ~30.72 px
        VIS_RADIUS = max(2, int(PHYSICAL_RADIUS * vis_scale))     # ~8 px
        VIS_THRESHOLD_PIXEL = int(COLLISION_THRESHOLD * vis_scale) # ~15 px

        activation_dist = 0.6

        obs_tensor = None
        obs_pixel = None
        obs_real_pos_arr = None 
        min_dist_to_obs = float('inf')
        
        if obstacle_pos_norm is not None:
            obs_tensor = torch.tensor([obstacle_pos_norm], device=device, dtype=torch.float32)
            obs_real_pos_arr = unnormalize_data(obstacle_pos_norm, stats['action'])
            px = int(obs_real_pos_arr[0] * vis_scale)
            py = int(obs_real_pos_arr[1] * vis_scale)
            obs_pixel = (px, py)

        recorded_pixels = []
        recorded_norms = []
        imgs = []
        episode_rewards = []
        
        done = False
        step_idx = 0
        max_steps = 250
        

        while not done and step_idx < max_steps:
            obs_seq = np.stack(obs_deque)
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            nobs_tensor = torch.from_numpy(nobs).to(device, dtype=torch.float32)
            
            exec_start_t = obs_horizon - 1
            exec_end_t = exec_start_t + action_horizon
            exec_start_t = int(np.clip(exec_start_t, 0, pred_horizon - 1))
            exec_end_t = int(np.clip(exec_end_t, exec_start_t + 1, pred_horizon))
            
            # Observation conditioning
            obs_cond = nobs_tensor.unsqueeze(0).flatten(start_dim=1)  # (1, To * A)
            
            with latency_tracker: 
                # 1. Initialize action from pure Gaussian noise
                x_t = torch.randn((1, pred_horizon, action_dim), device=device) # (1, Tp, A)
                
                # Init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)
                
                # Denoising loop
                for k in noise_scheduler.timesteps:
                    x_t = x_t.detach().requires_grad_(True)
                    
                    # Predict noise
                    noise_pred = ema_noise_pred_net_dp(
                        sample=x_t,
                        timestep=k,
                        global_cond=obs_cond
                    )
                    
                    # --- Guidance Calculation (Standard g_cov-A / Reconstruction Guidance) ---
                    if (not force_no_guidance) and (obstacle_pos_norm is not None):
                        alpha_prod_t = noise_scheduler.alphas_cumprod[k].to(device)
                        beta_prod_t = (1 - alpha_prod_t)
                        
                        noise_pred_detached = noise_pred.detach()
                        
                        x_0_pred = (x_t - (beta_prod_t ** 0.5) * noise_pred_detached) / (alpha_prod_t ** 0.5)
                        
                        obs_expanded = obs_tensor.reshape(1, 1, -1)
                        x_0_exec = x_0_pred[:, exec_start_t:exec_end_t, :]
                        dist_to_obs = torch.norm(x_0_exec - obs_expanded, dim=-1)  # (1, Ta)
                        min_dist_val, _ = torch.min(dist_to_obs, dim=1)           # (1,)
                        curr_dist_norm = min_dist_val.detach().item()
                        
                        grad_component = torch.zeros_like(noise_pred)
                        
                        if curr_dist_norm < activation_dist:
                            dist_sq = ((x_0_exec - obs_expanded)**2).sum(dim=-1)
                            
                            sigma_cost = 0.25 
                            J = torch.sum(torch.exp(-dist_sq / (2 * sigma_cost**2)))
                            
                            grad_xt = torch.autograd.grad(
                                J, x_t,
                                retain_graph=False,
                                create_graph=False
                            )[0]
                            if grad_clip_norm is not None and grad_clip_norm > 0:
                                grad_norm = torch.norm(grad_xt.reshape(1, -1), p=2, dim=1, keepdim=True)  # (1,1)
                                scale = torch.clamp(grad_clip_norm / (grad_norm + 1e-12), max=1.0)
                                grad_xt = grad_xt * scale.view(1, 1, 1)
                            
                            # Schedule
                            time_decay = (alpha_prod_t ** 0.5)
                            dynamic_scale = guidance_scale * (1.0 - curr_dist_norm / activation_dist) * time_decay
                            
                            # eps_hat = eps + sqrt(1-alpha) * grad_xt(J)
                            grad_component = guidance_sign * (beta_prod_t ** 0.5) * grad_xt * dynamic_scale
                            if debug_guidance and (k == noise_scheduler.timesteps[-1]):
                                print(f"[guidance dbg] curr_dist={curr_dist_norm:.3f} alpha_bar={alpha_prod_t.item():.4f} "
                                      f"dyn={dynamic_scale.item():.4f} grad_norm={torch.norm(grad_component).item():.4f}")
                        
                        noise_pred = noise_pred + grad_component
                            
                    # --- End Guidance ---

                    # Reverse diffusion step
                    x_t = noise_scheduler.step(
                        model_output=noise_pred.detach(),
                        timestep=k,
                        sample=x_t.detach(),
                    ).prev_sample

            na_traj = x_t.detach()
            
            # Unnormalize
            a_cpu = na_traj[0].to('cpu').numpy()
            a_traj_real = unnormalize_data(a_cpu, stats=stats['action'])
            
            start_t = exec_start_t
            end_t = exec_end_t
            
            # Execute actions
            for i in range(start_t, end_t):
                 a_real = a_traj_real[i]
                 # Record
                 if record_traj:
                    curr_px = (int(a_real[0] * vis_scale), int(a_real[1] * vis_scale))
                    recorded_pixels.append(curr_px)
                    recorded_norms.append(a_cpu[i])

                 if obs_real_pos_arr is not None:
                    agent_pos = info['pos_agent']
                    dist = np.linalg.norm(agent_pos - obs_real_pos_arr)
                    min_dist_to_obs = min(min_dist_to_obs, dist)

                 
                 obs, reward, done, _, info = env.step(a_real)
                 obs_deque.append(obs)
                 episode_rewards.append(reward)
                 
                 if save_gif and not record_traj:
                    img_rgb = env.render().copy()
                    if show_axes: draw_coordinate_axes(img_rgb)
                    
                    if obs_pixel is not None:
                        cv2.circle(img_rgb, obs_pixel, VIS_RADIUS, (255, 255, 0), -1) 
                        cv2.circle(img_rgb, obs_pixel, VIS_THRESHOLD_PIXEL, (0, 0, 255), 1) 

                    curr_agent_pos = info['pos_agent']
                    curr_px_vis = (int(curr_agent_pos[0] * vis_scale), int(curr_agent_pos[1] * vis_scale))
                    cv2.circle(img_rgb, curr_px_vis, VIS_RADIUS, (0, 0, 255), -1)

                    if show_baseline and baseline_pixel_traj is not None:
                        pts = np.array(baseline_pixel_traj, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img_rgb, [pts], False, (0, 255, 0), 2, cv2.LINE_AA)

                    if (not force_no_guidance) and (obstacle_pos_norm is not None):
                        
                        if 'grad_component' in locals():
                            idx_vis = int(np.clip(exec_start_t, 0, grad_component.shape[1] - 1))
                            vec_guide = grad_component[0, idx_vis, :].detach().cpu().numpy() # (2,)
                            # eps_hat = eps - sqrt(...) * grad(log p)
                            draw_vector_arrow(img_rgb, curr_px_vis, -vec_guide, (0, 0, 255), scale=500.0)

                    imgs.append(img_rgb)
                 
                 step_idx += 1
                 if done or step_idx >= max_steps: break
            
            if done or step_idx >= max_steps: break

        max_reward = max(episode_rewards) if episode_rewards else 0.0

        metrics = {}
        if not record_traj:
            is_collision = False
            if obs_real_pos_arr is not None:
                is_collision = min_dist_to_obs < COLLISION_THRESHOLD
            
            is_success = (not is_collision) and (max_reward > 0.85)
            
            metrics = {
                "collision": is_collision,
                "reward": max_reward,
                "success": is_success,
                "min_dist": min_dist_to_obs
            }

        if record_traj:
            return recorded_pixels, recorded_norms
        else:
            return metrics, imgs

    # === Main Logic ===
    
    # 1. Warm-up
    rng_state = torch.get_rng_state()
    base_pixels, base_norms = run_episode(obstacle_pos_norm=None, record_traj=True)
    torch.set_rng_state(rng_state)


    if len(base_norms) > 0:
        real_start_pos = base_norms[0] 
    else:
        real_start_pos = np.array([0.0, 0.0])



    if len(base_norms) == 0: return result_metrics, []
        
    if fixed_obstacle_pos is not None:
        auto_obs_pos = np.array(fixed_obstacle_pos, dtype=np.float32)
        print(f"Using Fixed Obstacle Pos: {auto_obs_pos}")
    else:
        target_idx = np.clip(int(len(base_norms) * obstacle_percentage), 0, len(base_norms) - 1)
        auto_obs_pos = base_norms[target_idx] 


    result_metrics["agent_start_x"] = float(real_start_pos[0])
    result_metrics["agent_start_y"] = float(real_start_pos[1])



    # 2. Baseline
    base_metrics, _ = run_episode(
        obstacle_pos_norm=auto_obs_pos, 
        record_traj=False, 
        force_no_guidance=True, 
        baseline_pixel_traj=None
    )
    
    result_metrics["base_collision"] = base_metrics["collision"]
    result_metrics["base_reward"] = base_metrics["reward"]
    result_metrics["base_success"] = base_metrics["success"]
    result_metrics["base_min_dist"] = base_metrics["min_dist"]

    #print(">>> Resetting Latency Tracker for Guided Run...")
    latency_tracker = LatencyMeasurer(warmup_steps=10, chunk_size=action_horizon, enabled=True)



    # 3. Guided
    print("Running GCOV-A Inference (Diffusion)...")
    guided_metrics, final_imgs = run_episode(
        obstacle_pos_norm=auto_obs_pos, 
        record_traj=False, 
        force_no_guidance=False, 
        baseline_pixel_traj=base_pixels
    )
    
    result_metrics["guided_collision"] = guided_metrics["collision"]
    result_metrics["guided_reward"] = guided_metrics["reward"]
    result_metrics["guided_success"] = guided_metrics["success"]
    result_metrics["guided_min_dist"] = guided_metrics["min_dist"]
    
    base_status = "FAIL" if base_metrics['collision'] else "PASS"
    guide_status = "FAIL" if guided_metrics['collision'] else "PASS"
    print("\n[GCOV-A Result]")
    print(f"Base:   Reward={base_metrics['reward']:.2f}, Collision={base_metrics['collision']}")
    print(f"Guided: Reward={guided_metrics['reward']:.2f}, Collision={guided_metrics['collision']}")
    print(f"Improvement: {base_status} -> {guide_status}")



    if save_gif and len(final_imgs) > 0:
        save_gif_separated(
            imgs=final_imgs,
            is_success=result_metrics["guided_success"], 
            is_collision=result_metrics["guided_collision"],
            base_collision=result_metrics["base_collision"],
            base_reward=result_metrics["base_reward"],
            output_dir=output_dir,
            seed=seed,
            fixed_pos=fixed_obstacle_pos,
            obs_pct=obstacle_percentage
        )
    #latency_tracker.print_summary(label=f"Guided Inference (Seed {seed})")⌚️
    record_latency_global(latency_tracker)




    return result_metrics, final_imgs




def run_batch_experiments_dp(start_seed=0, num_seeds=10, 
                          obstacles=None, 
                          guidance_scale=2.0,
                          save_gif=False):
    
    if obstacles is None:
        obstacles = []

    print(f"=== Starting Batch Experiment (Diffusion GCOV-A) ===")
    print(f"Seeds: {start_seed} -> {start_seed + num_seeds - 1}")
    print(f"Obstacles: {len(obstacles)} positions")
    print(f"Guidance Scale: {guidance_scale}")
    print(f"Save GIF: {save_gif}\n")
    
    all_results = []
    
    for i in range(num_seeds):
        seed = start_seed + i
        print(f"\n--- Processing Seed {seed} ({i+1}/{num_seeds}) ---")
        
        for obs_pos in obstacles:
            # Reuse the run_gcova_inference function defined previously
            metrics, _ = run_gconv_a_dp_inference(
                fixed_obstacle_pos=obs_pos, 
                guidance_scale=guidance_scale, 
                seed=seed,
                save_gif=save_gif,
                output_dir=PROJECT_ROOT / "outputs" / "gconv_a_dp_batch_results-10",
                show_baseline=False,
                show_axes=False
            )
            
            metrics['obs_pos_tuple'] = tuple(obs_pos)
            all_results.append(metrics)
            
            base_status = "FAIL" if metrics['base_collision'] else "PASS"
            guide_status = "FAIL" if metrics['guided_collision'] else "PASS"
            
            print(f"  [Pos {obs_pos}] Base: {metrics['base_reward']:.2f}({base_status}) -> Guided: {metrics['guided_reward']:.2f}({guide_status})")

    df = pd.DataFrame(all_results)
    
    print("\n" + "="*60)
    print("           FINAL BATCH REPORT (Baseline vs Guided)           ")
    print("="*60)
    
    if not df.empty and 'obs_pos_tuple' in df.columns:
        for obs_tuple in df['obs_pos_tuple'].unique():
            subset = df[df['obs_pos_tuple'] == obs_tuple]
            total = len(subset)
            
            # Baseline Stats
            base_coll = subset['base_collision'].mean() * 100
            base_succ = subset['base_success'].mean() * 100
            base_rew = subset['base_reward'].mean()
            
            # Guided Stats
            guide_coll = subset['guided_collision'].mean() * 100
            guide_succ = subset['guided_success'].mean() * 100
            guide_rew = subset['guided_reward'].mean()
            
            print(f"\n[Obstacle at {obs_tuple}] (N={total})")
            print(f"  Collision Rate: Base {base_coll:5.1f}%  ->  Guided {guide_coll:5.1f}%")
            print(f"  Success Rate:   Base {base_succ:5.1f}%  ->  Guided {guide_succ:5.1f}%")
            print(f"  Avg Reward:     Base {base_rew:5.4f} ->  Guided {guide_rew:5.4f}")


    print_conditional_fail_report(df)
    
    return df


if __name__ == "__main__":
    test_obstacles = [
        [-0.3, -0.5],
        [0.2, -0.5],
        [-0.7, -0.2],
        [0.6, -0.2],
        [-0.7, 0.4],
        [0.6, 0.4],
        [-0.3, 0.8],
        [0.2, 0.8],
    ]
    df_results = run_batch_experiments_dp(
        start_seed=0,
        num_seeds=50,
        obstacles=test_obstacles,
        guidance_scale=10.0,
        save_gif=True,
    )

    print_latency_global()
