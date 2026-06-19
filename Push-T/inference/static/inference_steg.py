# inference.py

import os
from pathlib import Path
import math
import collections

import numpy as np
import torch

from core.pusht_env import PushTEnv
from core.dataset import PushTDataset, normalize_data, unnormalize_data
from core.models import ConditionalUnet1D
from core.flow_policy import gamma_t_si, d_gamma_dt_si

import cv2
import imageio
from IPython.display import Image, display
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

si_velocity_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim * obs_horizon,
    updownsample_type="Linear",
    sin_embedding_scale=100,
).to(device)

si_denoiser_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim * obs_horizon,
    updownsample_type="Linear",
    sin_embedding_scale=100,
).to(device)



load_pretrained = True
ckpt_path_si = PROJECT_ROOT / "checkpoints" / "pusht_state_1000ep_si.ckpt"

if load_pretrained and os.path.isfile(ckpt_path_si):
    print(f"Found pretrained checkpoint: {ckpt_path_si}")
    checkpoint = torch.load(ckpt_path_si, map_location=device)

    si_velocity_net.load_state_dict(checkpoint['velocity_net'])
    si_denoiser_net.load_state_dict(checkpoint['denoiser_net'])

    ema_si_velocity_net = si_velocity_net
    ema_si_denoiser_net = si_denoiser_net

    print('Pretrained weights loaded for Streaming SI Policy. Ready for inference!')

else:
    print(f"Checkpoint {ckpt_path_si} not found. Please run the training cell below.")




def run_steg_sde_inference(obstacle_percentage=0.5, 
                              fixed_obstacle_pos=None,
                              guidance_scale=1.5,
                              N_ensemble=64,       
                              seed=0, 
                              save_gif=True, 
                              output_dir=PROJECT_ROOT / "outputs" / "steg_ode_results",
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
        # TEG Metrics
        "guided_collision": False,
        "guided_reward": 0.0,
        "guided_success": False,
        "guided_min_dist": 0.0
    }


    latency_tracker = LatencyMeasurer(enabled=False) 

    def run_episode(obstacle_pos_norm=None, record_traj=False, baseline_pixel_traj=None, force_no_guidance=False):
        
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            import random
            random.seed(seed)
        
        try:
            env.reset(seed=seed)
        except:
            env.seed(seed)
            
        obs, info = env.reset()
        # print(f"DEBUG OBS SUM: {obs.sum():.6f}") 
            
        obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
        
        start_img = env.render()
        img_h, img_w = start_img.shape[:2]
        env_window_size = getattr(env, 'window_size', 512)
        vis_scale = img_w / env_window_size
        
        TRAIN_NORM_R = 0.06 
        PHYSICAL_RADIUS = TRAIN_NORM_R * 256.0      
        COLLISION_THRESHOLD = PHYSICAL_RADIUS * 2   
        VIS_RADIUS = max(2, int(PHYSICAL_RADIUS * vis_scale))     
        VIS_THRESHOLD_PIXEL = int(COLLISION_THRESHOLD * vis_scale) 

        K_horizon = 3
        dt_est = 0.15
        activation_distance = 0.4


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

        a = obs[:action_dim]
        na = normalize_data(a, stats=stats['action'])
        na = torch.from_numpy(na).to(device, dtype=torch.float32)
        na_from_prev_chunk = na.unsqueeze(0).unsqueeze(0)
        
        recorded_pixels = []
        recorded_norms = []
        imgs = []
        episode_rewards = []
        
        done = False
        step_idx = 0
        max_steps = 250
        dt_real = 1.0 / (pred_horizon - obs_horizon)
        
        while not done and step_idx < max_steps:
            obs_seq = np.stack(obs_deque)
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            o_test = torch.from_numpy(nobs).to(device, dtype=torch.float32).flatten().unsqueeze(0)
            
            na = na_from_prev_chunk
            
            for i in range(action_horizon):

                a_cpu = na.detach().to('cpu').numpy().squeeze() 
                a_real = unnormalize_data(a_cpu, stats=stats['action']) 
                curr_px = (int(a_real[0] * vis_scale), int(a_real[1] * vis_scale))
                
                if record_traj:
                    recorded_pixels.append(curr_px)
                    recorded_norms.append(a_cpu)
                
                if obs_real_pos_arr is not None:
                    dist = np.linalg.norm(a_real - obs_real_pos_arr)
                    if dist < min_dist_to_obs:
                        min_dist_to_obs = dist
                
                obs, reward, done, _, info = env.step(a_real)
                obs_deque.append(obs)
                episode_rewards.append(reward)
                
                step_idx += 1
                if done or step_idx >= max_steps: break
                

                t_scalar = np.clip(i * dt_real, 1e-3, 1.0 - 1e-3)
                t_real = torch.tensor([[t_scalar]], device=device, dtype=torch.float32)
                

                with latency_tracker: 
                    with torch.no_grad():
                        v_real = ema_si_velocity_net(na, t_real.view(-1), o_test)
                        eta_real = ema_si_denoiser_net(na, t_real.view(-1), o_test)
                        gamma = gamma_t_si(t_real).view(-1, 1, 1)
                        g_dot = d_gamma_dt_si(t_real).view(-1, 1, 1)
                        s_real = -eta_real / (gamma + 1e-6)
                        
                        score_coeff =  - (gamma * g_dot)
                        base_drift = v_real + score_coeff * s_real
                    
                    dist_to_obs = 999.0
                    if obstacle_pos_norm is not None:
                        dist_to_obs = torch.norm(na.squeeze() - obs_tensor.squeeze()).item()

                    grad_teg = torch.zeros_like(base_drift)
                    dynamic_scale = 0.0
                    
                    if (not force_no_guidance) and (obstacle_pos_norm is not None) and (dist_to_obs < activation_distance):
                        with torch.enable_grad():
                            a_curr = na.squeeze(1).detach().requires_grad_(True)
                            
                            # Batch Expansion
                            a_ens = a_curr.repeat(N_ensemble, 1)
                            t_ens = t_real.view(-1).repeat(N_ensemble)
                            cond_ens = o_test.repeat(N_ensemble, 1)
                            obs_ens = obs_tensor.repeat(N_ensemble, 1)
                            
                            cum_cost = torch.zeros(N_ensemble, device=device)
                            curr_a_sim = a_ens
                            curr_t_sim = t_ens
                            
                            for k_step in range(K_horizon):
                                t_input = torch.clamp(curr_t_sim, 0.05, 0.95)
                                v_p = ema_si_velocity_net(curr_a_sim.unsqueeze(1), t_input, cond_ens).squeeze(1)
                                eta_p = ema_si_denoiser_net(curr_a_sim.unsqueeze(1), t_input, cond_ens).squeeze(1)
                                gamma = gamma_t_si(t_input).view(-1, 1)
                                g_dot = d_gamma_dt_si(t_input).view(-1, 1)
                                s_p = -eta_p / (gamma + 1e-6)
                                
                                sigma_est = 0.2
                                score_coeff_est = 0.5 * (sigma_est**2) - (gamma * g_dot)
                                b_drift_est = v_p + score_coeff_est * s_p
                                
                                dist_sq = ((curr_a_sim - obs_ens)**2).sum(dim=-1)
                                step_cost = 0.5 * torch.exp(-dist_sq / (2 * 0.1**2)) 
                                cum_cost += step_cost * dt_est
                                
                                # Update Simulation
                                noise_sim = torch.randn_like(curr_a_sim)
                                curr_a_sim = curr_a_sim + b_drift_est * dt_est + sigma_est * math.sqrt(dt_est) * noise_sim
                                curr_t_sim = curr_t_sim + dt_est
                            
                            # Gradient Calculation
                            neg_cost = -cum_cost
                            log_utility = torch.logsumexp(neg_cost, dim=0)
                            (g,) = torch.autograd.grad(log_utility, a_curr)
                            grad_teg = g.unsqueeze(1)
                        
                        dynamic_scale = guidance_scale * (1.0 - dist_to_obs / activation_distance)

                    total_drift = base_drift + dynamic_scale * grad_teg
                    
                    na = na + total_drift * dt_real
                    na = na.detach()



                if save_gif and not record_traj:
                    img_rgb = env.render().copy()
                    if show_axes: draw_coordinate_axes(img_rgb)
                    if show_baseline and baseline_pixel_traj is not None:
                        pts = np.array(baseline_pixel_traj, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img_rgb, [pts], False, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    if obs_pixel is not None:
                        cv2.circle(img_rgb, obs_pixel, VIS_RADIUS, (255, 255, 0), -1) 
                        cv2.circle(img_rgb, obs_pixel, VIS_THRESHOLD_PIXEL, (0, 0, 255), 1) 
                        
                    # 3. Agent (~8px)
                    cv2.circle(img_rgb, curr_px, VIS_RADIUS, (0, 0, 255), -1)
                    
                    if (not force_no_guidance) and (obstacle_pos_norm is not None) and (dynamic_scale > 0):
                        vec_guide = (dynamic_scale * grad_teg).detach().cpu().numpy().squeeze()
                        draw_vector_arrow(img_rgb, curr_px, vec_guide, (0, 0, 255), scale=150.0)
                        
                    imgs.append(img_rgb)

            na_from_prev_chunk = na
            if done: break

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
    
    # 1. Warm-up Run
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


    # 2. Baseline Evaluation (No Guidance)
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
    latency_tracker = LatencyMeasurer(warmup_steps=10, chunk_size=1, enabled=True)



    # 3. STEG Evaluation (With Guidance)
    print("Running STEG Inference...")
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
    print("\n[STEG Result]")
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


import pandas as pd

def run_batch_experiments_steg(start_seed=0, num_seeds=10, 
                               obstacles=None, 
                               save_gif=False):
    
    if obstacles is None:
        obstacles = []

    print(f"=== Starting Batch Experiment (STEG Honest Mode) ===")
    print(f"Seeds: {start_seed} -> {start_seed + num_seeds - 1}")
    print(f"Obstacles: {len(obstacles)} positions")
    print(f"Save GIF: {save_gif}\n")
    
    all_results = []
    
    for i in range(num_seeds):
        seed = start_seed + i
        print(f"\n--- Processing Seed {seed} ({i+1}/{num_seeds}) ---")
        
        for obs_pos in obstacles:
            metrics, _ = run_steg_sde_inference(
                fixed_obstacle_pos=obs_pos,
                guidance_scale=7,
                N_ensemble=64,
                seed=seed,
                save_gif=save_gif,
                output_dir=PROJECT_ROOT / "outputs" / "steg_ode_batch_results-7"
            )
            
            metrics['obs_pos_tuple'] = tuple(obs_pos)
            all_results.append(metrics)
            
            base_status = "FAIL" if metrics['base_collision'] else "PASS"
            guide_status = "FAIL" if metrics['guided_collision'] else "PASS"
            
            print(f"  [Pos {obs_pos}] Base: {metrics['base_reward']:.2f}({base_status}) -> STEG: {metrics['guided_reward']:.2f}({guide_status})")

    df = pd.DataFrame(all_results)
    
    print("\n" + "="*60)
    print("           FINAL STEG BATCH REPORT           ")
    print("="*60)
    
    if not df.empty:
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
            print(f"  Collision Rate: Base {base_coll:5.1f}%  ->  STEG {guide_coll:5.1f}%")
            print(f"  Success Rate:   Base {base_succ:5.1f}%  ->  STEG {guide_succ:5.1f}%")
            print(f"  Avg Reward:     Base {base_rew:5.4f} ->  STEG {guide_rew:5.4f}")

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

    df_steg_results = run_batch_experiments_steg(
        start_seed=0,
        num_seeds=50,
        obstacles=test_obstacles,
        save_gif=True,
    )

    print_latency_global()
