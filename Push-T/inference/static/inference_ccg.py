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
from core.critic import CollisionPredictionCritic
import cv2
import collections
import imageio
import pandas as pd
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



# loaded_critic = ConditionalHistoryCritic(action_dim, obs_dim, obs_horizon).to(device)
loaded_critic = CollisionPredictionCritic(action_dim, obs_dim, obs_horizon).to(device)

critic_ckpt_path = PROJECT_ROOT / "checkpoints" / "CCG_V6_ep200.pth"
# critic_ckpt_path = PROJECT_ROOT / "checkpoints" / "Robust_critic_3_ep140.pth"

if os.path.exists(critic_ckpt_path):
    loaded_critic.load_state_dict(torch.load(critic_ckpt_path, map_location=device))
    loaded_critic.eval()
    print(f"Successfully loaded model from {critic_ckpt_path}")
else:
    print(f"Error: Model file not found at {critic_ckpt_path}")

critic_net = loaded_critic

trained_critic = critic_net




def run_ccg_inference(obstacle_percentage=0.5, max_guidance_scale=1.0, 
                         show_baseline=True, show_axes=True, seed=42, 
                         save_gif=True, output_dir=PROJECT_ROOT / "outputs" / "ccg_batch_results",
                         fixed_obstacle_pos=None):
    
    result_metrics = {
        "seed": seed,
        "obs_pct": obstacle_percentage if fixed_obstacle_pos is None else -1,
        "obs_pos_str": str(fixed_obstacle_pos) if fixed_obstacle_pos is not None else f"{int(obstacle_percentage*100)}%",
        # Baseline Metrics
        "base_collision": False,
        "base_reward": 0.0,
        "base_success": False,
        "base_min_dist": 0.0,
        # Guided Metrics
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
        
        PHYSICAL_RADIUS = TRAIN_NORM_R * 256.0 
        
        COLLISION_THRESHOLD = PHYSICAL_RADIUS + PHYSICAL_RADIUS # ~30.7 px
        
        VIS_RADIUS = max(2, int(PHYSICAL_RADIUS * vis_scale))
        
        VIS_THRESHOLD_PIXEL = int(COLLISION_THRESHOLD * vis_scale)

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
        dt_val = 1.0 / (pred_horizon - obs_horizon)
        
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
                
                # Drift & Guidance
                t_scalar = np.clip(i * dt_val, 1e-3, 1.0 - 1e-3)
                t = torch.tensor([[t_scalar]], device=device, dtype=torch.float32)
                


                with latency_tracker: 
                    with torch.no_grad():
                        v_pred = ema_si_velocity_net(na, t.view(-1), o_test) 
                        eta_pred = ema_si_denoiser_net(na, t.view(-1), o_test)
                        gamma = gamma_t_si(t).view(-1, 1, 1)
                        g_dot = d_gamma_dt_si(t).view(-1, 1, 1)
                        s_pred = -eta_pred / (gamma + 1e-6)
                        score_coeff = - (gamma * g_dot)
                        base_drift = v_pred + score_coeff * s_pred
                    
                    guidance = torch.zeros_like(base_drift)
                    
                    if (not force_no_guidance) and (obstacle_pos_norm is not None) and (max_guidance_scale > 0):
                        with torch.enable_grad():
                            na_in = na.squeeze(1).detach().requires_grad_(True) 
                            val = critic_net(na_in, t, obs_tensor, o_test)
                            collision_prob = torch.sigmoid(val).detach()
                            k = 4
                            dynamic_scale = max_guidance_scale * torch.pow(collision_prob, k)
                            (grad_v,) = torch.autograd.grad(val.sum(), na_in)
                            guidance = -dynamic_scale * grad_v.unsqueeze(1)
                    
                    na = na + (base_drift + guidance) * dt_val 
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
                    
                    if (not force_no_guidance) and (obstacle_pos_norm is not None):
                        vec_guide = guidance.detach().cpu().numpy().squeeze()
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
    rng_state = torch.get_rng_state()
    base_pixels, base_norms = run_episode(obstacle_pos_norm=None, record_traj=True)
    torch.set_rng_state(rng_state)




    if len(base_norms) > 0:
        real_start_pos = base_norms[0] 
    else:
        real_start_pos = np.array([0.0, 0.0])




    if len(base_norms) == 0 and fixed_obstacle_pos is None: return result_metrics, []
        
    if fixed_obstacle_pos is not None:
        auto_obs_pos = np.array(fixed_obstacle_pos, dtype=np.float32)
    else:
        target_idx = np.clip(int(len(base_norms) * obstacle_percentage), 0, len(base_norms) - 1)
        auto_obs_pos = base_norms[target_idx] 



    result_metrics["agent_start_x"] = float(real_start_pos[0])
    result_metrics["agent_start_y"] = float(real_start_pos[1])




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


def run_batch_experiments(start_seed=240, num_seeds=10, 
                          save_gif=False):
    


    test_obstacles = [
        [-0.3, -0.5], 
        [0.2, -0.5], 
        [-0.7, -0.2], 
        [0.6, -0.2], 
        [-0.7, 0.4],
        [0.6, 0.4],
        [-0.3, 0.8],
        [0.2, 0.8]
    ]


    print(f"=== Starting Batch Experiment (Fixed Obstacles) ===")
    print(f"Seeds: {start_seed} -> {start_seed + num_seeds - 1}")
    print(f"Obstacles: {test_obstacles}")
    print(f"Save GIF: {save_gif}\n")
    
    all_results = []
    
    for i in range(num_seeds):
        seed = start_seed + i
        print(f"\n--- Processing Seed {seed} ({i+1}/{num_seeds}) ---")
        
        for obs_pos in test_obstacles:
            metrics, _ = run_ccg_inference(
                fixed_obstacle_pos=obs_pos,
                max_guidance_scale=4.0, 
                seed=seed,
                save_gif=save_gif,
                output_dir=PROJECT_ROOT / "outputs" / "ccg-p_batch_results_k=4-guidance=4"
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
    
    for obs_pos in test_obstacles:
        obs_str = str(obs_pos)
        subset = df[df['obs_pos_str'] == obs_str]
        total = len(subset)
        
        # Baseline Stats
        base_coll = subset['base_collision'].mean() * 100
        base_succ = subset['base_success'].mean() * 100
        base_rew = subset['base_reward'].mean()
        
        # Guided Stats
        guide_coll = subset['guided_collision'].mean() * 100
        guide_succ = subset['guided_success'].mean() * 100
        guide_rew = subset['guided_reward'].mean()
        
        print(f"\n[Obstacle at {obs_str}] (N={total})")
        print(f"  Collision Rate: Base {base_coll:5.1f}%  ->  Guided {guide_coll:5.1f}%")
        print(f"  Success Rate:   Base {base_succ:5.1f}%  ->  Guided {guide_succ:5.1f}%")
        print(f"  Avg Reward:     Base {base_rew:5.4f} ->  Guided {guide_rew:5.4f}")


    print_conditional_fail_report(df)
    return df


df_results = run_batch_experiments(
    start_seed=0, 
    num_seeds=50, 
    save_gif=True 
)


print_latency_global()