import numpy as np
import torch
import cv2
import collections
import math
import os
from pathlib import Path
import imageio
import pandas as pd
from IPython.display import Image, display
from core.my_utils import * 


from core.pusht_env import PushTEnv
from core.dataset import PushTDataset, normalize_data, unnormalize_data
from core.models import ConditionalUnet1D
from core.flow_policy import gamma_t_si, d_gamma_dt_si
import random


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



class DynamicObstacleManager:
    def __init__(self, mode, start_pos_norm=None, baseline_traj=None, dt=0.05, vis_scale=1.0, seed=None):
        self.mode = mode
        self.dt = dt
        self.vis_scale = vis_scale
        
        
        self.sine_amp = 40.0      
        self.sine_freq = 0.03     

        self.goal_center = np.array([256.0, 256.0]) 
        self.safe_radius = 300.0  
        
        # Mode 3: Chase 
        self.chase_speed = 1.0       
        self.chase_turn_rate = 0.4  
        
        if seed is None:
            seed = 42
        self.rng = np.random.RandomState(seed)
        
        if start_pos_norm is not None:
            self.pos = unnormalize_data(np.array(start_pos_norm), stats=stats['action'])
        else:
            self.pos = np.zeros(2)

        if self.mode == 3:
            self.pos = self.rng.uniform(50.0, 460.0, size=2)

        self.velocity = np.zeros(2)
        self.baseline_traj = baseline_traj
        
        
        self.target_intercept_pt = None
        self.steps_to_impact = 0

        if self.mode == 2 and self.baseline_traj is not None:
            intercept_idx = min(len(self.baseline_traj) - 1, 90) 
            self.target_intercept_pt = self.baseline_traj[intercept_idx]
            
            offset = np.array([60.0, 60.0]) 
            self.pos = self.target_intercept_pt + offset 
            self.steps_to_impact = intercept_idx + 1
            
            self.travel_duration = 50 
            
            self.velocity = (self.target_intercept_pt - self.pos) / self.travel_duration
            # self.velocity = (self.target_intercept_pt - self.pos) / self.steps_to_impact

        if self.mode == 1 and self.baseline_traj is not None:
            mid_idx = min(len(self.baseline_traj) - 1, 60)
            self.anchor_pos = self.baseline_traj[mid_idx]
            p1 = self.baseline_traj[max(0, mid_idx-5)]
            p2 = self.baseline_traj[min(len(self.baseline_traj)-1, mid_idx+5)]
            tangent = p2 - p1
            tangent /= (np.linalg.norm(tangent) + 1e-6)
            self.perp_vec = np.array([-tangent[1], tangent[0]]) 

        self._enforce_spawn_safety()

        self.start_pos = self.pos.copy()

    def _enforce_spawn_safety(self):
        dist = np.linalg.norm(self.pos - self.goal_center)
        if dist < self.safe_radius:
            # print(f"[ObstacleManager] Relocating from Goal area...")
            if dist < 1e-6:
                direction = np.array([1.0, 0.0])
            else:
                direction = (self.pos - self.goal_center) / dist
            
            new_dist = self.safe_radius + 20.0
            self.pos = self.goal_center + direction * new_dist
            self.pos = np.clip(self.pos, 10, 502)

    def update(self, step_idx, agent_pos_real=None):
        if self.mode == 0: # Static
            return self.pos

        elif self.mode == 1:
            val = math.sin(step_idx * self.sine_freq)
            self.pos = self.anchor_pos + self.sine_amp * val * self.perp_vec

        elif self.mode == 2:
            dist_to_target = np.linalg.norm(self.target_intercept_pt - self.pos)
            
            if dist_to_target < 2.0 or step_idx > self.steps_to_impact:
                self.velocity = np.zeros(2)
            
            self.pos += self.velocity
            
        elif self.mode == 3:
            if agent_pos_real is not None:
                vec_to_agent = agent_pos_real - self.pos
                dist = np.linalg.norm(vec_to_agent)
                
                if dist > 1e-1:
                    desired_velocity = (vec_to_agent / dist) * self.chase_speed
                    self.velocity = (1 - self.chase_turn_rate) * self.velocity + \
                                    self.chase_turn_rate * desired_velocity
                    self.pos += self.velocity
                else:
                    self.velocity *= 0.8

        return self.pos

def run_clean_baseline(seed, max_steps=200):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try: obs, info = env.reset(seed=seed)
        except: env.seed(seed); obs = env.reset(); info = {}
        if hasattr(env.action_space, 'seed'): env.action_space.seed(seed)
    else:
        obs, info = env.reset()
        
    if isinstance(obs, tuple): obs, info = obs

    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    
    a = obs[:action_dim]
    na = normalize_data(a, stats=stats['action'])
    na = torch.from_numpy(na).to(device, dtype=torch.float32)
    na_from_prev_chunk = na.unsqueeze(0).unsqueeze(0)
    
    trajectory = [] 
    
    done = False
    step_idx = 0
    dt_val = 1.0 / (pred_horizon - obs_horizon)
    
    while not done and step_idx < max_steps:
        obs_seq = np.stack(obs_deque)
        nobs = normalize_data(obs_seq, stats=stats['obs'])
        o_test = torch.from_numpy(nobs).to(device, dtype=torch.float32).flatten().unsqueeze(0)
        
        na = na_from_prev_chunk
        
        for i in range(action_horizon):
            a_cpu = na.detach().to('cpu').numpy().squeeze()
            a_real = unnormalize_data(a_cpu, stats=stats['action'])
            
            trajectory.append(a_real)
            
            step_res = env.step(a_real)
            if len(step_res) == 4: obs, reward, done, info = step_res
            else: obs, reward, done, _, info = step_res
            
            obs_deque.append(obs)
            step_idx += 1
            if done or step_idx >= max_steps: break
            t_scalar = np.clip(i * dt_val, 1e-3, 1.0 - 1e-3)
            t = torch.tensor([[t_scalar]], device=device, dtype=torch.float32)
            
            with torch.no_grad():
                v_pred = ema_si_velocity_net(na, t.view(-1), o_test) 
                eta_pred = ema_si_denoiser_net(na, t.view(-1), o_test)
                gamma = gamma_t_si(t).view(-1, 1, 1)
                g_dot = d_gamma_dt_si(t).view(-1, 1, 1)
                s_pred = -eta_pred / (gamma + 1e-6)
                score_coeff = - (gamma * g_dot)
                base_drift = v_pred + score_coeff * s_pred
                
            na = na + base_drift * dt_val 
            na = na.detach()

        na_from_prev_chunk = na
        if done: break
        
    return np.array(trajectory)

def run_guided_inference_visualized(obstacle_mode=0, start_obs_pos_norm=None, max_guidance_scale=0.3, seed=None):

    baseline_traj = None
    if obstacle_mode in [1, 2]: 
        baseline_traj = run_clean_baseline(seed=seed, max_steps=200)
        if len(baseline_traj) < 10:
             print("Warning: Baseline trajectory too short.")
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            obs, info = env.reset(seed=seed)
        except TypeError:
            env.seed(seed)
            obs = env.reset()
            info = {} 
        if hasattr(env.action_space, 'seed'):
            env.action_space.seed(seed)
    else:
        obs, info = env.reset()

    if isinstance(obs, tuple): 
        obs, info = obs 
        
    env_window_size = getattr(env, 'window_size', 512)
    start_img = env.render() 
    img_h, img_w = start_img.shape[:2] 
    vis_scale = img_w / env_window_size
    
    obs_manager = DynamicObstacleManager(
        mode=obstacle_mode, 
        start_pos_norm=start_obs_pos_norm,
        baseline_traj=baseline_traj,
        vis_scale=vis_scale,
        seed=seed 
    )

    TRAIN_NORM_R = 0.06
    PHYSICAL_RADIUS = TRAIN_NORM_R * 256.0
    COLLISION_THRESHOLD = PHYSICAL_RADIUS * 2.0
    min_dist_to_obs = float('inf')
    episode_rewards = []

    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    imgs = []
    
    a = obs[:action_dim]
    na = normalize_data(a, stats=stats['action'])
    na = torch.from_numpy(na).to(device, dtype=torch.float32)
    na_from_prev_chunk = na.unsqueeze(0).unsqueeze(0) 
    
    done = False
    step_idx = 0
    max_steps = 250
    dt_val = 1.0 / (pred_horizon - obs_horizon)
    arrow_vis_scale = 100.0 
    
    success = False 
    force_no_guidance = False

    while not done and step_idx < max_steps:
        obs_seq = np.stack(obs_deque)
        nobs = normalize_data(obs_seq, stats=stats['obs'])
        o_test = torch.from_numpy(nobs).to(device, dtype=torch.float32).flatten().unsqueeze(0)
        
        na = na_from_prev_chunk 
        
        for i in range(action_horizon):
            a_cpu = na.detach().to('cpu').numpy().squeeze()
            a_real = unnormalize_data(a_cpu, stats=stats['action'])
            
            obs_real_pos = obs_manager.update(step_idx, agent_pos_real=a_real)
            
            obs_norm = normalize_data(obs_real_pos, stats=stats['action'])
            obs_tensor = torch.tensor([obs_norm], device=device, dtype=torch.float32)
            obstacle_pos_norm = obs_norm
            
            obs_pixel_x = int(obs_real_pos[0] * vis_scale)
            obs_pixel_y = int(obs_real_pos[1] * vis_scale)
            obs_pixel = (obs_pixel_x, obs_pixel_y)

            if obs_real_pos is not None:
                dist = np.linalg.norm(a_real - obs_real_pos)
                if dist < min_dist_to_obs:
                    min_dist_to_obs = dist

            # Env Step
            step_res = env.step(a_real)
            if len(step_res) == 4: obs, reward, done, info = step_res
            else: obs, reward, done, _, info = step_res 
            
            obs_deque.append(obs)
            episode_rewards.append(reward)
            
            
            step_idx += 1
            if done or step_idx >= max_steps: break
            t_scalar = np.clip(i * dt_val, 1e-3, 1.0 - 1e-3)
            t = torch.tensor([[t_scalar]], device=device, dtype=torch.float32)
            
            # Base Drift
            with torch.no_grad():
                v_pred = ema_si_velocity_net(na, t.view(-1), o_test) 
                eta_pred = ema_si_denoiser_net(na, t.view(-1), o_test)
                gamma = gamma_t_si(t).view(-1, 1, 1)
                g_dot = d_gamma_dt_si(t).view(-1, 1, 1)
                s_pred = -eta_pred / (gamma + 1e-6)
                score_coeff = - (gamma * g_dot)
                base_drift = v_pred + score_coeff * s_pred

            # --- Naive Force Logic Pre-calculation ---
            activation_dist = 0.3
            guidance_scale = max_guidance_scale
            curr_dist_norm = 999.0
            if obstacle_pos_norm is not None:
                curr_dist_norm = torch.norm(na.squeeze() - obs_tensor.squeeze()).item()

            guidance_force = torch.zeros_like(base_drift)
            
            if (not force_no_guidance) and (obstacle_pos_norm is not None):
                if curr_dist_norm < activation_dist:
                    diff_vec = na.squeeze() - obs_tensor.squeeze()
                    
                    direction = diff_vec / (curr_dist_norm + 1e-6)
                    
                    strength = (1.0 - curr_dist_norm / activation_dist) ** 2
                    
                    magnitude = guidance_scale * strength
                    
                    # reshape to [1, 1, 2] match base_drift
                    guidance_force = (direction * magnitude).view(1, 1, 2)
            
            # 3. Apply Drift (Base + Naive Force)
            total_drift = base_drift + guidance_force
            
            # Update Agent Position
            na = na + total_drift * dt_val 
            na = na.detach()
            guidance = guidance_force
            collision_prob = torch.tensor(curr_dist_norm, device=device, dtype=torch.float32)

            img_rgb = env.render().copy()
            
            cv2.circle(img_rgb, obs_pixel, 8, (255, 255, 0), -1) 
            cv2.circle(img_rgb, obs_pixel, 8, (0, 0, 0), 2)
            label_list = ["Static", "Sine", "Intercept", "Chase"]
            if obstacle_mode < len(label_list):
                label = label_list[obstacle_mode]
            else:
                label = "Unknown"
                
            cv2.putText(img_rgb, f"OBS: {label}", (obs_pixel[0]-30, obs_pixel[1]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            curr_px = (int(a_real[0] * vis_scale), int(a_real[1] * vis_scale))
            cv2.circle(img_rgb, curr_px, 6, (0, 0, 255), -1)
            cv2.putText(img_rgb, f"Prob: {collision_prob.item():.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            vec_base = base_drift.detach().cpu().numpy().squeeze()
            vec_guide = guidance.detach().cpu().numpy().squeeze()
            draw_vector_arrow(img_rgb, curr_px, vec_base, (255, 0, 0), scale=arrow_vis_scale, thickness=2)
            draw_vector_arrow(img_rgb, curr_px, vec_guide, (0, 0, 255), scale=arrow_vis_scale, thickness=2)
            
            imgs.append(img_rgb)
            
            # D. Update

        na_from_prev_chunk = na
        if done: break
        
    max_reward = max(episode_rewards) if episode_rewards else 0.0
    is_collision = False
    if min_dist_to_obs != float('inf'):
        is_collision = (min_dist_to_obs < COLLISION_THRESHOLD)
    is_success = (not is_collision) and (max_reward > 0.85)

    metrics = {
        "collision": is_collision,
        "reward": float(max_reward),
        "success": bool(is_success),
        "min_dist": float(min_dist_to_obs) if min_dist_to_obs != float('inf') else float('inf'),
        "collision_threshold": float(COLLISION_THRESHOLD),
    }

    return imgs, metrics


if __name__ == "__main__":
    output_dir = PROJECT_ROOT / "outputs" / "naive-mode1-guidance=7"
    os.makedirs(output_dir, exist_ok=True)

    base_seed = 500
    num_trials = 50
    mode_to_test = 1

    print(f"Starting Batch Experiment: Mode {mode_to_test}, Trials {num_trials}")

    success_count = 0
    collision_count = 0
    reward_list = []

    for i in range(num_trials):
        current_seed = base_seed + i

        print(f"--- Trial {i+1}/{num_trials} (Seed: {current_seed}) ---")

        imgs, metrics = run_guided_inference_visualized(
            obstacle_mode=mode_to_test,
            start_obs_pos_norm=[0.5, 0.5],
            max_guidance_scale=7.0,
            seed=current_seed,
        )

        print(f"Reward(max): {metrics['reward']:.3f} | Collision: {metrics['collision']} | "
              f"Success(no collision & reward>0.85): {metrics['success']} | "
              f"MinDist: {metrics['min_dist']:.2f} (thr={metrics['collision_threshold']:.2f})")

        reward_list.append(metrics["reward"])
        if metrics["collision"]:
            collision_count += 1
        if metrics["success"]:
            success_count += 1

        if i < 50:
            save_name = f"trial_{i}_seed_{current_seed}_mode_{mode_to_test}.gif"
            save_path = os.path.join(output_dir, save_name)
            if len(imgs) > 0:
                imageio.mimsave(save_path, imgs, fps=20, loop=0)
                print(f"Saved GIF: {save_path}")

    print(f"\n========================================")
    print(f"Guided Collision Rate: {collision_count}/{num_trials} ({(collision_count/num_trials)*100:.1f}%)")
    print(f"Guided Success Rate (no collision & reward>0.85): {success_count}/{num_trials} ({(success_count/num_trials)*100:.1f}%)")
    avg_reward = (sum(reward_list) / len(reward_list)) if len(reward_list) > 0 else 0.0
    print(f"Guided Avg Reward(max): {avg_reward:.3f}")
    print(f"========================================")
