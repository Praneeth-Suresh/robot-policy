import os
import random
import numpy as np
import torch
import torch.nn as nn
import cv2
import collections
import math
import imageio
from tqdm.auto import tqdm

from core.config import *  # noqa: F401,F403  (device / obs_horizon / action_horizon / pred_horizon / RESULTS_BASE_DIR, etc.)

# Lazy singleton: zero side effects on import; env built on first access
from core.robotic_mimic_env import get_env
_env = get_env()
wrapper       = _env.wrapper
dataset_path  = _env.dataset_path
CONTROL_DELTA = _env.control_delta

from core.config import STEG_HP, STEG_RESULTS_DIR
from core.dataset import dataset
from core.flow_policy import gamma_t_si, d_gamma_dt_si
from core.my_utils import (
    _infer_task_name_from_dataset_path,
    _task_cfg,
    _get_eef_pos_start,
)

from inference.inference_si import ema_si_velocity_net, ema_si_denoiser_net

_steg_task_name = _infer_task_name_from_dataset_path(dataset_path)
_steg_cfg = _task_cfg(_steg_task_name)
print(f"[STEG] task = {_steg_task_name} | cfg = {_steg_cfg}")


# =========================================================
# Upstream dependencies
# =========================================================
def get_drift(x, t, global_cond, sigma_infer, eps=1e-6):
    """
    Compute corrected drift b(x, t) = v + score_correction
    x: (B, 1, 10)
    """
    if t.ndim == 0: t = t.unsqueeze(0)
    
    # Predict v and eta
    v_pred = ema_si_velocity_net(sample=x, timestep=t, global_cond=global_cond)
    eta_pred = ema_si_denoiser_net(sample=x, timestep=t, global_cond=global_cond)
    
    # Unified shape handling
    if v_pred.ndim == 2: v_pred = v_pred.unsqueeze(1)
    if eta_pred.ndim == 2: eta_pred = eta_pred.unsqueeze(1)

    # gamma and its derivative
    gamma = gamma_t_si(t).view(-1, 1, 1).to(device)
    gamma_dot = d_gamma_dt_si(t).view(-1, 1, 1).to(device)
    
    # Score and drift coefficient
    s_pred = -eta_pred / (gamma + eps)
    score_coeff = 0.5 * (sigma_infer ** 2) - (gamma * gamma_dot)
    
    b = v_pred + score_coeff * s_pred
    return b.squeeze(1)

def world_to_pixel_mujoco(sim, pos_3d, camera_name, img_h, img_w):
    """Project 3D point to pixels using MuJoCo camera intrinsics."""
    try:
        cam_id = sim.model.camera_name2id(camera_name)
        cam_pos = sim.data.cam_xpos[cam_id]
        cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)
        
        pos_rel = pos_3d - cam_pos
        pos_cam_opengl = cam_mat.T @ pos_rel
        
        x, y, z = pos_cam_opengl[0], -pos_cam_opengl[1], -pos_cam_opengl[2]
        
        if z < 0.01: return None
            
        fovy = sim.model.cam_fovy[cam_id]
        f = 0.5 * img_h / np.tan(fovy * np.pi / 360)
        cx, cy = img_w / 2, img_h / 2
        
        u = int(cx + f * x / z)
        v = int(cy + f * y / z)
        
        if 0 <= u < img_w and 0 <= v < img_h:
            return (u, v)
        return None
    except Exception as e:
        return None

# =========================================================
# 3. Inference with SSIP-TEG strategy
# =========================================================

def run_ssip_teg_inference(
    obstacle_pos_phys, 
    # --- TEG parameters ---
    guidance_scale=15.0,     
    activation_dist=0.6,
    N_ensemble=16,        
    update_interval=4,
    K_horizon=3,           # forward simulation steps
    # --- Common parameters ---
    seed=0,                  
    collect_trajectory=False, # collect trajectory (for base run)
    render: bool = True        # render frames (disable in batch runs for speed)
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    wrapper.seed(seed)
    # =========================================================
    # Dynamic start index of 'robot0_eef_pos' in flat obs (no hard-coded start_dim)
    key_dims = []
    for key in wrapper.obs_keys:
        dummy_obs = wrapper.env.get_observation()[key]
        dim = np.array(dummy_obs).flatten().shape[0]
        key_dims.append(dim)
    target_key = 'robot0_eef_pos'
    if target_key not in wrapper.obs_keys:
        raise ValueError(f"{target_key} not found in obs keys; current obs_keys={wrapper.obs_keys}")
    _eef_pos_start = int(sum(key_dims[: wrapper.obs_keys.index(target_key)]))

    def _raw_to_norm(raw: np.ndarray, min_v: np.ndarray, max_v: np.ndarray) -> np.ndarray:
        """Map raw action to action_normalizer space [-1, 1] (matches MinMaxNormalizer)."""
        raw = raw.astype(np.float32)
        min_v = min_v.astype(np.float32)
        max_v = max_v.astype(np.float32)
        denom = (max_v - min_v + 1e-6)
        norm01 = (raw - min_v) / denom
        norm01 = np.clip(norm01, 0.0, 1.0)
        return norm01 * 2.0 - 1.0

    is_delta = bool(CONTROL_DELTA) if 'CONTROL_DELTA' in globals() and CONTROL_DELTA is not None else True

    # --- 1. Prepare obstacle data ---
    if obstacle_pos_phys is None:
        obstacle_pos_phys = np.array([10.0, 10.0, 10.0]) # Dummy position

    # Ensure obstacle coordinates are float32 tensors
    obs_phys_tensor = torch.tensor(obstacle_pos_phys, device=device, dtype=torch.float32)

    # --- 2. Environment reset ---
    obs = wrapper.reset()
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    imgs = []
    if (not collect_trajectory) and render:
        imgs = [wrapper.render()]
    rewards = []
    
    trajectory_log = [] 
    has_collided = False
    
    # Physical collision threshold
    COLLISION_THRESHOLD = 0.02

    # Initial action
    curr_eef_pos = obs[_eef_pos_start : _eef_pos_start + 3].astype(np.float32)
    pos_min = dataset.action_normalizer.min_val[:3].astype(np.float32)
    pos_max = dataset.action_normalizer.max_val[:3].astype(np.float32)
    # delta: initial position channel is zero increment; absolute: use current eef_pos
    raw_pos = np.zeros(3, dtype=np.float32) if is_delta else curr_eef_pos
    norm_pos = _raw_to_norm(raw_pos, pos_min, pos_max)

    na = torch.zeros((1, 1, 10), device=device, dtype=torch.float32)
    na[0, 0, :3] = torch.from_numpy(norm_pos).to(device, dtype=torch.float32)
    # Gripper: -1 in normalized space often maps to raw minimum (usually open)
    na[0, 0, 9] = -1.0
    # 6D rotation: avoid degeneracy (all zeros -> NaN / instability)
    rot6d_raw = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)  # stable identity 6D
    rot_min = dataset.action_normalizer.min_val[3:9].astype(np.float32)
    rot_max = dataset.action_normalizer.max_val[3:9].astype(np.float32)
    rot6d_raw = np.clip(rot6d_raw, rot_min, rot_max)
    rot6d_norm = _raw_to_norm(rot6d_raw, rot_min, rot_max)
    na[0, 0, 3:9] = torch.from_numpy(rot6d_norm).to(device, dtype=torch.float32)
    # Numerical safety
    na = torch.nan_to_num(na, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
    na_from_prev_chunk = na
    
    # State (TEG cache)
    cached_grad = torch.zeros_like(na).squeeze(1)
    cached_scale = 0.0          
    
    done = False
    step_idx = 0
    max_steps = 400
    dt_val = 1.0 / (pred_horizon - obs_horizon)
    dt_est = 0.10 # TEG simulation step
    sigma_infer = 0.05 
    teg_cost_scale = 50.0 # internal cost scaling

    # Normalization params for differentiable denormalization
    min_v_tensor = torch.from_numpy(dataset.action_normalizer.min_val).to(device, dtype=torch.float32)
    max_v_tensor = torch.from_numpy(dataset.action_normalizer.max_val).to(device, dtype=torch.float32)
    scale_v_tensor = max_v_tensor - min_v_tensor
    scale_v_tensor[scale_v_tensor==0] = 1.0

    with torch.no_grad():
        # Progress bar
        pbar = tqdm(total=max_steps, desc=f"Seed {seed}", disable=collect_trajectory)
        
        while not done and step_idx < max_steps:
            
            # --- Prepare observation ---
            obs_seq = np.stack(obs_deque)
            nobs = dataset.obs_normalizer.normalize(obs_seq)
            cond_flat = torch.from_numpy(nobs).to(device, dtype=torch.float32).flatten(start_dim=0).unsqueeze(0)
            o_test = cond_flat 
            
            na = na_from_prev_chunk
            # [Format consistency] training latent in [-1,1]; integration may drift or produce NaN/Inf
            na = torch.nan_to_num(na, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
            
            # --- Execute action chunk ---
            for i in range(action_horizon):
                t_scalar = np.clip(i * dt_val, 1e-3, 1.0 - 1e-3)
                
                # --- A. Physical distance and collision ---
                # True position from robot0_eef_pos in obs (delta/absolute compatible)
                curr_eef_pos = obs[_eef_pos_start : _eef_pos_start + 3].astype(np.float32)
                eef_pos_now_t = torch.from_numpy(curr_eef_pos).to(device=device, dtype=torch.float32)
                dist_to_obs = torch.norm(eef_pos_now_t - obs_phys_tensor).item()

                if collect_trajectory:
                    trajectory_log.append(curr_eef_pos.copy())

                if dist_to_obs < COLLISION_THRESHOLD:
                    has_collided = True

                # --- B. TEG guidance (steering) ---
                effective_scale = 0.0 if collect_trajectory else guidance_scale

                # Recompute gradient when close + interval elapsed + guidance enabled
                should_compute = (dist_to_obs < activation_dist) and (step_idx % update_interval == 0) and (effective_scale > 0)

                if should_compute:
                    with torch.enable_grad():
                        # 1. Gradient input
                        na_in = na.squeeze(1).detach().requires_grad_(True)
                        
                        # 2. Expand batch for parallel simulation
                        na_ens = na_in.repeat(N_ensemble, 1) 
                        cond_ens = o_test.repeat(N_ensemble, 1)
                        # Ensure time is float32
                        t_ens = torch.tensor([t_scalar], device=device, dtype=torch.float32).repeat(N_ensemble)
                        
                        curr_na_sim = na_ens
                        curr_t_sim = t_ens
                        cum_cost = torch.zeros(N_ensemble, device=device, dtype=torch.float32)
                        # delta: accumulate increments to position for cost; absolute: use action position directly
                        sim_pos = eef_pos_now_t.view(1, 3).repeat(N_ensemble, 1)
                        
                        # 3. Forward rollout loop
                        for k_step in range(K_horizon):
                            # [Format consistency] keep latent in [-1,1] and numerically safe during rollout
                            curr_na_sim = torch.nan_to_num(curr_na_sim, nan=0.0, posinf=0.0, neginf=0.0)
                            curr_na_sim = torch.clamp(curr_na_sim, -1.0, 1.0)
                            t_input = torch.clamp(curr_t_sim, 0.05, 0.95)
                            
                            # Network prediction
                            v_p = ema_si_velocity_net(curr_na_sim.unsqueeze(1), t_input, cond_ens).squeeze(1)
                            eta_p = ema_si_denoiser_net(curr_na_sim.unsqueeze(1), t_input, cond_ens).squeeze(1)
                            
                            # Compute coefficients
                            gamma = gamma_t_si(t_input).view(-1, 1)
                            g_dot = d_gamma_dt_si(t_input).view(-1, 1)
                            s_p = -eta_p / (gamma + 1e-6)
                            sigma_est = 0.05 
                            score_coeff = 0.5 * (sigma_est**2) - (gamma * g_dot)
                            b_drift = v_p + score_coeff * s_p
                            
                            # --- Physical-space cost ---
                            # Differentiable denormalization
                            denorm_sim = ((curr_na_sim + 1) / 2) * scale_v_tensor + min_v_tensor
                            if is_delta:
                                delta_xyz = denorm_sim[:, :3]
                                sim_pos = sim_pos + delta_xyz
                                sim_xyz = sim_pos
                            else:
                                sim_xyz = denorm_sim[:, :3] # XYZ position only
                            
                            # Gaussian repulsion cost vs obstacle
                            dist_sq = ((sim_xyz - obs_phys_tensor)**2).sum(dim=-1)
                            cost_step = teg_cost_scale * torch.exp(- dist_sq / (2 * 0.05**2)) # 0.05 obstacle radius
                            cum_cost = cum_cost + cost_step * dt_est
                            
                            # --- SDE update (with noise) ---
                            noise_step = torch.randn_like(curr_na_sim)
                            diffusion_term = sigma_est * math.sqrt(dt_est) * noise_step
                            curr_na_sim = curr_na_sim + b_drift * dt_est + diffusion_term
                            
                            curr_t_sim = curr_t_sim + dt_est
                            
                        # 4. Backprop for gradient
                        neg_cost = -cum_cost
                        log_utility = torch.logsumexp(neg_cost, dim=0)
                        
                        (g,) = torch.autograd.grad(log_utility, na_in)
                        
                        # Update cache
                        cached_grad = torch.nan_to_num(g.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                        
                    # Guidance strength increases as distance decreases
                    cached_scale = sigma_infer**2 * effective_scale * max(0, (1.0 - dist_to_obs / activation_dist))
                
                elif dist_to_obs >= activation_dist:
                    # Far away: clear cache
                    cached_scale = 0.0
                    cached_grad = torch.zeros_like(na).squeeze(1)

                # --- C. Drift computation and update ---
                t_tensor = torch.tensor([t_scalar], device=device, dtype=torch.float32)
                # [Format consistency] drift network input in training normalization range
                na = torch.nan_to_num(na, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
                base_drift = get_drift(na, t_tensor, o_test, sigma_infer)
                
                # Add SSIP-TEG gradient
                total_drift = base_drift + cached_scale * cached_grad#.unsqueeze(0)
                
                noise = torch.randn_like(total_drift)
                na = na + total_drift.unsqueeze(1) * dt_val + sigma_infer * math.sqrt(dt_val) * noise.unsqueeze(1)
                # [Format consistency] sanitize and clamp after update
                na = torch.nan_to_num(na, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
                
                # --- D. Step & Render ---
                a_real = dataset.get_unnormalized_action(na).squeeze()
                obs, reward, done, _ = wrapper.step(a_real)
                obs_deque.append(obs)
                rewards.append(reward)
                
                if not collect_trajectory:
                    if not render:
                        # Disable rendering in batch runs for speed
                        step_idx += 1
                        pbar.update(1)
                        if done or step_idx >= max_steps:
                            break
                        continue

                    img_rgb = wrapper.render().copy()
                    
                    # === Visualization (updated) ===
                    if img_rgb.shape[0] > 64:
                        h, w = img_rgb.shape[:2]
                        sim_obj = getattr(wrapper, 'sim', None) 
                        if sim_obj:
                            pos_obs_3d = obstacle_pos_phys
                            pos_agent_3d = curr_eef_pos
                            pos_obs_ground_3d = pos_obs_3d.copy()
                            pos_obs_ground_3d[2] = 0.8 # table_z

                            # Projection
                            px_obs = world_to_pixel_mujoco(sim_obj, pos_obs_3d, "agentview", h, w)
                            px_obs_ground = world_to_pixel_mujoco(sim_obj, pos_obs_ground_3d, "agentview", h, w)
                            px_agent = world_to_pixel_mujoco(sim_obj, pos_agent_3d, "agentview", h, w)
                            
                            # Helper lines
                            if px_obs is not None and px_obs_ground is not None:
                                cv2.line(img_rgb, px_obs, px_obs_ground, (255, 255, 0), 1, cv2.LINE_AA)
                                cv2.circle(img_rgb, px_obs_ground, 3, (255, 255, 0), 1)
                            if px_agent is not None and px_obs is not None:
                                cv2.line(img_rgb, px_agent, px_obs, (200, 200, 200), 1, cv2.LINE_AA)

                            # Draw markers
                            if px_obs is not None:
                                cv2.circle(img_rgb, px_obs, 6, (255, 0, 255), -1) 
                                cv2.putText(img_rgb, "OBS", (px_obs[0]+8, px_obs[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                            
                            if px_agent is not None:
                                cv2.circle(img_rgb, px_agent, 6, (0, 0, 255), -1)

                                # --- Unified Arrow Viz (CCG/TEG) : direction + magnitude, comparable ---
                                if cached_scale > 1e-5:
                                    vec_3d = (cached_grad * cached_scale).detach().cpu().numpy().squeeze()[:3]  # (3,)
                                    v = np.array(vec_3d, dtype=np.float32)
                                    v_norm = float(np.linalg.norm(v))

                                    if v_norm > 1e-6:
                                        direction = v / (v_norm + 1e-8)

                                        # magnitude -> length mapping (meters), saturating to avoid explosion
                                        L_MIN = 0.02   # 2cm
                                        L_MAX = 0.20   # 20cm
                                        ALPHA = 2.5
                                        vis_len_m = L_MIN + (L_MAX - L_MIN) * (1.0 - np.exp(-ALPHA * v_norm))

                                        px_end = None
                                        for s in [1.0, 0.5, 0.25, 0.125]:
                                            pos_end = pos_agent_3d + direction * (vis_len_m * s)
                                            px_end = world_to_pixel_mujoco(sim_obj, pos_end, "agentview", h, w)
                                            if px_end is not None:
                                                cv2.arrowedLine(img_rgb, px_agent, px_end, (0, 255, 0), 2, tipLength=0.3)
                                                break

                                        # debug (enable one run to check small vector vs projection failure)
                                        # print(f"[arrow_dbg][teg] |v|={v_norm:.3e} vis_len_m={vis_len_m:.3f} px_agent={px_agent} px_end={px_end}")

                        
                        # Info Panel
                        # cv2.putText(img_rgb, f"Scale: {cached_scale:.4f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

                        # Red Border Warning
                        COLLISION_DIST = 0.02
                        WARNING_DIST = 0.05  
                        if dist_to_obs < WARNING_DIST:
                            severity = np.clip((WARNING_DIST - dist_to_obs) / (WARNING_DIST - COLLISION_DIST), 0.0, 1.0)
                            border_color = (0, 0, int(50 + 205 * severity))
                            thickness = int(2 + 13 * severity)
                            cv2.rectangle(img_rgb, (0, 0), (w-1, h-1), border_color, thickness)
                            if dist_to_obs < COLLISION_DIST:
                                text = "COLLISION!"
                                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                                cv2.putText(img_rgb, text, ((w - text_size[0]) // 2, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                    imgs.append(img_rgb)
                
                step_idx += 1
                pbar.update(1)
                
                if done or step_idx >= max_steps: break
            
            na_from_prev_chunk = na
            if done: break
            
        pbar.close()

    max_r = max(rewards) if rewards else 0
    return imgs, max_r, np.array(trajectory_log), has_collided



def run_experiment_100_trials(total_needed: int = 100, render_gifs: bool = False):
    # Task-adaptive results directory
    base_dir = STEG_RESULTS_DIR
    sub_dirs = {
        "collision": os.path.join(base_dir, "1_collision"),
        "failure":   os.path.join(base_dir, "2_failure_no_collision"),
        "success":   os.path.join(base_dir, "3_success_no_collision")
    }
    
    # Create output folders
    for p in sub_dirs.values():
        os.makedirs(p, exist_ok=True)
    
    stats = {
        "collision": 0,
        "failure": 0,
        "success": 0
    }
    
    valid_count = 0
    current_seed = 0

    print(f"[{_steg_task_name}] Starting {total_needed} trials: Base Pass -> Set Obstacle -> TEG Run (render_gifs={render_gifs})")

    pbar = tqdm(total=total_needed, desc="Valid Trials")
    
    while valid_count < total_needed:
        current_seed += 1
        
        # --- Phase 1: Base run (no obstacle, no guidance) ---
        _, max_r_base, traj_base, _ = run_ssip_teg_inference(
            obstacle_pos_phys=None, # no obstacle
            guidance_scale=0.0,     # no guidance
            seed=current_seed,
            collect_trajectory=True, # numeric only, no rendering
            render=False
        )
        
        # Skip seed if base policy fails
        if max_r_base < 1.0:
            continue
            
        # --- Phase 2: Place obstacle ---
        traj_len = len(traj_base)
        if traj_len < 10: continue 
        
        idx_start = int(traj_len * 0.02)
        idx_end = int(traj_len * 0.04)
        if idx_start >= idx_end: idx_end = idx_start + 1
        
        obs_idx = np.random.randint(idx_start, idx_end)

        base_pos = traj_base[obs_idx]
        
        # Random XY offset (-1cm to +1cm)
        # Jitter within a 2cm x 2cm box
        offset_xy = np.random.uniform(-0.02, 0.02, size=2)
        random_offset = np.array([offset_xy[0], offset_xy[1], 0.0]) # no Z offset
        
        obstacle_pos_selected = base_pos + random_offset
        # obstacle_pos_selected = traj_base[obs_idx]
        
        # --- Phase 3: TEG run (obstacle + TEG guidance) ---
        # No critic_model_path; TEG parameters only
        imgs, max_r_ccg, _, has_collided = run_ssip_teg_inference(
            obstacle_pos_phys=obstacle_pos_selected,
            guidance_scale=float(STEG_HP["guidance_scale"]),
            activation_dist=float(STEG_HP["activation_dist"]),
            N_ensemble=int(STEG_HP["N_ensemble"]),
            update_interval=int(STEG_HP["update_interval"]),
            K_horizon=int(STEG_HP["K_horizon"]),
            seed=current_seed,
            collect_trajectory=False,
            render=bool(render_gifs)
        )
        
        # --- Phase 4: Classify and save ---
        category = ""
        save_path = ""
        
        if has_collided:
            category = "collision"
            stats["collision"] += 1
            save_path = os.path.join(sub_dirs["collision"], f"seed_{current_seed}_collided.gif")
        elif max_r_ccg < 1.0:
            category = "failure"
            stats["failure"] += 1
            save_path = os.path.join(sub_dirs["failure"], f"seed_{current_seed}_failed.gif")
        else:
            category = "success"
            stats["success"] += 1
            save_path = os.path.join(sub_dirs["success"], f"seed_{current_seed}_success.gif")
            
        # Save GIF (only when render_gifs=True)
        if render_gifs and imgs:
            imageio.mimsave(save_path, imgs, fps=15)
        
        valid_count += 1
        pbar.update(1)
        pbar.set_postfix(
            Coll=f"{stats['collision']}", 
            Fail=f"{stats['failure']}", 
            Succ=f"{stats['success']}"
        )

    pbar.close()
    
    # --- Final statistics ---
    print("\n" + "="*40)
    print(f"Experiment finished! task={_steg_task_name}, summary:")
    print(f"Total valid trials: {total_needed}")
    def _pct(x: int) -> float:
        return (100.0 * float(x) / float(total_needed)) if total_needed > 0 else 0.0
    print(f"1. Collision:                         {stats['collision']} ({_pct(stats['collision']):.1f}%)")
    print(f"2. Avoided obstacle but task failed:  {stats['failure']} ({_pct(stats['failure']):.1f}%)")
    print(f"3. Success (avoided + task done):     {stats['success']} ({_pct(stats['success']):.1f}%)")
    print("="*40)
    print(f"Results saved under {base_dir}")


if __name__ == "__main__":
    _N = int(os.environ.get("TOTAL_NEEDED", "100"))
    _RENDER = bool(int(os.environ.get("RENDER_GIFS", "0")))
    run_experiment_100_trials(total_needed=_N, render_gifs=_RENDER)
