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

from core.config import *  # noqa: F401,F403  (device / obs_horizon / action_horizon / pred_horizon / GUIDANCE_SCALE / RESULTS_BASE_DIR, etc.)
from core.robomimic_env import get_env
_env = get_env()
wrapper        = _env.wrapper
dataset_path   = _env.dataset_path
CONTROL_DELTA  = _env.control_delta
from core.dataset import dataset
from core.flow_policy import gamma_t_si, d_gamma_dt_si
from core.critic import CollisionPredictionCritic
from core.my_utils import (
    _infer_task_name_from_dataset_path,
    _task_cfg,
    _get_eef_pos_start,
    _get_eef_pos_from_obs,
    _r_obs_norm_from_phys,
    _init_na_from_obs,
)
# Reuse pretrained SI models loaded by inference_si (import loads ckpt only, no SI inference)
from inference.inference_si import ema_si_velocity_net, ema_si_denoiser_net

# =========================================================
# Assumes upstream deps defined (get_drift, world_to_pixel_mujoco, etc.)
# =========================================================
def get_drift(x, t, global_cond, sigma_infer, eps=1e-6):
    """
    Compute corrected drift b(x, t) = v + score_correction
    x: (B, 1, 10)
    """
    if t.ndim == 0: t = t.unsqueeze(0)
    
    v_pred = ema_si_velocity_net(sample=x, timestep=t, global_cond=global_cond)
    eta_pred = ema_si_denoiser_net(sample=x, timestep=t, global_cond=global_cond)
    
    # Unified shape handling: network output may be (B, 10) or (B, 1, 10)
    if v_pred.ndim == 2: v_pred = v_pred.unsqueeze(1)
    if eta_pred.ndim == 2: eta_pred = eta_pred.unsqueeze(1)

    # gamma and its derivative
    gamma = gamma_t_si(t).view(-1, 1, 1).to(device)
    gamma_dot = d_gamma_dt_si(t).view(-1, 1, 1).to(device)
    
    # Score and drift coefficient
    s_pred = -eta_pred / (gamma + eps)
    score_coeff = 0.5 * (sigma_infer ** 2) - (gamma * gamma_dot)
    
    b = v_pred + score_coeff * s_pred
    return b.squeeze(1) # This is really important otherwise there will be bug!!! This is the only differnece to the original get_drift() function.

def world_to_pixel_mujoco(sim, pos_3d, camera_name, img_h, img_w):
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
        u, v = int(cx + f * x / z), int(cy + f * y / z)
        if 0 <= u < img_w and 0 <= v < img_h: return (u, v)
        return None
    except Exception: return None


def run_ccg_phys_vis_inference(
    obstacle_pos_phys, 
    critic_model_path,       
    guidance_scale=20.0,     
    power_k=3.0,             
    activation_dist=0.6,
    seed=0,                  # optional random seed
    collect_trajectory=False, # collect trajectory (for base run)
    render=True,              # render frames (visualization / GIF)
):
    # =========================================================
    # [Key change 1] Fix all random seeds (RNG seeding)
    # =========================================================
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    wrapper.seed(seed)
    # =========================================================

    global _ccg_cfg, _ccg_action_dim, _ccg_obs_dim, _ccg_eef_pos_start
    if "_ccg_cfg" not in globals():
        _ccg_task_name = _infer_task_name_from_dataset_path(dataset_path)
        _ccg_cfg = _task_cfg(_ccg_task_name)
    if "_ccg_action_dim" not in globals():
        _ccg_action_dim = int(dataset[0]["action"].shape[-1])
    if "_ccg_obs_dim" not in globals():
        _ccg_obs_dim = int(dataset[0]["obs"].shape[-1])
    if "_ccg_eef_pos_start" not in globals():
        _ccg_eef_pos_start = _get_eef_pos_start(wrapper, "robot0_eef_pos")

    # --- 1. Initialize models ---
    critic_net = CollisionPredictionCritic(
        action_dim=_ccg_action_dim,
        obs_dim=_ccg_obs_dim,
        obs_horizon=obs_horizon,
        eef_pos_start=_ccg_eef_pos_start,
        use_obs_eef_pos=True,
    ).to(device)
    # checkpoint = torch.load(critic_model_path, map_location=device, weights_only=False)
    # critic_net = HighCapacityCritic(action_dim=10, obs_dim=19, obs_horizon=2, hidden_dim=1024, depth=6).to(device)
    checkpoint = torch.load(critic_model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    critic_net.load_state_dict(state_dict)
    critic_net.eval()
    
    # --- 2. Prepare obstacle data ---
    if obstacle_pos_phys is None:
        obstacle_pos_phys = np.array([10.0, 10.0, 10.0]) # Dummy position

    # A. Normalized coordinates
    # Note: assumes absolute control (action pos min/max match physical workspace)
    if bool(CONTROL_DELTA) if CONTROL_DELTA is not None else False:
        print("[warn] CONTROL_DELTA=True: CCG obstacle normalization still uses action_normalizer pos range, which may not match physical space. Prefer *_abs datasets.")
    pos_min = dataset.action_normalizer.min_val[:3]
    pos_max = dataset.action_normalizer.max_val[:3]
    norm_obs_pos = (obstacle_pos_phys - pos_min) / (pos_max - pos_min) * 2 - 1
    obs_pos_tensor = torch.tensor(norm_obs_pos, device=device, dtype=torch.float32).unsqueeze(0)

    # B. Physical coordinates
    obs_phys_tensor = torch.tensor(obstacle_pos_phys, device=device, dtype=torch.float32)

    # --- 3. Environment reset ---
    # (wrapper.seed set above; reset directly)
    obs = wrapper.reset()
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    # Render toggle: skip render when off; imgs stays empty
    imgs = [wrapper.render()] if (render and (not collect_trajectory)) else []
    rewards = []
    
    trajectory_log = [] 
    has_collided = False
    
    # [Key change 2] Physical collision threshold
    COLLISION_THRESHOLD = float(_ccg_cfg["collision_threshold_phys"])

    # Initial latent action (cross-task: auto-init from obs; no hard-coded start_dim)
    na = _init_na_from_obs(obs, dataset, _ccg_eef_pos_start, CONTROL_DELTA, device)
    na_from_prev_chunk = na
    
    # State variables
    cached_grad = torch.zeros_like(na).squeeze(1)
    cached_prob = 0.0          
    current_dynamic_scale = 0.0 
    
    done = False
    step_idx = 0
    max_steps = 400
    dt_val = 1.0 / (pred_horizon - obs_horizon)
    sigma_infer = 0.05 

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
            
            # --- Execute action chunk ---
            for i in range(action_horizon):
                t_scalar = np.clip(i * dt_val, 1e-3, 1.0 - 1e-3)
                t_tensor = torch.tensor([t_scalar], device=device, dtype=torch.float32)

                # --- A. Physical distance ---
                # Use true eef_pos from obs (avoids delta/absolute confusion)
                curr_xyz_np = _get_eef_pos_from_obs(obs, _ccg_eef_pos_start)
                curr_xyz = torch.tensor(curr_xyz_np, device=device, dtype=torch.float32)
                dist_to_obs = torch.norm(curr_xyz - obs_phys_tensor).item()

                if collect_trajectory:
                    trajectory_log.append(curr_xyz_np.copy())

                # [Unified threshold]
                if dist_to_obs < COLLISION_THRESHOLD:
                    has_collided = True

                # --- B. CCG guidance ---
                effective_scale = 0.0 if collect_trajectory else guidance_scale

                if dist_to_obs < activation_dist and effective_scale > 0:
                    with torch.enable_grad():
                        na_in = na.squeeze(1).detach().requires_grad_(True)
                        risk_prob = critic_net(na_in, t_tensor, obs_pos_tensor, cond_flat)
                        (g_risk,) = torch.autograd.grad(risk_prob, na_in)
                        
                        prob_val = risk_prob.item()
                        dynamic_scale = effective_scale * (prob_val ** power_k)
                        
                        cached_grad = g_risk.detach()
                        cached_prob = prob_val
                        current_dynamic_scale = dynamic_scale
                else:
                    cached_grad = torch.zeros_like(na).squeeze(1)
                    cached_prob = 0.0
                    current_dynamic_scale = 0.0

                # --- C. Drift computation ---
                base_drift = get_drift(na, t_tensor, o_test, sigma_infer)
                total_drift = base_drift - current_dynamic_scale * cached_grad
                
                noise = torch.randn_like(total_drift)
                na = na + total_drift.unsqueeze(1) * dt_val + sigma_infer * math.sqrt(dt_val) * noise.unsqueeze(1)
                
                # --- D. Step & Render ---
                a_real = dataset.get_unnormalized_action(na).squeeze()
                obs, reward, done, _ = wrapper.step(a_real)
                obs_deque.append(obs)
                rewards.append(reward)
                
                if (not collect_trajectory) and render:
                    img_rgb = wrapper.render().copy()
                    
                    # === Visualization ===
                    if img_rgb.shape[0] > 64:
                        h, w = img_rgb.shape[:2]
                        sim_obj = getattr(wrapper, 'sim', None) 
                        if sim_obj:
                            pos_obs_3d = obstacle_pos_phys
                            pos_agent_3d = curr_xyz_np.copy()
                            pos_obs_ground_3d = pos_obs_3d.copy()
                            pos_obs_ground_3d[2] = float(_ccg_cfg["table_z"]) # table_z

                            px_obs = world_to_pixel_mujoco(sim_obj, pos_obs_3d, "agentview", h, w)
                            px_obs_ground = world_to_pixel_mujoco(sim_obj, pos_obs_ground_3d, "agentview", h, w)
                            px_agent = world_to_pixel_mujoco(sim_obj, pos_agent_3d, "agentview", h, w)
                            
                            if px_obs is not None and px_obs_ground is not None:
                                cv2.line(img_rgb, px_obs, px_obs_ground, (255, 255, 0), 1, cv2.LINE_AA)
                                cv2.circle(img_rgb, px_obs_ground, 3, (255, 255, 0), 1)
                            if px_agent is not None and px_obs is not None:
                                cv2.line(img_rgb, px_agent, px_obs, (200, 200, 200), 1, cv2.LINE_AA)

                            if px_obs is not None:
                                cv2.circle(img_rgb, px_obs, 6, (255, 0, 255), -1) 
                                cv2.putText(img_rgb, "OBS", (px_obs[0]+8, px_obs[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                            
                            if px_agent is not None:
                                cv2.circle(img_rgb, px_agent, 6, (0, 255, 0), -1) 
                                if current_dynamic_scale > 1e-2:
                                    v = (-cached_grad * current_dynamic_scale).detach().cpu().numpy().squeeze()[:3].astype(np.float32)
                                    v_norm = float(np.linalg.norm(v))
                                    if v_norm > 1e-6:
                                        direction = v / (v_norm + 1e-8)

                                        L_MIN = 0.02
                                        L_MAX = 0.20
                                        ALPHA = 2.5
                                        vis_len_m = L_MIN + (L_MAX - L_MIN) * (1.0 - np.exp(-ALPHA * v_norm))

                                        px_end = None
                                        for s in [1.0, 0.5, 0.25, 0.125]:
                                            pos_end = pos_agent_3d + direction * (vis_len_m * s)
                                            px_end = world_to_pixel_mujoco(sim_obj, pos_end, "agentview", h, w)
                                            if px_end is not None:
                                                cv2.arrowedLine(img_rgb, px_agent, px_end, (0, 0, 255), 2, tipLength=0.3)
                                                break


                        
                        # Info Panel
                        # overlay = img_rgb.copy()
                        # cv2.rectangle(overlay, (5, 5), (170, 55), (0, 0, 0), -1)
                        # alpha = 0.6
                        # cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0, img_rgb)
                        # risk_color = (0, 255, 0) if cached_prob < 0.5 else (0, 0, 255)
                        #cv2.putText(img_rgb, f"Risk: {cached_prob:.3f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 1, cv2.LINE_AA)
                        #cv2.putText(img_rgb, f"Scale: {current_dynamic_scale:.1f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

                        # Red Border Warning
                        # [Key change 3] visualization threshold matches collision threshold
                        COLLISION_DIST = float(_ccg_cfg["collision_threshold_phys"])
                        WARNING_DIST = float(_ccg_cfg["warning_dist_phys"])
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


# =========================================================
# 4. Main experiment loop
# =========================================================

def run_experiment_100_trials(render_and_save=True):
    # Ensure cross-task config is ready (no dependency on training cell above)
    global _ccg_cfg
    if "_ccg_cfg" not in globals():
        _ccg_task_name = _infer_task_name_from_dataset_path(dataset_path)
        _ccg_cfg = _task_cfg(_ccg_task_name)

    # 1. Create folder structure (only when render_and_save=True)
    base_dir = CCG_RESULTS_DIR
    sub_dirs = None
    if render_and_save:
        sub_dirs = {
            "collision": os.path.join(base_dir, "1_collision"),
            "failure":   os.path.join(base_dir, "2_failure_no_collision"),
            "success":   os.path.join(base_dir, "3_success_no_collision")
        }
        for p in sub_dirs.values():
            os.makedirs(p, exist_ok=True)
    
    stats = {
        "collision": 0,
        "failure": 0,
        "success": 0
    }
    
    total_needed = int(os.environ.get("TOTAL_NEEDED", "40"))
    valid_count = 0
    current_seed = 0
    
    print(f"Starting 100 trials: Base Pass -> Set Obstacle -> CCG Run")
    
    pbar = tqdm(total=total_needed, desc="Valid Trials")
    
    while valid_count < total_needed:
        current_seed += 1
        
        # --- Phase 1: Base run (no obstacle, no guidance) ---
        _, max_r_base, traj_base, _ = run_ccg_phys_vis_inference(
            obstacle_pos_phys=None, # no obstacle
            critic_model_path=str(_ccg_cfg["critic_checkpoint_path"]),
            guidance_scale=0.0,     # no guidance
            seed=current_seed,
            collect_trajectory=True # numeric only, no rendering
        )
        
        # Skip seed if base policy already fails
        if max_r_base < 1.0:
            continue
            
        # --- Phase 2: Place obstacle ---
        traj_len = len(traj_base)
        if traj_len < 10: continue # trajectory too short
        
        idx_start = int(traj_len * 0.02)
        idx_end = int(traj_len * 0.04)
        
        if idx_start >= idx_end: idx_end = idx_start + 1
        
        obs_idx = np.random.randint(idx_start, idx_end)
        base_pos = traj_base[obs_idx]

        offset_xy = np.random.uniform(
            -float(_ccg_cfg["obstacle_offset_xy"]),
            float(_ccg_cfg["obstacle_offset_xy"]),
            size=2,
        )
        random_offset = np.array(
            [offset_xy[0], offset_xy[1], float(_ccg_cfg["obstacle_offset_z"])],
            dtype=np.float32,
        )
        
        obstacle_pos_selected = base_pos + random_offset
        #obstacle_pos_selected = traj_base[obs_idx]
        
        # --- Phase 3: CCG run (obstacle + guidance) ---
        imgs, max_r_ccg, _, has_collided = run_ccg_phys_vis_inference(
            obstacle_pos_phys=obstacle_pos_selected,
            critic_model_path=str(_ccg_cfg["critic_checkpoint_path"]),
            guidance_scale=float(CCG_HP["guidance_scale"]),
            power_k=float(CCG_HP["power_k"]),
            activation_dist=float(CCG_HP["activation_dist"]),
            seed=current_seed,
            collect_trajectory=False, # trajectory collection off
            render=render_and_save,   # render for GIF
        )
        
        # --- Phase 4: Classify and save ---
        category = ""
        save_path = ""
        
        if has_collided:
            category = "collision"
            stats["collision"] += 1
            if render_and_save:
                save_path = os.path.join(sub_dirs["collision"], f"seed_{current_seed}_collided.gif")
        elif max_r_ccg < 1.0:
            category = "failure"
            stats["failure"] += 1
            if render_and_save:
                save_path = os.path.join(sub_dirs["failure"], f"seed_{current_seed}_failed.gif")
        else:
            category = "success"
            stats["success"] += 1
            if render_and_save:
                save_path = os.path.join(sub_dirs["success"], f"seed_{current_seed}_success.gif")
            
        # Save GIF (optional)
        if render_and_save:
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
    print("Experiment finished! Summary:")
    print(f"Total valid trials: {total_needed}")
    print(f"1. Collision:                         {stats['collision']} ({stats['collision']}%)")
    print(f"2. Avoided obstacle but task failed:  {stats['failure']} ({stats['failure']}%)")
    print(f"3. Success (avoided + task done):     {stats['success']} ({stats['success']}%)")
    print("="*40)
    if render_and_save:
        print(f"Results saved under {base_dir}")
    else:
        print("Rendering/saving disabled (render_and_save=False); no GIF files generated.")

# Run experiment
# Toggle: render and save GIFs
RENDER_AND_SAVE = True #🚪
# --- Module entry: __main__ guard for `python -m inference.inference_ccg` ---
if __name__ == "__main__":
    run_experiment_100_trials(
        render_and_save=RENDER_AND_SAVE,
    )
