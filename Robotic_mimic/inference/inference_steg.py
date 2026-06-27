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

from core.config import *  # noqa: F401,F403  (device / obs_horizon / action_horizon / pred_horizon / RESULTS_BASE_DIR 等)

# 惰性单例：import 时零副作用，首次访问才建环境
from core.pusht_env import get_env
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
# 前置依赖
# =========================================================
def get_drift(x, t, global_cond, sigma_infer, eps=1e-6):
    """
    计算修正后的漂移项 b(x, t) = v + score_correction
    x: (B, 1, 10)
    """
    if t.ndim == 0: t = t.unsqueeze(0)
    
    # 预测 v 和 eta
    v_pred = ema_si_velocity_net(sample=x, timestep=t, global_cond=global_cond)
    eta_pred = ema_si_denoiser_net(sample=x, timestep=t, global_cond=global_cond)
    
    # 统一维度处理
    if v_pred.ndim == 2: v_pred = v_pred.unsqueeze(1)
    if eta_pred.ndim == 2: eta_pred = eta_pred.unsqueeze(1)

    # 获取 gamma 及其导数
    gamma = gamma_t_si(t).view(-1, 1, 1).to(device)
    gamma_dot = d_gamma_dt_si(t).view(-1, 1, 1).to(device)
    
    # 计算 score 和 漂移系数
    s_pred = -eta_pred / (gamma + eps)
    score_coeff = 0.5 * (sigma_infer ** 2) - (gamma * gamma_dot)
    
    b = v_pred + score_coeff * s_pred
    return b.squeeze(1)

def world_to_pixel_mujoco(sim, pos_3d, camera_name, img_h, img_w):
    """ 利用 MuJoCo 内参矩阵进行精确投影 (新版) """
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
# 3. 修改后的推理函数 (使用 SSIP-TEG 策略)
# =========================================================

def run_ssip_teg_inference(
    obstacle_pos_phys, 
    # --- TEG 参数 ---
    guidance_scale=15.0,     
    activation_dist=0.6,
    N_ensemble=16,        
    update_interval=4,
    K_horizon=3,           # 向前模拟步数
    # --- 通用参数 ---
    seed=0,                  
    collect_trajectory=False, # 是否收集轨迹 (用于 Base Run)
    render: bool = True        # 是否渲染并返回帧（批量实验建议关闭以提速）
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    wrapper.seed(seed)
    # =========================================================
    # 动态计算 'robot0_eef_pos' 在扁平 obs 向量中的起始位置（避免硬编码 start_dim）
    key_dims = []
    for key in wrapper.obs_keys:
        dummy_obs = wrapper.env.get_observation()[key]
        dim = np.array(dummy_obs).flatten().shape[0]
        key_dims.append(dim)
    target_key = 'robot0_eef_pos'
    if target_key not in wrapper.obs_keys:
        raise ValueError(f"Obs keys 里面没有 {target_key}，当前 obs_keys={wrapper.obs_keys}")
    _eef_pos_start = int(sum(key_dims[: wrapper.obs_keys.index(target_key)]))

    def _raw_to_norm(raw: np.ndarray, min_v: np.ndarray, max_v: np.ndarray) -> np.ndarray:
        """把 raw action 映射到 action_normalizer 的归一化空间 [-1, 1]（与 MinMaxNormalizer 一致）"""
        raw = raw.astype(np.float32)
        min_v = min_v.astype(np.float32)
        max_v = max_v.astype(np.float32)
        denom = (max_v - min_v + 1e-6)
        norm01 = (raw - min_v) / denom
        norm01 = np.clip(norm01, 0.0, 1.0)
        return norm01 * 2.0 - 1.0

    is_delta = bool(CONTROL_DELTA) if 'CONTROL_DELTA' in globals() and CONTROL_DELTA is not None else True

    # --- 1. 准备障碍物数据 ---
    if obstacle_pos_phys is None:
        obstacle_pos_phys = np.array([10.0, 10.0, 10.0]) # Dummy position

    # 确保障碍物坐标是 float32 Tensor
    obs_phys_tensor = torch.tensor(obstacle_pos_phys, device=device, dtype=torch.float32)

    # --- 2. 环境重置 ---
    obs = wrapper.reset()
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    imgs = []
    if (not collect_trajectory) and render:
        imgs = [wrapper.render()]
    rewards = []
    
    trajectory_log = [] 
    has_collided = False
    
    # 物理碰撞判定阈值
    COLLISION_THRESHOLD = 0.02

    # 初始 Action
    curr_eef_pos = obs[_eef_pos_start : _eef_pos_start + 3].astype(np.float32)
    pos_min = dataset.action_normalizer.min_val[:3].astype(np.float32)
    pos_max = dataset.action_normalizer.max_val[:3].astype(np.float32)
    # delta 控制：初始位置通道应为 0 增量；absolute 控制：用当前 eef_pos 更合理
    raw_pos = np.zeros(3, dtype=np.float32) if is_delta else curr_eef_pos
    norm_pos = _raw_to_norm(raw_pos, pos_min, pos_max)

    na = torch.zeros((1, 1, 10), device=device, dtype=torch.float32)
    na[0, 0, :3] = torch.from_numpy(norm_pos).to(device, dtype=torch.float32)
    # 夹爪：归一化空间中 -1 往往对应 raw 最小值（通常是 open）
    na[0, 0, 9] = -1.0
    # 6D Rotation：禁止退化（全 0 会导致 RotationTransformer/数值不稳定）
    rot6d_raw = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)  # 单位旋转的稳定 6D
    rot_min = dataset.action_normalizer.min_val[3:9].astype(np.float32)
    rot_max = dataset.action_normalizer.max_val[3:9].astype(np.float32)
    rot6d_raw = np.clip(rot6d_raw, rot_min, rot_max)
    rot6d_norm = _raw_to_norm(rot6d_raw, rot_min, rot_max)
    na[0, 0, 3:9] = torch.from_numpy(rot6d_norm).to(device, dtype=torch.float32)
    # 数值安全
    na = torch.nan_to_num(na, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
    na_from_prev_chunk = na
    
    # 状态变量 (TEG 缓存)
    cached_grad = torch.zeros_like(na).squeeze(1)
    cached_scale = 0.0          
    
    done = False
    step_idx = 0
    max_steps = 400
    dt_val = 1.0 / (pred_horizon - obs_horizon)
    dt_est = 0.10 # TEG 模拟步长
    sigma_infer = 0.05 
    teg_cost_scale = 50.0 # 内部 Cost 缩放系数

    # 准备归一化参数 (用于可微反归一化)
    min_v_tensor = torch.from_numpy(dataset.action_normalizer.min_val).to(device, dtype=torch.float32)
    max_v_tensor = torch.from_numpy(dataset.action_normalizer.max_val).to(device, dtype=torch.float32)
    scale_v_tensor = max_v_tensor - min_v_tensor
    scale_v_tensor[scale_v_tensor==0] = 1.0

    with torch.no_grad():
        # 进度条
        pbar = tqdm(total=max_steps, desc=f"Seed {seed}", disable=collect_trajectory)
        
        while not done and step_idx < max_steps:
            
            # --- 准备 Observation ---
            obs_seq = np.stack(obs_deque)
            nobs = dataset.obs_normalizer.normalize(obs_seq)
            cond_flat = torch.from_numpy(nobs).to(device, dtype=torch.float32).flatten(start_dim=0).unsqueeze(0)
            o_test = cond_flat 
            
            na = na_from_prev_chunk
            # [数据格式一致性] 模型训练时 latent action 在 [-1,1]，推理积分可能漂出该范围或出现 NaN/Inf
            na = torch.nan_to_num(na, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
            
            # --- 执行 Action Chunk ---
            for i in range(action_horizon):
                t_scalar = np.clip(i * dt_val, 1e-3, 1.0 - 1e-3)
                
                # --- A. 物理距离与碰撞检测 ---
                # 用 observation 中的 robot0_eef_pos 作为真实位置（兼容 delta/absolute）
                curr_eef_pos = obs[_eef_pos_start : _eef_pos_start + 3].astype(np.float32)
                eef_pos_now_t = torch.from_numpy(curr_eef_pos).to(device=device, dtype=torch.float32)
                dist_to_obs = torch.norm(eef_pos_now_t - obs_phys_tensor).item()

                if collect_trajectory:
                    trajectory_log.append(curr_eef_pos.copy())

                if dist_to_obs < COLLISION_THRESHOLD:
                    has_collided = True

                # --- B. TEG 引导计算 (Steering) ---
                effective_scale = 0.0 if collect_trajectory else guidance_scale

                # 判断是否需要重新计算梯度 (距离近 + 间隔步数到了 + 有引导需求)
                should_compute = (dist_to_obs < activation_dist) and (step_idx % update_interval == 0) and (effective_scale > 0)

                if should_compute:
                    with torch.enable_grad():
                        # 1. 准备梯度输入
                        na_in = na.squeeze(1).detach().requires_grad_(True)
                        
                        # 2. 扩展 Batch 进行并行模拟
                        na_ens = na_in.repeat(N_ensemble, 1) 
                        cond_ens = o_test.repeat(N_ensemble, 1)
                        # 确保时间是 float32
                        t_ens = torch.tensor([t_scalar], device=device, dtype=torch.float32).repeat(N_ensemble)
                        
                        curr_na_sim = na_ens
                        curr_t_sim = t_ens
                        cum_cost = torch.zeros(N_ensemble, device=device, dtype=torch.float32)
                        # delta 控制需要对“增量”累加成位置来算避障 cost；absolute 则直接用动作位置
                        sim_pos = eef_pos_now_t.view(1, 3).repeat(N_ensemble, 1)
                        
                        # 3. 向前模拟循环 (Rollout)
                        for k_step in range(K_horizon):
                            # [数据格式一致性] rollout 中同样保持 latent 在 [-1,1] 且数值安全
                            curr_na_sim = torch.nan_to_num(curr_na_sim, nan=0.0, posinf=0.0, neginf=0.0)
                            curr_na_sim = torch.clamp(curr_na_sim, -1.0, 1.0)
                            t_input = torch.clamp(curr_t_sim, 0.05, 0.95)
                            
                            # 网络预测
                            v_p = ema_si_velocity_net(curr_na_sim.unsqueeze(1), t_input, cond_ens).squeeze(1)
                            eta_p = ema_si_denoiser_net(curr_na_sim.unsqueeze(1), t_input, cond_ens).squeeze(1)
                            
                            # 计算系数
                            gamma = gamma_t_si(t_input).view(-1, 1)
                            g_dot = d_gamma_dt_si(t_input).view(-1, 1)
                            s_p = -eta_p / (gamma + 1e-6)
                            sigma_est = 0.05 
                            score_coeff = 0.5 * (sigma_est**2) - (gamma * g_dot)
                            b_drift = v_p + score_coeff * s_p
                            
                            # --- 物理空间 Cost 计算 ---
                            # 可微反归一化 (Differentiable Denormalization)
                            denorm_sim = ((curr_na_sim + 1) / 2) * scale_v_tensor + min_v_tensor
                            if is_delta:
                                delta_xyz = denorm_sim[:, :3]
                                sim_pos = sim_pos + delta_xyz
                                sim_xyz = sim_pos
                            else:
                                sim_xyz = denorm_sim[:, :3] # 只取 XYZ 位置
                            
                            # 计算与障碍物的距离 Cost (高斯排斥)
                            dist_sq = ((sim_xyz - obs_phys_tensor)**2).sum(dim=-1)
                            cost_step = teg_cost_scale * torch.exp(- dist_sq / (2 * 0.05**2)) # 0.05 是障碍物半径参数
                            cum_cost = cum_cost + cost_step * dt_est
                            
                            # --- SDE 更新 (含噪声) ---
                            noise_step = torch.randn_like(curr_na_sim)
                            diffusion_term = sigma_est * math.sqrt(dt_est) * noise_step
                            curr_na_sim = curr_na_sim + b_drift * dt_est + diffusion_term
                            
                            curr_t_sim = curr_t_sim + dt_est
                            
                        # 4. 反向传播求梯度
                        neg_cost = -cum_cost
                        log_utility = torch.logsumexp(neg_cost, dim=0)
                        
                        (g,) = torch.autograd.grad(log_utility, na_in)
                        
                        # 更新缓存
                        cached_grad = torch.nan_to_num(g.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                        
                    # 更新引导强度 (距离越近，强度越大)
                    cached_scale = sigma_infer**2 * effective_scale * max(0, (1.0 - dist_to_obs / activation_dist))
                
                elif dist_to_obs >= activation_dist:
                    # 距离远时，清空
                    cached_scale = 0.0
                    cached_grad = torch.zeros_like(na).squeeze(1)

                # --- C. Drift 计算与更新 ---
                t_tensor = torch.tensor([t_scalar], device=device, dtype=torch.float32)
                # [数据格式一致性] 漂移网络输入保持在训练的归一化范围
                na = torch.nan_to_num(na, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
                base_drift = get_drift(na, t_tensor, o_test, sigma_infer)
                
                # 叠加 SSIP-TEG 梯度
                total_drift = base_drift + cached_scale * cached_grad#.unsqueeze(0)
                
                noise = torch.randn_like(total_drift)
                na = na + total_drift.unsqueeze(1) * dt_val + sigma_infer * math.sqrt(dt_val) * noise.unsqueeze(1)
                # [数据格式一致性] 更新后再做一次数值安全与范围裁剪
                na = torch.nan_to_num(na, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
                
                # --- D. Step & Render ---
                a_real = dataset.get_unnormalized_action(na).squeeze()
                obs, reward, done, _ = wrapper.step(a_real)
                obs_deque.append(obs)
                rewards.append(reward)
                
                if not collect_trajectory:
                    if not render:
                        # 批量实验时关闭渲染以显著提速
                        step_idx += 1
                        pbar.update(1)
                        if done or step_idx >= max_steps:
                            break
                        continue

                    img_rgb = wrapper.render().copy()
                    
                    # === 可视化部分 (新版) ===
                    if img_rgb.shape[0] > 64:
                        h, w = img_rgb.shape[:2]
                        sim_obj = getattr(wrapper, 'sim', None) 
                        if sim_obj:
                            pos_obs_3d = obstacle_pos_phys
                            pos_agent_3d = curr_eef_pos
                            pos_obs_ground_3d = pos_obs_3d.copy()
                            pos_obs_ground_3d[2] = 0.8 # table_z

                            # 投影
                            px_obs = world_to_pixel_mujoco(sim_obj, pos_obs_3d, "agentview", h, w)
                            px_obs_ground = world_to_pixel_mujoco(sim_obj, pos_obs_ground_3d, "agentview", h, w)
                            px_agent = world_to_pixel_mujoco(sim_obj, pos_agent_3d, "agentview", h, w)
                            
                            # 辅助线
                            if px_obs is not None and px_obs_ground is not None:
                                cv2.line(img_rgb, px_obs, px_obs_ground, (255, 255, 0), 1, cv2.LINE_AA)
                                cv2.circle(img_rgb, px_obs_ground, 3, (255, 255, 0), 1)
                            if px_agent is not None and px_obs is not None:
                                cv2.line(img_rgb, px_agent, px_obs, (200, 200, 200), 1, cv2.LINE_AA)

                            # 绘制点
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

                                        # debug（建议先打开一轮，确认到底是“向量小”还是“投影失败”）
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
    # 任务自适应的结果目录
    base_dir = STEG_RESULTS_DIR
    sub_dirs = {
        "collision": os.path.join(base_dir, "1_collision"),
        "failure":   os.path.join(base_dir, "2_failure_no_collision"),
        "success":   os.path.join(base_dir, "3_success_no_collision")
    }
    
    # 清理或创建文件夹
    for p in sub_dirs.values():
        os.makedirs(p, exist_ok=True)
    
    stats = {
        "collision": 0,
        "failure": 0,
        "success": 0
    }
    
    valid_count = 0
    current_seed = 0

    print(f"[{_steg_task_name}] 开始 {total_needed} 次实验: Base Pass -> Set Obstacle -> TEG Run (render_gifs={render_gifs})")

    pbar = tqdm(total=total_needed, desc="Valid Trials")
    
    while valid_count < total_needed:
        current_seed += 1
        
        # --- Phase 1: Base Run (无障碍物，无引导) ---
        _, max_r_base, traj_base, _ = run_ssip_teg_inference(
            obstacle_pos_phys=None, # 无障碍
            guidance_scale=0.0,     # 无引导
            seed=current_seed,
            collect_trajectory=True, # 只跑数值，不渲染
            render=False
        )
        
        # 如果 Base Policy 本身就失败，跳过这个种子
        if max_r_base < 1.0:
            continue
            
        # --- Phase 2: 设置障碍物 ---
        traj_len = len(traj_base)
        if traj_len < 10: continue 
        
        idx_start = int(traj_len * 0.02)
        idx_end = int(traj_len * 0.04)
        if idx_start >= idx_end: idx_end = idx_start + 1
        
        obs_idx = np.random.randint(idx_start, idx_end)

        base_pos = traj_base[obs_idx]
        
        # [新增] 生成 XY 平面的随机偏移 (-1cm 到 +1cm)
        # 相当于在一个 2cm x 2cm 的方框内随机抖动
        offset_xy = np.random.uniform(-0.02, 0.02, size=2)
        random_offset = np.array([offset_xy[0], offset_xy[1], 0.0]) # Z轴不偏移
        
        obstacle_pos_selected = base_pos + random_offset
        # obstacle_pos_selected = traj_base[obs_idx]
        
        # --- Phase 3: TEG Run (有障碍物，TEG 引导) ---
        # 移除 critic_model_path，添加 TEG 参数
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
        
        # --- Phase 4: 分类与保存 ---
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
            
        # 保存 GIF（仅在 render_gifs=True 时有效）
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
    
    # --- 输出最终统计 ---
    print("\n" + "="*40)
    print(f"实验结束！任务={_steg_task_name}，统计结果如下：")
    print(f"总有效实验次数: {total_needed}")
    def _pct(x: int) -> float:
        return (100.0 * float(x) / float(total_needed)) if total_needed > 0 else 0.0
    print(f"1. 碰撞 (Collision):              {stats['collision']} ({_pct(stats['collision']):.1f}%)")
    print(f"2. 避障但在任务中失败 (Failure):    {stats['failure']} ({_pct(stats['failure']):.1f}%)")
    print(f"3. 成功避障并完成任务 (Success):    {stats['success']} ({_pct(stats['success']):.1f}%)")
    print("="*40)
    print(f"结果已保存在 {base_dir} 目录下")


if __name__ == "__main__":
    _N = int(os.environ.get("TOTAL_NEEDED", "100"))
    _RENDER = bool(int(os.environ.get("RENDER_GIFS", "0")))
    run_experiment_100_trials(total_needed=_N, render_gifs=_RENDER)
