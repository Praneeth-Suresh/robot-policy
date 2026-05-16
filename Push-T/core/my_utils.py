import os

def print_conditional_fail_report(df):
    import numpy as np
    print("\n" + "="*70)
    print("   CONDITIONAL REPORT (Collision + High Reward + Safe Distance)   ")
    print("="*70)
    
    if df.empty:
        print("DataFrame is empty.")
        return

    TRAIN_NORM_R = 0.06 
    # MIN_SAFE_DIST = TRAIN_NORM_R * 2.0  # 0.12
    MIN_SAFE_DIST = 0.2
    
    def get_start_dist(row):
        start_pt = np.array([row.get('agent_start_x', 0.0), row.get('agent_start_y', 0.0)])
        obs_pt = np.array(row['obs_pos_tuple'])
        return np.linalg.norm(start_pt - obs_pt)

    df['spawn_dist'] = df.apply(get_start_dist, axis=1)

    candidate_mask = (df['base_collision'] == True) & (df['base_reward'] > 0.85)
    unsafe_mask = df['spawn_dist'] < MIN_SAFE_DIST
    safe_mask = df['spawn_dist'] >= MIN_SAFE_DIST

    removed_df = df[candidate_mask & unsafe_mask]
    
    df_base_fail = df[candidate_mask & safe_mask]

    print(f"Filter Logic:")
    print(f"  1. Base Collision == True")
    print(f"  2. Base Reward > 0.85")
    print(f"  3. Spawn Distance >= {MIN_SAFE_DIST:.2f}")
    print(f"     -> Valid Entries:   {len(df_base_fail)}")
    print(f"     -> Removed Entries: {len(removed_df)}")

    if not removed_df.empty:
        print("\n" + "-"*30)
        print(" [WARNING] The following entries were REMOVED (Spawn Too Close):")
        print("  Format: Seed | Obstacle Pos | Dist to Start")
        print("-" * 30)
        
        removed_sorted = removed_df.sort_values(by='spawn_dist')
        
        for idx, row in removed_sorted.iterrows():
            seed = row['seed']
            obs = row['obs_pos_tuple']
            dist = row['spawn_dist']
            print(f"  - Seed {seed:<4} | Obs {str(obs):<15} | Dist: {dist:.4f} < {MIN_SAFE_DIST}")
        print("-" * 30 + "\n")
    
    total_fails = len(df_base_fail)
    
    if total_fails > 0:
        unique_obstacles_fail = sorted(df_base_fail['obs_pos_tuple'].unique(), key=lambda x: (x[0], x[1]))

        for obs_tuple in unique_obstacles_fail:
            subset = df_base_fail[df_base_fail['obs_pos_tuple'] == obs_tuple]
            
            avg_dist = subset['spawn_dist'].mean()
            
            print(f"\n[Obstacle at {obs_tuple}] (N={len(subset)}) | Avg Dist: {avg_dist:.2f}")
            print(f"  Collision Rate: Base 100.0%  ->  Guided {subset['guided_collision'].mean()*100:5.1f}%")
            print(f"  Success Rate:   Base {subset['base_success'].mean()*100:5.1f}%  ->  Guided {subset['guided_success'].mean()*100:5.1f}%")
            print(f"  Avg Reward:     Base {subset['base_reward'].mean():5.4f} ->  Guided {subset['guided_reward'].mean():5.4f}")

        print("-" * 70)
        print(f"OVERALL AVERAGE (Across {total_fails} valid cases):")
        print(f"  Collision Rate: Base 100.0%  ->  Guided {df_base_fail['guided_collision'].mean()*100:5.1f}%")
        print(f"  Success Rate:   Base {df_base_fail['base_success'].mean()*100:5.1f}%  ->  Guided {df_base_fail['guided_success'].mean()*100:5.1f}%")
        print(f"  Avg Reward:     Base {df_base_fail['base_reward'].mean():5.4f} ->  Guided {df_base_fail['guided_reward'].mean():5.4f}")
        
    else:
        print("No entries found matching criteria.")




import os
import imageio

def save_gif_separated(imgs, 
                       is_success, is_collision,
                       base_collision, base_reward,
                       output_dir, seed, fixed_pos, obs_pct):
    if not imgs:
        return

    if is_success:
        if not base_collision:
            sub_folder = "success/base_no_collision"
        else:
            sub_folder = "success"
            
    else:
        if is_collision:
            sub_folder = "fail/collision"
        else:
            if base_reward < 0.85: 
                sub_folder = "fail/low_reward/base_low_reward"
            else:
                sub_folder = "fail/low_reward"
    
    full_save_dir = os.path.join(output_dir, sub_folder)
    os.makedirs(full_save_dir, exist_ok=True)

    if fixed_pos is not None:
        save_name = f"seed{seed}_pos_{fixed_pos[0]:.2f}_{fixed_pos[1]:.2f}.gif"
    else:
        save_name = f"seed{seed}_pct{obs_pct}.gif"
    
    save_path = os.path.join(full_save_dir, save_name)
    
    try:
        imageio.mimsave(save_path, imgs, fps=20, loop=0)
    except Exception as e:
        print(f"Error saving GIF: {e}")


import time
import torch
import numpy as np

class LatencyMeasurer:
    def __init__(self, warmup_steps=10, chunk_size=1, enabled=True):
        self.warmup_steps = warmup_steps
        self.chunk_size = chunk_size
        self.enabled = enabled
        
        self.raw_times = []
        self.call_count = 0
        self.start_event = None
        self.end_event = None
        
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if not self.enabled:
            return
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_event.record()
        else:
            self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return

        if torch.cuda.is_available():
            self.end_event.record()
            torch.cuda.synchronize()
            # calculate time in ms
            elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
        else:
            self.end_time = time.time()
            elapsed_time_ms = (self.end_time - self.start_time) * 1000

        self.call_count += 1
        if self.call_count > self.warmup_steps:
            self.raw_times.append(elapsed_time_ms)

    def get_results(self):
        if not self.raw_times:
            return {"avg_latency_ms": 0.0, "fps": 0.0}
        
        avg_inference_time_ms = np.mean(self.raw_times)
        
        latency_per_action_ms = avg_inference_time_ms / self.chunk_size
        
        return {
            "latency_per_action_ms": latency_per_action_ms,
            "avg_inference_ms": avg_inference_time_ms,
            "min_ms": np.min(self.raw_times),
            "max_ms": np.max(self.raw_times),
            "samples": len(self.raw_times)
        }
        
    def print_summary(self, label="Model"):
        res = self.get_results()
        print(f"[{label} Latency] Samples: {res['samples']}")
        print(f"  > Latency per Action: {res['latency_per_action_ms']:.2f} ms (Paper Metric)")
        print(f"  > Raw Inference Time: {res['avg_inference_ms']:.2f} ms")



GLOBAL_LAT = {
    "sum_time_ms": 0.0,
    "sum_actions": 0,
    "n_runs": 0,
    "n_records": 0
}

def record_latency_global(latency_tracker):
    res = latency_tracker.get_results()
    samples = int(res.get("samples", 0) or 0)
    avg_inf_ms = float(res.get("avg_inference_ms", 0.0) or 0.0)

    chunk_size = int(getattr(latency_tracker, "chunk_size", 1) or 1)

    if samples <= 0 or chunk_size <= 0:
        return

    GLOBAL_LAT["sum_time_ms"] += avg_inf_ms * samples
    GLOBAL_LAT["sum_actions"] += samples * chunk_size
    GLOBAL_LAT["n_runs"] += 1
    GLOBAL_LAT["n_records"] += samples


def print_latency_global(prefix="[Global Avg Latency]"):
    if GLOBAL_LAT["sum_actions"] <= 0:
        print(f"{prefix} No valid samples collected.")
        return

    mean_ms_per_action = GLOBAL_LAT["sum_time_ms"] / GLOBAL_LAT["sum_actions"]
    print(
        f"{prefix} {mean_ms_per_action:.3f} ms/action | "
        f"runs={GLOBAL_LAT['n_runs']} | records={GLOBAL_LAT['n_records']} | actions={GLOBAL_LAT['sum_actions']}"
    )

import cv2
def draw_vector_arrow(img, start_pt, vector, color, scale=100, thickness=2):
    if vector is None or np.linalg.norm(vector) < 1e-6:
        return
    end_pt = (int(start_pt[0] + vector[0] * scale), 
              int(start_pt[1] + vector[1] * scale))
    cv2.arrowedLine(img, start_pt, end_pt, color, thickness, tipLength=0.3)

def draw_coordinate_axes(img, color=(100, 100, 100)):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(img, (0, cy), (w, cy), color, 1) 
    cv2.line(img, (cx, 0), (cx, h), color, 1) 
    
    tick_color = (60, 60, 60)
    cv2.line(img, (int(w*0.75), 0), (int(w*0.75), h), tick_color, 1, cv2.LINE_AA)
    cv2.line(img, (int(w*0.25), 0), (int(w*0.25), h), tick_color, 1, cv2.LINE_AA)
    cv2.line(img, (0, int(h*0.75)), (w, int(h*0.75)), tick_color, 1, cv2.LINE_AA)
    cv2.line(img, (0, int(h*0.25)), (w, int(h*0.25)), tick_color, 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    cv2.putText(img, "(0,0)", (cx+5, cy+15), font, scale, color, 1)