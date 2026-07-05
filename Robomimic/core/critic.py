
import torch
import torch.nn as nn
from typing import Optional


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Mish()
        )
        
    def forward(self, x):
        return x + self.net(x)

class CollisionPredictionCritic(nn.Module):
    def __init__(
        self,
        action_dim,
        obs_dim,
        obs_horizon,
        hidden_dim=1024,
        depth=6,
        eef_pos_start: Optional[int] = None,
        use_obs_eef_pos: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.eef_pos_start = eef_pos_start
        self.use_obs_eef_pos = use_obs_eef_pos
        
        # 1. Context Encoder: obs_dim-dimensional observation sequence (varies by task)
        self.context_encoder = nn.Sequential(
            nn.Conv1d(obs_dim, 256, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.Mish(),
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten()             
        )
        
        # 2. Geometric Encoder
        # Input dimension breakdown:
        # a (10) + t (1) + obs_pos (3) + rel_vec (3) + dist (1) + alignment (1) = 19
        self.geo_input_dim = action_dim + 1 + 3 + 3 + 1 + 1 
        
        self.geo_encoder = nn.Sequential(
            nn.Linear(self.geo_input_dim, 512),
            nn.Mish(),
            nn.LayerNorm(512)
        )
        
        self.fusion = nn.Linear(512 + 512, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Mish(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, a, t, obs_pos, global_cond):
        """
        a: (B, 10) - Current Action (normalized)
        t: (B, 1) or (B,)
        obs_pos: (B, 3) - virtual obstacle 3D position (normalized)
        global_cond: (B, obs_dim * obs_horizon) - flattened observation
        """
        B = a.shape[0]
        if t.dim() == 1: t = t.unsqueeze(-1)
        
        # 1. Context encoding
        seq_len = global_cond.shape[1] // self.obs_dim
        obs_seq = global_cond.view(B, seq_len, self.obs_dim)  # (B, T, obs_dim)
        ctx_feat = self.context_encoder(obs_seq.transpose(1, 2))

        # 2. Geometric features (XYZ position only, first 3 dims)
        # Cross-task/control: prefer eef_pos from obs (normalized); avoid treating a[:3] as current position
        if self.use_obs_eef_pos and (self.eef_pos_start is not None):
            curr_pos = obs_seq[:, -1, self.eef_pos_start : self.eef_pos_start + 3]  # (B, 3)
        else:
            curr_pos = a[:, :3] # (B, 3)
        
        rel_vec = obs_pos - curr_pos # (B, 3)
        dist = torch.norm(rel_vec, dim=-1, keepdim=True) # (B, 1)
        
        # Alignment score
        # Normalized action vector (position motion direction only)
        pos_vel_norm = curr_pos / (torch.norm(curr_pos, dim=-1, keepdim=True) + 1e-7)
        obs_dir = rel_vec / (dist + 1e-7)
        alignment = (pos_vel_norm * obs_dir).sum(dim=-1, keepdim=True)
        
        # Concatenate all geometric features
        geo_in = torch.cat([a, t, obs_pos, rel_vec, dist, alignment], dim=-1)
        geo_feat = self.geo_encoder(geo_in)

        # 3. Fusion and residual inference
        x = torch.cat([ctx_feat, geo_feat], dim=-1)
        x = self.fusion(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

# =========================================================
# 2. Risk assessment logic (Labeling Function - 3D Version)
# =========================================================

def compute_hybrid_risk_3d(trajectory, obs_pos, r_obs=0.07, sharpness=100.0):
    """
    trajectory: (Steps, Total_B, 10) - predicted action trajectory
    obs_pos: (Total_B, 3) - virtual obstacle position
    """
    # XYZ position only (first 3 dims)
    traj_pos = trajectory[..., :3] # (Steps, B, 3)
    
    # 1. Collision probability (Prob Hit) - min distance from trajectory to obstacle
    # dists: (Steps, B)
    dists = torch.norm(traj_pos - obs_pos.unsqueeze(0), p=2, dim=-1)
    min_dist = torch.min(dists, dim=0)[0] # (B,)
    
    # r_obs is collision radius; 0.05 in normalized space ≈ 2.5–5 cm in physical space
    prob_hit = torch.sigmoid(sharpness * (r_obs - min_dist))
    
    # 2. Aim probability (Prob Aim) - whether initial motion points toward obstacle
    start_pos = traj_pos[0] # (B, 3)
    future_idx = min(5, traj_pos.shape[0]-1)
    future_pos = traj_pos[future_idx] 
    
    vec_move = future_pos - start_pos
    dist_move = torch.norm(vec_move, p=2, dim=-1) + 1e-7
    dir_move = vec_move / dist_move.unsqueeze(-1)
    
    vec_to_obs = obs_pos - start_pos
    dist_to_obs = torch.norm(vec_to_obs, p=2, dim=-1) + 1e-7
    dir_to_obs = vec_to_obs / dist_to_obs.unsqueeze(-1)
    
    alignment = torch.relu((dir_move * dir_to_obs).sum(dim=-1))
    
    # Distance decay: far away -> less dangerous even if aligned
    dist_factor = torch.exp(-(dist_to_obs**2) / (2 * 0.5**2))
    
    prob_aim = (alignment ** 2) * dist_factor
    
    # Visualize
    # if 0.6 *prob_aim[0] > prob_hit[0]:
    #     print("Probaim", 0.6 *prob_aim[0])
    # if prob_hit[0] > 0.6 * prob_aim[0]:
    #     print("Prob_____hit", prob_hit[0])


    return torch.maximum(prob_hit, 0.6 * prob_aim)


