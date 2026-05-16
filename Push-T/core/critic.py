import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # Add the specific numpy scalar to the allowed list
# torch.serialization.add_safe_globals([np._core.multiarray.scalar])
# torch.serialization.add_safe_globals([
#     np.dtype, 
#     np._core.multiarray.scalar, 
#     np._core.multiarray._reconstruct
# ])


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
    def __init__(self, action_dim, obs_dim, obs_horizon, hidden_dim=1024, depth=6):
        super().__init__()
        self.obs_dim = obs_dim
        self.context_encoder = nn.Sequential(
            nn.Conv1d(obs_dim, 256, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.Mish(),
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten()             
        )
        self.geo_encoder = nn.Sequential(
            nn.Linear(9, 512),
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
        B = a.shape[0]
        if t.dim() == 1: t = t.unsqueeze(-1)
        
        total_elements = global_cond.numel()
        seq_len = (total_elements // B) // self.obs_dim
        
        ctx = global_cond.view(B, seq_len, self.obs_dim).transpose(1, 2)
        ctx_feat = self.context_encoder(ctx)

        rel_vec = obs_pos - a
        dist = torch.norm(rel_vec, dim=-1, keepdim=True)
        a_norm = a / (torch.norm(a, dim=-1, keepdim=True) + 1e-7)
        obs_dir = rel_vec / (dist + 1e-7)
        alignment = (a_norm * obs_dir).sum(dim=-1, keepdim=True)
        
        geo_in = torch.cat([a, t, obs_pos, rel_vec, dist, alignment], dim=-1)
        geo_feat = self.geo_encoder(geo_in)

        x = torch.cat([ctx_feat, geo_feat], dim=-1)
        x = self.fusion(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def compute_hybrid_risk(trajectory, obs_pos, r_obs=0.05, sharpness=60.0):
    # trajectory: (Steps, Total_B, 2), obs_pos: (Total_B, 2)
    dists = torch.norm(trajectory - obs_pos.unsqueeze(0), p=2, dim=-1)
    min_dist = torch.min(dists, dim=0)[0] 
    prob_hit = torch.sigmoid(sharpness * (r_obs - min_dist))
    
    start_pos = trajectory[0]
    future_idx = min(5, trajectory.shape[0]-1)
    future_pos = trajectory[future_idx] 
    vec_move = future_pos - start_pos
    dist_move = torch.norm(vec_move, p=2, dim=-1) + 1e-7
    dir_move = vec_move / dist_move.unsqueeze(-1)
    
    vec_to_obs = obs_pos - start_pos
    dist_to_obs = torch.norm(vec_to_obs, p=2, dim=-1) + 1e-7
    dir_to_obs = vec_to_obs / dist_to_obs.unsqueeze(-1)
    
    alignment = torch.relu((dir_move * dir_to_obs).sum(dim=-1))
    dist_factor = torch.exp(-(dist_to_obs**2) / (2 * 0.8**2))
    prob_aim = (alignment ** 2) * dist_factor
    

    # prob_aim_term = 0.6 * (alignment ** 2) * dist_factor
    # is_hit_dominant = prob_hit > prob_aim_term
    # print("-" * 30)
    # print(f"[Risk Debug] Batch Size: {prob_hit.shape[0]}")
    # print(f"  > Mean prob_hit:     {prob_hit.mean().item():.4f}")
    # print(f"  > Mean prob_aim_0.6: {prob_aim_term.mean().item():.4f}")
    # print(f"  > Count dominated by hit: {is_hit_dominant.sum().item()} ({(is_hit_dominant.float().mean().item()*100):.1f}%)")
    # print(f"  > Count dominated by aim: {(~is_hit_dominant).sum().item()}")



    # return torch.maximum(prob_hit, 0.6 * prob_aim)
    return prob_hit


rewards= 0