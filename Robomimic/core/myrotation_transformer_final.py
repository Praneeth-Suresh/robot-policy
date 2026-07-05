import torch
import torch.nn.functional as F
import numpy as np

class RotationTransformer:
    def __init__(self, from_rep='axis_angle', to_rep='rotation_6d'):
        self.from_rep = from_rep
        self.to_rep = to_rep

    def forward(self, x):
        """
        Forward pass: used during dataset loading.
        Maps input (e.g. axis_angle) -> output (e.g. rotation_6d).
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        
        # 1. Convert to 3x3 matrix first
        if self.from_rep == 'axis_angle':
            matrix = self._axis_angle_to_matrix(x)
        elif self.from_rep == 'rotation_6d':
            matrix = self._rotation_6d_to_matrix(x)
        else:
            raise NotImplementedError(f"Unsupported input format: {self.from_rep}")

        # 2. Convert from 3x3 matrix to target format
        if self.to_rep == 'rotation_6d':
            return self._matrix_to_rotation_6d(matrix)
        elif self.to_rep == 'axis_angle':
            return self._matrix_to_axis_angle(matrix)
        else:
            raise NotImplementedError(f"Unsupported output format: {self.to_rep}")

    def inverse(self, x):
        """
        Inverse pass: used during inference.
        Maps output (e.g. rotation_6d) -> original input (e.g. axis_angle).
        """
        # Inverse = forward with swapped source and target
        temp_transformer = RotationTransformer(from_rep=self.to_rep, to_rep=self.from_rep)
        return temp_transformer.forward(x)

    # ================= Core algorithm implementations =================
    
    def _axis_angle_to_matrix(self, axis_angle):
        """Convert axis-angle to rotation matrix."""
        # Compute angle
        angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
        half_angles = angles * 0.5
        eps = 1e-6
        small_angles = angles.abs() < eps
        
        sin_half_angles_over_angles = torch.empty_like(angles)
        sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
        )
        sin_half_angles_over_angles[small_angles] = 0.5
        
        # Convert to quaternion [w, x, y, z]
        quat = torch.cat(
            [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
        )
        return self._quaternion_to_matrix(quat)

    def _quaternion_to_matrix(self, quaternions):
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    def _rotation_6d_to_matrix(self, d6):
        """6D -> 3x3 Matrix (Gram-Schmidt orthogonalization)."""
        # Numerical stability: 6D may be all zeros / NaN / collinear; naive normalize -> NaN
        eps = 1e-6
        d6 = torch.nan_to_num(d6, nan=0.0, posinf=0.0, neginf=0.0)

        a1, a2 = d6[..., :3], d6[..., 3:]

        # If a1 ≈ 0, fall back to x-axis
        default_b1 = torch.tensor([1.0, 0.0, 0.0], device=d6.device, dtype=d6.dtype)
        a1_norm = torch.linalg.norm(a1, dim=-1, keepdim=True)
        a1 = torch.where(a1_norm < eps, default_b1.expand_as(a1), a1)
        b1 = F.normalize(a1, dim=-1, eps=eps)

        # Remove a2 component along b1
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2_norm = torch.linalg.norm(b2, dim=-1, keepdim=True)

        # If b2 ≈ 0 (a2 is zero or collinear with b1), pick a basis vector least collinear with b1
        cand1 = torch.tensor([0.0, 1.0, 0.0], device=d6.device, dtype=d6.dtype).expand_as(b1)
        cand2 = torch.tensor([0.0, 0.0, 1.0], device=d6.device, dtype=d6.dtype).expand_as(b1)
        dot1 = (b1 * cand1).sum(-1, keepdim=True).abs()
        dot2 = (b1 * cand2).sum(-1, keepdim=True).abs()
        # Tie-break: when equally non-collinear, prefer cand1 (y-axis) so b1=x, a2=0 -> identity rotation
        fallback = torch.where(dot1 <= dot2, cand1, cand2)
        fallback = fallback - (b1 * fallback).sum(-1, keepdim=True) * b1

        b2 = torch.where(b2_norm < eps, fallback, b2)
        b2 = F.normalize(b2, dim=-1, eps=eps)

        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    def _matrix_to_rotation_6d(self, matrix):
        """3x3 Matrix -> 6D (first two columns)."""
        return matrix[..., :2, :].clone().reshape(matrix.shape[:-2] + (6,))

    def _matrix_to_axis_angle(self, matrix):
        return self._quaternion_to_axis_angle(self._matrix_to_quaternion(matrix))
    
    def _matrix_to_quaternion(self, matrix):
        # Numerically stable version (avoid div-by-zero / NaN when trace≈-1 and w≈0)
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

        m = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        batch_dim = m.shape[:-2]

        m00 = m[..., 0, 0]
        m11 = m[..., 1, 1]
        m22 = m[..., 2, 2]
        trace = m00 + m11 + m22

        q = torch.empty(batch_dim + (4,), device=m.device, dtype=m.dtype)

        # trace > 0
        t_mask = trace > 0.0
        if t_mask.any():
            t = trace[t_mask]
            r = torch.sqrt(t + 1.0)
            qw = 0.5 * r
            s = 0.5 / (r + 1e-8)
            qx = (m[t_mask, 2, 1] - m[t_mask, 1, 2]) * s
            qy = (m[t_mask, 0, 2] - m[t_mask, 2, 0]) * s
            qz = (m[t_mask, 1, 0] - m[t_mask, 0, 1]) * s
            q[t_mask] = torch.stack([qw, qx, qy, qz], dim=-1)

        # trace <= 0: pick largest diagonal entry
        f_mask = ~t_mask
        if f_mask.any():
            mm = m[f_mask]
            m00 = mm[..., 0, 0]
            m11 = mm[..., 1, 1]
            m22 = mm[..., 2, 2]

            # idx = argmax([m00,m11,m22])
            stacked = torch.stack([m00, m11, m22], dim=-1)
            idx = torch.argmax(stacked, dim=-1)

            q_sub = torch.empty(mm.shape[:-2] + (4,), device=mm.device, dtype=mm.dtype)

            # idx == 0
            mask0 = idx == 0
            if mask0.any():
                r = torch.sqrt(1.0 + m00[mask0] - m11[mask0] - m22[mask0])
                qx = 0.5 * r
                s = 0.5 / (r + 1e-8)
                qw = (mm[mask0, 2, 1] - mm[mask0, 1, 2]) * s
                qy = (mm[mask0, 0, 1] + mm[mask0, 1, 0]) * s
                qz = (mm[mask0, 0, 2] + mm[mask0, 2, 0]) * s
                q_sub[mask0] = torch.stack([qw, qx, qy, qz], dim=-1)

            # idx == 1
            mask1 = idx == 1
            if mask1.any():
                r = torch.sqrt(1.0 - m00[mask1] + m11[mask1] - m22[mask1])
                qy = 0.5 * r
                s = 0.5 / (r + 1e-8)
                qw = (mm[mask1, 0, 2] - mm[mask1, 2, 0]) * s
                qx = (mm[mask1, 0, 1] + mm[mask1, 1, 0]) * s
                qz = (mm[mask1, 1, 2] + mm[mask1, 2, 1]) * s
                q_sub[mask1] = torch.stack([qw, qx, qy, qz], dim=-1)

            # idx == 2
            mask2 = idx == 2
            if mask2.any():
                r = torch.sqrt(1.0 - m00[mask2] - m11[mask2] + m22[mask2])
                qz = 0.5 * r
                s = 0.5 / (r + 1e-8)
                qw = (mm[mask2, 1, 0] - mm[mask2, 0, 1]) * s
                qx = (mm[mask2, 0, 2] + mm[mask2, 2, 0]) * s
                qy = (mm[mask2, 1, 2] + mm[mask2, 2, 1]) * s
                q_sub[mask2] = torch.stack([qw, qx, qy, qz], dim=-1)

            q[f_mask] = q_sub

        # Normalize to prevent accumulated error
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        q = F.normalize(q, dim=-1, eps=1e-8)
        return q

    def _quaternion_to_axis_angle(self, quaternions):
        norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
        half_angles = torch.atan2(norms, quaternions[..., :1])
        angles = 2 * half_angles
        eps = 1e-6
        small_angles = angles.abs() < eps
        sin_half_angles_over_angles = torch.empty_like(angles)
        sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
        )
        sin_half_angles_over_angles[small_angles] = 0.5
        return quaternions[..., 1:] / sin_half_angles_over_angles
