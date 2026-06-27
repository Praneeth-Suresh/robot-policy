import torch
import torch.nn.functional as F
import numpy as np

class RotationTransformer:
    def __init__(self, from_rep='axis_angle', to_rep='rotation_6d'):
        self.from_rep = from_rep
        self.to_rep = to_rep

    def forward(self, x):
        """
        前向传播：用于 Dataset 加载阶段
        将输入 (如 axis_angle) -> 输出 (如 rotation_6d)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        
        # 1. 先转为 3x3 矩阵
        if self.from_rep == 'axis_angle':
            matrix = self._axis_angle_to_matrix(x)
        elif self.from_rep == 'rotation_6d':
            matrix = self._rotation_6d_to_matrix(x)
        else:
            raise NotImplementedError(f"不支持的输入格式: {self.from_rep}")

        # 2. 再从 3x3 矩阵转为目标格式
        if self.to_rep == 'rotation_6d':
            return self._matrix_to_rotation_6d(matrix)
        elif self.to_rep == 'axis_angle':
            return self._matrix_to_axis_angle(matrix)
        else:
            raise NotImplementedError(f"不支持的输出格式: {self.to_rep}")

    def inverse(self, x):
        """
        逆向传播：用于推理阶段
        将输出 (如 rotation_6d) -> 还原为输入 (如 axis_angle)
        """
        # 巧妙利用：逆变换 = 交换源和目标后的 forward
        temp_transformer = RotationTransformer(from_rep=self.to_rep, to_rep=self.from_rep)
        return temp_transformer.forward(x)

    # ================= 核心算法实现 =================
    
    def _axis_angle_to_matrix(self, axis_angle):
        """将轴角转换为旋转矩阵"""
        # 计算角度
        angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
        half_angles = angles * 0.5
        eps = 1e-6
        small_angles = angles.abs() < eps
        
        sin_half_angles_over_angles = torch.empty_like(angles)
        sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
        )
        sin_half_angles_over_angles[small_angles] = 0.5
        
        # 转换为四元数 [w, x, y, z]
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
        """6D -> 3x3 Matrix (Gram-Schmidt 正交化)"""
        # 数值稳定：6D 可能出现全 0 / NaN / 两列共线，直接 normalize 会产生 NaN
        eps = 1e-6
        d6 = torch.nan_to_num(d6, nan=0.0, posinf=0.0, neginf=0.0)

        a1, a2 = d6[..., :3], d6[..., 3:]

        # 若 a1 近似为 0，则回退到 x 轴
        default_b1 = torch.tensor([1.0, 0.0, 0.0], device=d6.device, dtype=d6.dtype)
        a1_norm = torch.linalg.norm(a1, dim=-1, keepdim=True)
        a1 = torch.where(a1_norm < eps, default_b1.expand_as(a1), a1)
        b1 = F.normalize(a1, dim=-1, eps=eps)

        # 对 a2 去除在 b1 上的分量
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2_norm = torch.linalg.norm(b2, dim=-1, keepdim=True)

        # 若 b2 近似为 0（a2 为 0 或与 b1 共线），选择一个与 b1 尽量不共线的基向量作为回退
        cand1 = torch.tensor([0.0, 1.0, 0.0], device=d6.device, dtype=d6.dtype).expand_as(b1)
        cand2 = torch.tensor([0.0, 0.0, 1.0], device=d6.device, dtype=d6.dtype).expand_as(b1)
        dot1 = (b1 * cand1).sum(-1, keepdim=True).abs()
        dot2 = (b1 * cand2).sum(-1, keepdim=True).abs()
        # tie-break：当两者同样“不共线”时，优先选 cand1（y 轴），使 b1=x 且 a2=0 的情况回退为单位旋转
        fallback = torch.where(dot1 <= dot2, cand1, cand2)
        fallback = fallback - (b1 * fallback).sum(-1, keepdim=True) * b1

        b2 = torch.where(b2_norm < eps, fallback, b2)
        b2 = F.normalize(b2, dim=-1, eps=eps)

        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    def _matrix_to_rotation_6d(self, matrix):
        """3x3 Matrix -> 6D (取前两列)"""
        return matrix[..., :2, :].clone().reshape(matrix.shape[:-2] + (6,))

    def _matrix_to_axis_angle(self, matrix):
        return self._quaternion_to_axis_angle(self._matrix_to_quaternion(matrix))
    
    def _matrix_to_quaternion(self, matrix):
        # 数值稳定版本（避免 trace≈-1 时 w≈0 导致除 0 / NaN）
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

        # trace <= 0：选对角线最大项
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

        # 规范化，防止累积误差
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