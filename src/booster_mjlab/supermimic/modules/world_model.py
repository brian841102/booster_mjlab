"""Deterministic world model s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments (paper Table A.4)."""

from __future__ import annotations

import torch
import torch.nn as nn

from booster_mjlab.supermimic.modules.common import RunningNorm, mlp, run_mlp
from booster_mjlab.supermimic.state import invariant_dim, invariant_features
from booster_mjlab.supermimic.utils.rotations import so3_exp, so3_log, yaw_mat


class WorldModel(nn.Module):
    def __init__(self, num_joints: int, xi_dim: int, pos_mask: torch.Tensor, action_hist: int = 0, hidden=(2048, 2048, 2048)):
        super().__init__()
        self.num_joints = num_joints
        self.out_dim = 12 + 2 * num_joints
        self.in_dim = invariant_dim(num_joints) + num_joints * (1 + action_hist) + xi_dim
        self.net = mlp(self.in_dim, list(hidden), self.out_dim, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)
        self.register_buffer("pos_mask", pos_mask.float().clone())

    def inputs(self, S, a, xi_n, a_hist):
        """a_hist: (..., num_joints * action_hist) previous actions, newest first (empty without action history)."""
        return torch.cat((invariant_features(S, self.pos_mask), a, a_hist, xi_n), -1)

    @staticmethod
    def targets(S, S1):
        """Frame-local increments from S to S1 (the regression targets)."""
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ R1),
                (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def step(self, S, a, xi_n, a_hist):
        p, R, v, w, q, qd = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, a, xi_n, a_hist)))
        d = y * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        J = self.num_joints
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, J, J), -1)
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        return (p1, R1, v1, w1, q + dq, qd + dqd)

    @staticmethod
    def state_error(S_hat, S):
        """Frame-local state error components (same layout as ``targets``)."""
        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (ph - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ Rh),
                (Rt @ (vh - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (wh - w).unsqueeze(-1)).squeeze(-1),
                qh - q,
                qdh - qd,
            ),
            -1,
        )
