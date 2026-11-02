"""Analytic, differentiable MDP heads shared by the simulator and the world model (paper Appendix A).

Everything here is a function of the state tuple ``S = (p, R, v, w, q, qd)`` (``booster_mjlab.supermimic.state``)
and the reference frame index ``f``: the simulator labels its transitions with it, and the policy is trained by
backpropagating it through world-model rollouts. Hence no mjlab reward / observation managers.
"""

from __future__ import annotations

import torch

from booster_mjlab.supermimic.state import invariant_features
from booster_mjlab.supermimic.utils.rotations import mat_to_6d, rot_angle_sq, yaw_mat
from booster_mjlab.tasks.supermimic.mdp.kinematics import Kinematics
from booster_mjlab.tasks.supermimic.mdp.motion import CONTROL_DT, MotionLib


class TrackingMdp:
    """Observation, reward, termination and critic features for tracking ``motion`` (cfg: ``SuperMimicCfg``)."""

    def __init__(self, cfg, kin: Kinematics, motion: MotionLib, default_q: torch.Tensor, pos_mask: torch.Tensor):
        self.cfg, self.kin, self.motion = cfg, kin, motion
        dev = default_q.device
        self.default_q = default_q
        self.pos_mask = pos_mask.to(dev)
        J = self.num_joints = default_q.shape[-1]
        self.num_tracked = len(cfg.tracked_bodies)
        self.obs_dim = 3 * J + 12
        self.xi_dim = 1 + J + 3  # friction, joint offsets, CoM offset
        self.critic_extra = J + 3 * self.num_tracked + 3 + 6 + self.xi_dim
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - half * cfg.soft_limit_factor, mid + half * cfg.soft_limit_factor
        ls = cfg.length_scale
        self.std = {"body_pos": 0.3 * ls, "body_ori": 0.4, "body_linvel": 1.0 * ls, "body_angvel": 3.14, "root_pos": 0.3 * ls, "root_ori": 0.4}
        # wheel / foot height term (robots with lift_points only)
        self.lift_idx = [cfg.tracked_bodies.index(b) for b, _ in cfg.lift_points]
        self.lift_offset = torch.tensor([o for _, o in cfg.lift_points], device=dev).reshape(-1, 3)
        self.w_lift = 1.0 if self.lift_idx else 0.0
        self.max_step_reward = (4 * 1.0 + 2 * 0.5 + self.w_lift) * CONTROL_DT
        # Table A.1 sensor noise (uniform half-widths), plus the reference-relative orientation
        self.obs_noise = torch.cat(
            (torch.full((J,), 0.01, device=dev) * self.pos_mask, torch.full((J,), 0.5, device=dev), torch.full((3,), 0.2, device=dev),
             torch.full((3,), 0.05, device=dev), torch.zeros(J, device=dev), torch.full((6,), 0.05, device=dev))
        )  # fmt: skip

    def lift_height(self, bp, bR):
        """World height of the lift points from tracked body poses (..., NUM_TRACKED, 3 / 3x3)."""
        idx = self.lift_idx
        return bp[..., idx, 2] + (bR[..., idx, :, :] @ self.lift_offset.unsqueeze(-1)).squeeze(-1)[..., 2]

    # ---------------- observation ----------------
    def obs(self, S, a_prev, xi, f) -> torch.Tensor:
        """Noise-free proprioceptive observation (+ reference-relative root orientation)."""
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        dq0 = xi[..., 1 : 1 + self.num_joints]
        g_b = -R[..., 2, :]  # R^T (0,0,-1)
        rel = mat_to_6d(Rt @ self.motion.root_R[f])
        return torch.cat(((q - self.default_q - dq0) * self.pos_mask, qd, (Rt @ w.unsqueeze(-1)).squeeze(-1), g_b, a_prev, rel), -1)

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        return o + (2 * torch.rand_like(o) - 1) * self.obs_noise

    # ---------------- reward / termination ----------------
    def _ref_bodies(self, p, R, f):
        """Reference bodies re-anchored to the robot's root (xy + yaw), as in BeyondMimic."""
        m = self.motion
        rp, rR = m.root_p[f], m.root_R[f]
        dR = yaw_mat(R @ rR.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + ((dR.unsqueeze(-3) @ (m.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1))
        return ref_bp, dR

    def reward(self, S, f, a, a_prev, terms: bool = False):
        """Per-step reward (already multiplied by dt) comparing S with reference frame f."""
        c, m, std = self.cfg, self.motion, self.std
        p, R, v, w, q, qd = S
        bp, bR, bv, bw = self.kin.forward(p, R, v, w, q, qd)
        rp, rR = m.root_p[f], m.root_R[f]
        ref_bp, dR = self._ref_bodies(p, R, f)
        ref_bR = dR.unsqueeze(-3) @ m.body_R[f]
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((m.body_v[f] - bv) ** 2).sum(-1).mean(-1)
        e_av = ((m.body_w[f] - bw) ** 2).sum(-1).mean(-1)
        e_rp = ((rp - p) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, R)
        r_track = (
            torch.exp(-e_pos / std["body_pos"] ** 2)
            + torch.exp(-e_ori / std["body_ori"] ** 2)
            + torch.exp(-e_lv / std["body_linvel"] ** 2)
            + torch.exp(-e_av / std["body_angvel"] ** 2)
            + 0.5 * torch.exp(-e_rp / std["root_pos"] ** 2)
            + 0.5 * torch.exp(-e_ro / std["root_ori"] ** 2)
        )
        r_floor = torch.zeros_like(r_track)
        if self.w_lift:
            z_ref, z = self.lift_height(m.body_p[f], m.body_R[f]), self.lift_height(bp, bR)
            e_lift = ((z_ref - z) ** 2).mean(-1)
            r_track = r_track + self.w_lift * torch.exp(-e_lift / c.lift_std**2)
            # per metre a lifted reference point (above the lowest one by > lift_floor_on) is below its reference:
            # linear, so its pull does not fade with distance like the exp term
            up = (z_ref - z_ref.min(-1, keepdim=True).values) > c.lift_floor_on
            r_floor = (torch.relu(z_ref - z) * up).sum(-1)
        r_rate = ((a - a_prev) ** 2).sum(-1)
        r_lim = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        r = (r_track + c.w_action_rate * r_rate + c.w_joint_limit * r_lim + c.w_lift_floor * r_floor) * CONTROL_DT
        if terms:
            return r, {
                "track": r_track * CONTROL_DT,
                "e_body_pos": e_pos,
                "e_body_ori": e_ori,
                "e_root_pos": e_rp,
                "action_rate": r_rate,
                "joint_limit": r_lim,
                **({"e_lift": e_lift, "lift_floor": r_floor} if self.w_lift else {}),
            }
        return r

    def termination(self, S, f) -> torch.Tensor:
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        return (e_z > self.cfg.term_height) | (e_tilt > self.cfg.term_tilt)

    def soft_termination(self, S, f, tz: float = 0.02, tt: float = 0.04) -> torch.Tensor:
        """Differentiable termination probability for imagined rollouts."""
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        alive = (1 - torch.sigmoid((e_z - self.cfg.term_height) / tz)) * (1 - torch.sigmoid((e_tilt - self.cfg.term_tilt) / tt))
        return 1 - alive

    # ---------------- critic / sim context ----------------
    def critic_features(self, S, f, xi_n) -> torch.Tensor:
        """Critic input: invariant state, sim context, and robot-vs-reference errors (the goal is appended by the runner)."""
        m = self.motion
        p, R, v, w, q, qd = S
        Ht = yaw_mat(R).transpose(-1, -2)
        rp, rR = m.root_p[f], m.root_R[f]
        bp = self.kin.forward(p, R, v, w, q, qd)[0]
        ref_bp, _ = self._ref_bodies(p, R, f)
        e_body = (Ht.unsqueeze(-3) @ (ref_bp - bp).unsqueeze(-1)).squeeze(-1).flatten(-2)
        e_root = (Ht @ (rp - p).unsqueeze(-1)).squeeze(-1)
        rel = mat_to_6d(R.transpose(-1, -2) @ rR)
        return torch.cat((invariant_features(S, self.pos_mask), (m.q[f] - q) * self.pos_mask, e_body, e_root, rel, xi_n), -1)

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        """xi = [friction, joint offsets (J), CoM offset (3)] scaled to roughly unit range."""
        lo, hi = self.cfg.friction_range
        J = self.num_joints
        return torch.cat(
            ((xi[..., :1] - (lo + hi) / 2) / ((hi - lo) / 2), xi[..., 1 : 1 + J] / self.cfg.joint_offset_range, xi[..., 1 + J :] / self.cfg.com_range[1]),
            -1,
        )
