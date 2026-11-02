"""Differentiable forward kinematics in torch, built from the robot's compiled MuJoCo model.

The same FK computes tracked-body poses and velocities for the reference motion, for simulator states and for
world-model predictions, so the analytic reward the world model is trained through matches the reward the simulator
transitions are labeled with.

Only the root and the ancestors of the tracked bodies are kept. Bodies without a joint (welded links, e.g. the
nemov3 waist) are rigid children of their parent.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from booster_mjlab.supermimic.utils.rotations import axis_angle_to_mat, quat_to_mat

FIXED = -2  # joint index of a welded body
UNLIMITED = 1e4  # range used for continuous joints (wheels)


class Kinematics:
    def __init__(self, model: mujoco.MjModel, joint_names: tuple[str, ...], tracked: tuple[str, ...], device: str):
        """``model``: the robot alone (``Entity.compile()``); ``joint_names``: the order of q / qd."""
        m = model
        self.device = device
        jname_to_idx = {n: i for i, n in enumerate(joint_names)}
        # keep the root and every ancestor of a tracked body (mujoco orders parents before children)
        keep = set()
        for name in tracked:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            assert b > 0, f"tracked body {name} not in the robot model"
            while b > 0:
                keep.add(b)
                b = m.body_parentid[b]
        bodies, parent, joint_idx, axes, offs_p, offs_R = [], [], [], [], [], []
        mj_to_local = {}
        for b in range(1, m.nbody):
            if b not in keep:
                continue
            mj_to_local[b] = len(bodies)
            bodies.append(m.body(b).name)
            if m.body_jntnum[b] == 0:
                parent.append(mj_to_local[m.body_parentid[b]])
                joint_idx.append(FIXED)
                axes.append(np.array([0.0, 0.0, 1.0]))
            else:
                jid = m.body_jntadr[b]
                if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                    parent.append(-1)
                    joint_idx.append(-1)
                else:
                    parent.append(mj_to_local[m.body_parentid[b]])
                    joint_idx.append(jname_to_idx[m.joint(jid).name])
                axes.append(m.jnt_axis[jid].copy())
            offs_p.append(m.body_pos[b].copy())
            offs_R.append(m.body_quat[b].copy())
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(np.array(axes), dtype=torch.float32, device=device)
        self.off_p = torch.tensor(np.array(offs_p), dtype=torch.float32, device=device)
        self.off_R = quat_to_mat(torch.tensor(np.array(offs_R), dtype=torch.float32, device=device))
        self.tracked_idx = [bodies.index(n) for n in tracked]
        self.num_bodies = len(bodies)
        # joint limits in q order; continuous joints get a wide range
        nj = len(joint_names)
        lo, hi = np.full(nj, -UNLIMITED), np.full(nj, UNLIMITED)
        for j in range(m.njnt):
            n = m.joint(j).name
            if n in jname_to_idx and m.jnt_limited[j]:
                lo[jname_to_idx[n]], hi[jname_to_idx[n]] = m.jnt_range[j]
        self.q_lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.q_hi = torch.tensor(hi, dtype=torch.float32, device=device)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions, rotations, linear (link-origin) and angular velocities.

        Args:
            p, R, v, w: root position (B,3), rotation (B,3,3), world lin/ang velocity (B,3).
            q, qd: joint positions/velocities (B,J).
        """
        P, Rs, V, W = [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            pi = P[par] + (Rp @ self.off_p[i].unsqueeze(-1)).squeeze(-1)
            R_off = Rp @ self.off_R[i]
            if j == FIXED:
                Ri, wi = R_off, W[par]
            else:
                Ri = R_off @ axis_angle_to_mat(self.axes[i], q[..., j])
                axis_w = (R_off @ self.axes[i].unsqueeze(-1)).squeeze(-1)
                wi = W[par] + axis_w * qd[..., j : j + 1]
            vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )
