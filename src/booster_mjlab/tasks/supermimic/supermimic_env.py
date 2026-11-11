"""Batched motion-tracking simulator on an mjlab env (SuperMimic paper Appendix D.1).

A thin layer over ``ManagerBasedRlEnv``: it sets the joint targets, steps the physics, labels each transition with
the analytic reward / termination of ``mdp/tracking.py`` and resets episodes to reference-motion states drawn by
return-based bin sampling. The world model learns the dynamics s_{t+1} = f(s_t, a_t, xi) of exactly this loop.
"""

from __future__ import annotations

import os

import torch
from mjlab.envs import ManagerBasedRlEnv

from booster_mjlab import BOOSTER_MJLAB_SRC_PATH
from booster_mjlab.supermimic.state import flatten_state
from booster_mjlab.supermimic.utils.rotations import mat_to_quat, quat_to_mat, so3_exp
from booster_mjlab.tasks.supermimic.mdp.kinematics import Kinematics
from booster_mjlab.tasks.supermimic.mdp.motion import CONTROL_DT, MotionLib
from booster_mjlab.tasks.supermimic.mdp.tracking import TrackingMdp
from booster_mjlab.tasks.supermimic.supermimic_env_cfg import SuperMimicEnvCfg

REPO_ROOT = str(BOOSTER_MJLAB_SRC_PATH.parents[1])


def roll_action_hist(a_hist: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Previous-action history (..., J * H), newest first, after action a."""
    return torch.cat((a, a_hist[..., : a_hist.shape[-1] - a.shape[-1]]), -1) if a_hist.shape[-1] else a_hist


class SuperMimicEnv:
    def __init__(self, env: ManagerBasedRlEnv, gamma: float = 0.99, motion_dir: str | None = None, action_hist: int = 0):
        cfg: SuperMimicEnvCfg = env.cfg  # type: ignore[assignment]
        self.env, self.cfg, self.sm = env, cfg, cfg.supermimic
        assert abs(env.step_dt - CONTROL_DT) < 1e-9, f"SuperMimic runs at {1 / CONTROL_DT:.0f} Hz, env step_dt {env.step_dt}"
        self.device = dev = env.device
        self.robot = robot = env.scene["robot"]
        self.N = env.num_envs
        self.gamma = gamma
        self.eval_mode = cfg.eval_mode
        self.max_episode_steps = self.sm.max_episode_steps if not self.eval_mode else 10**9
        self.reset_noise = self.sm.reset_noise and not self.eval_mode
        self.joint_names = tuple(robot.joint_names)
        J = len(self.joint_names)
        self.vel = torch.tensor([n in self.sm.velocity_joints for n in self.joint_names], device=dev)
        self.pos = (~self.vel).float()
        self.alpha = torch.where(self.vel, self.sm.velocity_action_scale, self.sm.action_scale)
        self.default_q = robot.data.default_joint_pos[0].clone()
        self.q_adr = robot.data.indexing.free_joint_q_adr
        self.v_adr = robot.data.indexing.free_joint_v_adr

        # analytic MDP and reference motion, built from the robot model itself
        robot_model = cfg.scene.entities["robot"].spec_fn().compile()  # the robot alone (its entity spec is attached)
        self.kin = Kinematics(robot_model, self.joint_names, self.sm.tracked_bodies, dev)
        self.motion_dir = os.path.join(REPO_ROOT, motion_dir or self.sm.motion_dir)  # relative paths: to the repo root
        self.motion = MotionLib.from_dir(
            self.motion_dir,
            kin=self.kin,
            joint_names=self.joint_names,
            csv_joint_names=self.sm.csv_joint_names,
            pos_mask=self.pos,
            device=dev,
        )
        self.mdp = TrackingMdp(self.sm, self.kin, self.motion, self.default_q, self.pos)

        # ---- sim context xi (layout: TrackingMdp) ----
        # the startup events drew friction, CoM (and with sim2real mass, inertia, PD gains, ground contact): read back
        self.xi = torch.zeros(self.N, self.mdp.xi_dim, device=dev)
        self._read_model_context()
        self.s2r = self.sm.sim2real
        self.wheel_limit = self.s2r.wheel_speed_limit if self.s2r else None
        self.next_push = torch.zeros(self.N, dtype=torch.long, device=dev)

        # ---- episode bookkeeping ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, J, device=dev)
        self.a_hist = torch.zeros(self.N, J * action_hist, device=dev)  # a_{t-1}, ..., a_{t-H} (world model input)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = self.mdp.max_step_reward / (1 - gamma)
        self.bin_G = torch.full((self.motion.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((self.motion.num_bins,), 1.0 / self.motion.num_bins, device=dev)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.reset(torch.arange(self.N, device=dev))

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def _read_model_context(self):
        env, robot, dev, J, b = self.env, self.robot, self.device, self.num_joints, self.mdp.xi_base
        model, mj = env.sim.model, env.sim.mj_model

        def field(name: str) -> torch.Tensor:  # (N, ...) whether or not a DR event expanded it per env
            x = torch.as_tensor(getattr(model, name), device=dev)
            return x.expand(self.N, *x.shape[1:]) if x.shape[0] == 1 else x

        def default(name: str, idx: int) -> torch.Tensor:
            return torch.as_tensor(getattr(mj, name)[idx], dtype=torch.float32, device=dev)

        geom = int(robot.data.indexing.geom_ids[robot.find_geoms(self.sm.friction_geoms)[0][0]])
        self.xi[:, 0] = field("geom_friction")[:, geom, 0]
        if self.sm.com_body:
            torso = int(robot.data.indexing.body_ids[robot.find_bodies(self.sm.com_body)[0][0]])
            self.xi[:, 1 + J : b] = field("body_ipos")[:, torso] - default("body_ipos", torso)
        if not self.sm.sim2real:
            return
        self.xi[:, b] = field("body_mass")[:, torso] - default("body_mass", torso)
        # principal moments, sorted (pseudo_inertia re-diagonalizes, so the principal axes may come out in another order)
        moments = field("body_inertia")[:, torso].sort(-1).values
        self.xi[:, b + 1 : b + 4] = torch.log(moments / default("body_inertia", torso).sort().values)
        # joint -> its actuator: kp = gainprm[0], kd = -biasprm[2] (position); kv = gainprm[0] = -biasprm[2] (wheels)
        jids = robot.data.indexing.joint_ids.tolist()
        ctrl = [next(a for a in range(mj.nu) if mj.actuator_trnid[a, 0] == j) for j in jids]
        self.xi[:, b + 4 : b + 4 + J] = field("actuator_gainprm")[:, ctrl, 0] / default("actuator_gainprm", ctrl)[:, 0]
        self.xi[:, b + 4 + J : b + 4 + 2 * J] = field("actuator_biasprm")[:, ctrl, 2] / default("actuator_biasprm", ctrl)[:, 2]
        ground = int(env.scene["terrain"].data.indexing.geom_ids[0]) if "terrain" in env.scene.entities else 0
        o = b + 4 + 3 * J
        self.xi[:, o : o + 2] = field("geom_solref")[:, ground, :2]
        self.xi[:, o + 2 : o + 5] = field("geom_solimp")[:, ground, :3]

    def _sample_delays(self, env_ids: torch.Tensor):
        """Per-episode command lags (AMP Exp 85+ merged delay), recorded in xi [s]."""
        from booster_mjlab.tasks.velocity.mdp.motor_dynamics import sample_actuator_delay

        d = self.s2r.actuator_delay
        lags = sample_actuator_delay(self.env, env_ids, d.groups, d.max_lag, d.cmd_groups, d.cmd_delay_range_s)
        o = self.mdp.xi_base + 4 + 2 * self.num_joints
        for j, name in enumerate(self.joint_names):
            self.xi[env_ids, o + j] = lags[name].float() * self.env.physics_dt

    def _push(self, done: torch.Tensor) -> torch.Tensor:
        """Random horizontal root velocity kicks, applied after the transition was recorded. -> the envs pushed."""
        cut = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        if not self.s2r or not self.s2r.push_interval_s or self.eval_mode:
            return cut
        cut = (self.ep_step >= self.next_push) & ~done
        ids = cut.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            _, _, v, w = self.state()[:4]
            v = v[ids].clone()
            v[:, :2] += torch.empty(len(ids), 2, device=self.device).uniform_(-self.s2r.push_vel_xy, self.s2r.push_vel_xy)
            self.robot.write_root_link_velocity_to_sim(torch.cat((v, w[ids]), -1), env_ids=ids)
            self._schedule_push(ids)
        return cut

    def _schedule_push(self, env_ids: torch.Tensor):
        lo, hi = self.s2r.push_interval_s
        n = len(env_ids)
        self.next_push[env_ids] = self.ep_step[env_ids] + torch.empty(n, device=self.device).uniform_(lo / CONTROL_DT, hi / CONTROL_DT).long()

    # ------------------------------------------------------------------ state
    def state(self):
        """Read from qpos / qvel directly (current right after a reset or a physics step, no forward() needed)."""
        d = self.robot.data.data
        qpos, qvel = d.qpos[:, self.q_adr], d.qvel[:, self.v_adr]
        p = qpos[:, :3] - self.env.scene.env_origins
        R = quat_to_mat(qpos[:, 3:7])  # wxyz
        w = (R @ qvel[:, 3:6].unsqueeze(-1)).squeeze(-1)  # free joint: angular velocity in the body frame
        return (p, R, qvel[:, :3].clone(), w, self.robot.data.joint_pos.clone(), self.robot.data.joint_vel.clone())

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])  # keep >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        p, R, v, w, q, qd = (x.clone() for x in self.motion.ref_state(frames))
        if self.reset_noise:
            u = lambda *r: torch.stack([torch.empty(n, device=dev).uniform_(-h, h) for h in r], -1)  # noqa: E731
            p = p + u(0.05, 0.05, 0.01)
            R = so3_exp(u(0.1, 0.1, 0.2)) @ R
            v = v + u(0.5, 0.5, 0.2)
            w = w + u(0.52, 0.52, 0.78)
            q = q + torch.empty_like(q).uniform_(-0.1, 0.1)
        q = torch.clamp(q, self.kin.q_lo, self.kin.q_hi)
        # clear the solver warm start, actuator state and targets first (both zero the targets set below)
        self.env.sim.reset(env_ids)
        self.env.scene.reset(env_ids)
        root = torch.cat((p + self.env.scene.env_origins[env_ids], mat_to_quat(R), v, w), -1)
        self.robot.write_root_state_to_sim(root, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        self.robot.set_joint_velocity_target(qd * self.vel, env_ids=env_ids)
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.a_hist[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        if not self.eval_mode:
            J = self.num_joints
            self.xi[env_ids, 1 : 1 + J] = torch.empty(n, J, device=dev).uniform_(-1, 1) * self.sm.joint_offset_range * self.pos
            if self.s2r and self.s2r.actuator_delay:
                self._sample_delays(env_ids)
            if self.s2r and self.s2r.push_interval_s:
                self._schedule_push(env_ids)

    # ------------------------------------------------------------------ step
    def _physics_step(self):
        env = self.env
        for _ in range(env.cfg.decimation):
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(dt=env.physics_dt)

    def step(self, a: torch.Tensor) -> dict:
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        J = self.num_joints
        # position joints: target = default + offset + scale * a; velocity joints (wheels): speed target = scale * a
        self.robot.set_joint_position_target(self.default_q + xi[:, 1 : 1 + J] + self.alpha * a * self.pos)
        wheel = self.alpha * a * self.vel
        if self.wheel_limit:
            wheel = wheel.clamp(-self.wheel_limit, self.wheel_limit)
        self.robot.set_joint_velocity_target(wheel)
        self._physics_step()
        S1 = self.state()
        f1 = f + 1
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        term = self.mdp.termination(S1, f1)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r,
            "term": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms, "a_hist": self.a_hist.clone(),
        }  # fmt: skip
        out["cut"] = self._push(done)  # the next transition starts from a kicked state: world-model windows stop here
        # bookkeeping
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        self.a_hist = roll_action_hist(self.a_hist, a)
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # bin return estimate; truncated episodes bootstrap with their mean reward
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G)
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * self.mdp.max_step_reward)
            self.reset(ids)
        return out
