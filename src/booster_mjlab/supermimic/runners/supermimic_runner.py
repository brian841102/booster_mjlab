"""SuperMimic training loop (Algorithm A.1): collect -> fit world model -> BPTT policy + critic update.

Plugs into mjlab's ``train`` like an rsl_rl runner: ``runner_cls(env, train_cfg, log_dir, device)`` then
``learn(num_learning_iterations)``. The env must be a SuperMimic task (``tasks/supermimic``); the runner drives it
through ``SuperMimicEnv`` instead of ``env.step``.
"""

from __future__ import annotations

import math
import os
import time

import torch
import torch.nn as nn

import booster_mjlab.supermimic.modules.common as common
from booster_mjlab.supermimic.modules import Critic, Policy, WorldModel
from booster_mjlab.supermimic.state import invariant_dim, unflatten_state
from booster_mjlab.supermimic.storage import ReplayBuffer
from booster_mjlab.tasks.supermimic.mdp import goal_frame_dim
from booster_mjlab.tasks.supermimic.supermimic_env import SuperMimicEnv, roll_action_hist

LOG_COLUMNS = (
    "iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,"
    "term_frac,it_per_s,lr"
)


def build_networks(sim: SuperMimicEnv, cfg: dict, device: str):
    """World model, policy and critic for ``sim``'s robot and the runner settings ``cfg``."""
    mdp, J = sim.mdp, sim.num_joints
    goal_dim = goal_frame_dim(J) * cfg["goal_frames"]
    wm = WorldModel(J, mdp.xi_dim, sim.pos, action_hist=cfg.get("wm_action_hist", 0)).to(device)
    pi = Policy(mdp.obs_dim, goal_dim, J, hist=cfg["hist"]).to(device)
    critic = Critic(invariant_dim(J) + mdp.critic_extra + goal_dim).to(device)
    return wm, pi, critic


class SuperMimicRunner:
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu", **kwargs) -> None:
        del kwargs
        self.cfg = c = train_cfg
        self.device = dev = device
        self.log_dir = log_dir
        self.git_status_repos: list[str] = []
        torch.manual_seed(c["seed"])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        common.AMP_DTYPE = torch.bfloat16 if c["amp_dtype"] == "bf16" else None

        self.env = env
        self.sim = sim = SuperMimicEnv(env.unwrapped, gamma=c["gamma"], action_hist=c["wm_action_hist"])
        self.mdp, self.motion = sim.mdp, sim.motion
        self.wm, self.pi, self.critic = build_networks(sim, c, dev)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=c["lr"])
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=c["lr"], betas=(0.9, 0.95))
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=c["lr"])
        self.buffer = ReplayBuffer(c["buffer_steps"], sim.N, sim.num_joints, self.mdp.xi_dim, dev, c["wm_action_hist"])
        self.obs_hist = torch.zeros(sim.N, c["hist"], self.mdp.obs_dim, device=dev)
        self.current_learning_iteration, self.wall_offset, self.sim_steps = 0, 0.0, 0
        J = sim.num_joints
        # per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
        self.err_floor = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                                    torch.full((J,), 3e-3), torch.full((J,), 5e-2))).to(dev)  # fmt: skip
        self.wm_rollout = torch.compile(self._wm_rollout)
        self.wm_err = torch.compile(self._wm_err)
        self.imag_step = torch.compile(self._imag_step)
        self.reward_term = torch.compile(self._reward_term)
        self.critic_input_c = torch.compile(self._critic_input)
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            latest = os.path.join(os.path.dirname(log_dir), "latest")  # -> newest run, for evaluate-supermimic --watch
            if os.path.islink(latest) or not os.path.exists(latest):
                if os.path.islink(latest):
                    os.remove(latest)
                os.symlink(os.path.basename(log_dir), latest)

    # ------------------------------------------------------------------ compiled pieces
    def _wm_rollout(self, s, a, xi, ah):
        J = self.sim.num_joints
        S = unflatten_state(s[:, 0], J)
        preds = []
        for j in range(self.cfg["wm_window"]):
            S = self.wm.step(S, a[:, j], self.mdp.normalize_xi(xi[:, j]), ah)
            ah = roll_action_hist(ah, a[:, j])
            preds.append(S)
        return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)

    def _wm_err(self, S_hat, s1, mask):
        std = torch.maximum(self.wm.out_norm.std(), self.err_floor)
        e = (self.wm.state_error(S_hat, unflatten_state(s1, self.sim.num_joints)) / std) ** 2  # (B, k, D)
        e = e.mean(-1)
        return (e * mask).sum() / s1.shape[0], e[:, -1]

    def _imag_step(self, S, o_hist, a_prev, f, xi, xi_n, g, ah):
        a = self.pi(o_hist, g)
        S1 = self.wm.step(S, a, xi_n, ah)
        f1 = f + 1
        o1 = self.mdp.add_noise(self.mdp.obs(S1, a, xi, f1))
        o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
        return S1, a, o_hist, f1, self.motion.goal(f1, self.cfg["goal_frames"]), roll_action_hist(ah, a)

    def _reward_term(self, S, f, a, a_prev):
        return self.mdp.reward(S, f, a, a_prev), self.mdp.soft_termination(S, f)

    def _critic_input(self, S, f, xi_n, g):
        return torch.cat((self.mdp.critic_features(S, f, xi_n), g), -1)

    # ------------------------------------------------------------------ updates
    def lr_at(self, it: int) -> float:
        c = self.cfg
        if it >= c["lr_decay_iters"]:
            return c["lr"] * 0.03
        return c["lr"] * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / c["lr_decay_iters"])))

    def wm_update(self):
        c = self.cfg
        w = self.buffer.sample_windows(c["wm_batch"], c["wm_window"])
        S_hat = self.wm_rollout(w["s"], w["a"], w["xi"], w["a_hist"])
        loss, e_last = self.wm_err(S_hat, w["s1"], w["mask"])
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        m = w["mask"][:, -1]
        return loss.detach() / c["wm_window"], (e_last.detach() * m).sum() / m.sum().clamp(min=1)

    def sample_valid_starts(self, B: int):
        starts, hists, n = [], [], 0
        while n < B:
            s, h = self.buffer.sample_starts(2 * B, self.cfg["hist"])
            f = s["f"]
            ok = f + self.cfg["horizon"] <= self.motion.clip_end[self.motion.clip_id[f]]
            idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
            starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
            n += len(idx)
        cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
        return cat(starts), cat(hists)

    def policy_update(self):
        c, mdp, motion, dev = self.cfg, self.mdp, self.motion, self.device
        J, K, HZ, B = self.sim.num_joints, c["goal_frames"], c["horizon"], c["pi_batch"]
        s, h = self.sample_valid_starts(B)
        S = unflatten_state(s["s"], J)
        xi, f, a_prev, ah = s["xi"], s["f"], s["a_prev"], s["a_hist"]
        xi_n = mdp.normalize_xi(xi)
        with torch.no_grad():
            o_hist = mdp.obs(unflatten_state(h["s"], J), h["a_prev"], h["xi"], h["f"])
        o_hist = mdp.add_noise(o_hist)
        _set_requires_grad(self.wm, False)
        _set_requires_grad(self.critic, False)
        g0 = motion.goal(f, K)
        g = g0
        S0 = S
        Ss, As, Aps, Fs, Gs = [], [], [], [], []
        for _ in range(HZ):
            Aps.append(a_prev)
            S, a_prev, o_hist, f, g, ah = self.imag_step(S, o_hist, a_prev, f, xi, xi_n, g, ah)
            Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        r, d = self.reward_term(S_all, flat(Fs), flat(As), flat(Aps))
        r, d = r.view(HZ, B), d.view(HZ, B)
        alive = torch.cumprod(1 - d, 0)  # c_{t+1}
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = c["gamma"] ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        Jret = (disc * c_t * r).sum(0)
        surv = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, dones = r.detach(), d.detach()
        if c["use_critic"]:
            Jret = Jret + (c["gamma"] ** HZ) * surv * self.critic(self._critic_input(S, f, xi_n, g))
        loss = -Jret.mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        if torch.isfinite(gn):
            self.opt_pi.step()
        _set_requires_grad(self.wm, True)
        _set_requires_grad(self.critic, True)
        # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
        v_loss = torch.zeros(())
        if c["use_critic"]:
            with torch.no_grad():
                S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
                f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
                g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
                xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
                X = self.critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
            self.critic.norm.update(X)
            with torch.no_grad():
                V = self.critic(X)
                R = V[-1]
                targets = []
                for t in range(HZ - 1, -1, -1):
                    R = rews[t] + c["gamma"] * (1 - dones[t]) * ((1 - c["lam"]) * V[t + 1] + c["lam"] * R)
                    targets.append(R)
                targets = torch.stack(targets[::-1])  # (H, B)
            Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
            n = Xf.shape[0]
            mb = n // c["critic_minibatches"]
            for _ in range(c["critic_epochs"]):
                perm = torch.randperm(n, device=dev)
                for i in range(c["critic_minibatches"]):
                    idx = perm[i * mb : (i + 1) * mb]
                    v_loss = ((self.critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                    self.opt_v.zero_grad(set_to_none=True)
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.opt_v.step()
        return loss.detach(), v_loss.detach(), (r_sum / HZ / 0.1).mean(), gn.detach(), surv.mean().detach()

    @torch.no_grad()
    def collect(self, n_steps: int, stats: dict):
        sim, mdp, pi, K = self.sim, self.mdp, self.pi, self.cfg["goal_frames"]
        for _ in range(n_steps):
            S = sim.state()
            o = mdp.add_noise(mdp.obs(S, sim.a_prev, sim.xi, sim.f))
            fresh = sim.ep_step == 0
            self.obs_hist = torch.where(
                fresh[:, None, None], o.unsqueeze(1).expand_as(self.obs_hist), torch.cat((self.obs_hist[:, 1:], o.unsqueeze(1)), 1)
            )
            g = self.motion.goal(sim.f, K)
            pi.obs_norm.update(o)
            pi.goal_norm.update(g)
            a = pi(self.obs_hist, g)
            a = torch.clamp(a + self.cfg["action_noise"] * torch.randn_like(a), -20, 20)
            out = sim.step(a)
            self.buffer.add(out)
            J = sim.num_joints
            S0, S1 = unflatten_state(out["s"], J), unflatten_state(out["s1"], J)
            self.wm.in_norm.update(self.wm.inputs(S0, out["a"], mdp.normalize_xi(out["xi"]), out["a_hist"]))
            self.wm.out_norm.update(self.wm.targets(S0, S1))
            self.sim_steps += sim.N
            stats["r"].append(out["r"].mean() / 0.1)
            if "ep_len" in out:
                stats["ep_len"].append(out["ep_len"].mean())
                stats["ep_ret"].append(out["ep_ret_norm"].mean())
                stats["term"].append(out["term"].float().sum())
                stats["done"].append(out["done"].float().sum())

    # ------------------------------------------------------------------ main loop
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        del init_at_random_ep_len
        c = self.cfg
        start_it = self.current_learning_iteration
        total = start_it + num_learning_iterations
        stats = {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
        if self.buffer.size == 0:  # a new run, or resumed (the replay buffer is not saved)
            self.collect(c["warmup_steps"], stats)
        log_path = os.path.join(self.log_dir, "train_log.csv")
        new_log = not os.path.exists(log_path)
        log_f = open(log_path, "a")
        if new_log:
            log_f.write(LOG_COLUMNS + "\n")
        writer = None
        if c["logger"] == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(self.log_dir)
        t_start = time.time() - self.wall_offset
        t_log = time.time()
        step_acc = 0.0
        agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
        for it in range(start_it, total):
            lr = self.lr_at(it)
            for o in (self.opt_wm, self.opt_pi, self.opt_v):
                for gparam in o.param_groups:
                    gparam["lr"] = lr
            step_acc += 1.0 / c["utd"]
            n = int(step_acc)
            step_acc -= n
            self.collect(n, stats)
            for _ in range(c["wm_batches"]):
                wl, e8 = self.wm_update()
            agg["wm"].append(wl), agg["wm8"].append(e8)
            if it >= c["wm_only_iters"]:
                pl, vl, ir, gn, surv = self.policy_update()
                agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
            self.current_learning_iteration = it + 1
            if (it + 1) % c["log_interval"] == 0:
                mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
                ips = c["log_interval"] / (time.time() - t_log)
                t_log = time.time()
                wall_h = (time.time() - t_start) / 3600
                term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
                row = [it + 1, wall_h, self.sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
                       mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean(stats["ep_len"]), mean(stats["ep_ret"]), term_frac, ips, lr]  # fmt: skip
                log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
                log_f.flush()
                if writer is not None:
                    for name, x in zip(LOG_COLUMNS.split(",")[1:], row[1:]):
                        writer.add_scalar(f"SuperMimic/{name}", x, it + 1)
                print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} "
                      f"surv {row[9]:.3f} | sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | {ips:.2f} it/s",
                      flush=True)  # fmt: skip
                stats = {k: [] for k in stats}
                agg = {k: [] for k in agg}
            every = c["save_interval_late"] if c["save_interval_late"] and it + 1 > c["save_late_from"] else c["save_interval"]
            if (it + 1) % every == 0 or it + 1 == total:
                self.wall_offset = time.time() - t_start
                self.save(os.path.join(self.log_dir, f"model_{it + 1}.pt"))
        log_f.close()
        if writer is not None:
            writer.close()

    # ------------------------------------------------------------------ checkpoints
    def save(self, path: str, infos=None) -> None:
        """``model_<it>.pt``: networks + config (what evaluation needs); ``optim.pt`` next to it: the optimizer state of
        the newest checkpoint, for resuming."""
        del infos
        ck = {
            "wm": self.wm.state_dict(), "pi": self.pi.state_dict(), "critic": self.critic.state_dict(),
            "iteration": self.current_learning_iteration, "wall_time": self.wall_offset, "sim_steps": self.sim_steps,
            "cfg": self.cfg, "joint_names": self.sim.joint_names, "motion_dir": self.sim.motion_dir,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }  # fmt: skip
        _atomic_save(ck, path)
        optim = {"opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict(), "opt_v": self.opt_v.state_dict(),
                 "iteration": self.current_learning_iteration}  # fmt: skip
        _atomic_save(optim, os.path.join(os.path.dirname(path), "optim.pt"))

    def load(self, path: str, load_optimizer: bool = True, **kwargs):
        """Load a checkpoint (``kwargs``: rsl_rl's load options from mjlab's play, unused)."""
        del kwargs
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.wm.load_state_dict(ck["wm"]), self.pi.load_state_dict(ck["pi"]), self.critic.load_state_dict(ck["critic"])
        self.current_learning_iteration, self.wall_offset, self.sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
        optim_path = os.path.join(os.path.dirname(path), "optim.pt")
        if load_optimizer and os.path.exists(optim_path):
            optim = torch.load(optim_path, map_location=self.device, weights_only=False)
            if optim["iteration"] == ck["iteration"]:
                self.opt_wm.load_state_dict(optim["opt_wm"])
                self.opt_pi.load_state_dict(optim["opt_pi"])
                self.opt_v.load_state_dict(optim["opt_v"])
        return ck

    def get_inference_policy(self, device: str | None = None) -> SuperMimicPlayPolicy:
        """For mjlab's ``play``: a stateful policy that also keeps the reference frame (see SuperMimicPlayPolicy)."""
        del device
        return SuperMimicPlayPolicy(self)

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)


class SuperMimicPlayPolicy:
    """Deterministic policy for mjlab's ``play`` viewers, which loop ``action = policy(obs); env.step(action)``.

    The tracking state is not in the env's managers, so this keeps it: env i plays clip ``i % num_clips`` from its
    first frame, the reference frame advances one per call, and a clip restarts at its end or on a fall (the training
    termination); the viewer's reset (``episode_length_buf == 0``) restarts every clip. Observations are computed from
    the simulator state (no noise); the action is returned in the order of the env's action terms.
    """

    def __init__(self, runner: SuperMimicRunner):
        self.sim, self.pi, self.cfg = runner.sim, runner.pi, runner.cfg
        self.pi.eval()
        env, sim = self.sim.env, self.sim
        # action manager order (joint_pos targets, then wheel_vel) -> robot joint order of the policy output
        am = env.action_manager
        self.order = torch.cat([am.get_term(n).target_ids.long() for n in am.active_terms]).to(sim.device)
        assert len(self.order) == sim.num_joints, "the action terms must cover every joint"
        self.hist = None
        self.started = False

    def _restart(self, ids: torch.Tensor):
        sim, m = self.sim, self.sim.motion
        sim.reset(ids, m.clip_start[ids % len(m.names)])
        sim.env.sim.forward()

    @torch.no_grad()
    def __call__(self, obs=None) -> torch.Tensor:
        del obs
        sim, mdp, motion, K = self.sim, self.sim.mdp, self.sim.motion, self.cfg["goal_frames"]
        everything = torch.arange(sim.N, device=sim.device)
        if not self.started or bool((sim.env.episode_length_buf == 0).all()):
            self._restart(everything)
            self.started, self.hist = True, None
        else:  # one env step has passed since the last call
            sim.f = sim.f + 1
            done = mdp.termination(sim.state(), sim.f) | (sim.f >= motion.clip_end[motion.clip_id[sim.f - 1]])
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(ids) > 0:
                self._restart(ids)
        S = sim.state()
        o = mdp.obs(S, sim.a_prev, sim.xi, sim.f)
        fresh = (sim.ep_step == 0)[:, None, None]
        if self.hist is None:
            self.hist = o.unsqueeze(1).repeat(1, self.cfg["hist"], 1)
        self.hist = torch.where(fresh, o.unsqueeze(1).expand_as(self.hist), torch.cat((self.hist[:, 1:], o.unsqueeze(1)), 1))
        a = self.pi(self.hist, motion.goal(sim.f, K))
        sim.a_prev = a.clone()  # the raw action, as the hardware observes it
        sim.ep_step = sim.ep_step + 1
        if sim.wheel_limit:  # the wheel speed target clamp of SuperMimicEnv.step, in action units
            lim = sim.wheel_limit / sim.alpha
            a = torch.where(sim.vel, a.clamp(-lim, lim), a)
        return a[:, self.order]


def _set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


def _atomic_save(obj, path: str):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)
