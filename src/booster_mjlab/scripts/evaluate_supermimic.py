"""Evaluate SuperMimic checkpoints on every clip of their motion library (SuperMimic paper Appendix F.1).

Each clip plays once from its first frame with friction 1.0, no domain randomization, no observation or action noise.
A clip fails if the rollout triggers the training termination. The normalized return is the undiscounted reward sum
divided by its maximum attainable value.

    uv run evaluate-supermimic Mjlab-SuperMimic-Flat-Booster-Nemov3 --run-dir logs/rsl_rl/nemov3_supermimic/latest --watch True

writes ``eval.csv`` and, per checkpoint, ``model_<it>_eval.json`` and the rollout ``model_<it>_traj.npz`` into the
run directory. ``--checkpoint`` evaluates one file and prints the result.
"""

from __future__ import annotations

import glob
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Annotated

import numpy as np
import torch
import tyro

from mjlab import TYRO_FLAGS


@dataclass(frozen=True)
class EvalConfig:
    task: Annotated[str, tyro.conf.Positional]
    run_dir: str | None = None
    """Run directory to evaluate (``logs/rsl_rl/<experiment>/latest`` follows the newest run)."""
    checkpoint: str | None = None
    """Evaluate this checkpoint only."""
    watch: bool = False
    """Keep evaluating new checkpoints as they appear."""
    every: int = 1
    """Evaluate every n-th checkpoint."""
    motion_dir: str | None = None
    """Default: the checkpoint's motion directory."""
    device: str = "cuda:0"


def _iteration(path: str) -> int:
    return int(re.fullmatch(r"model_(\d+)\.pt", os.path.basename(path)).group(1))


class Evaluator:
    def __init__(self, task: str, motion_dir: str, device: str):
        import mjlab.tasks  # noqa: F401  (registers the tasks)
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        from booster_mjlab.tasks.supermimic.supermimic_env import SuperMimicEnv

        cfg = load_env_cfg(task, play=True)
        assert cfg.eval_mode, f"{task}: the play env of a SuperMimic task must have eval_mode=True"
        cfg.scene.num_envs = len(glob.glob(os.path.join(motion_dir, "*.csv")))
        self.device = device
        self.sim = SuperMimicEnv(ManagerBasedRlEnv(cfg, device=device), motion_dir=motion_dir)

    @torch.no_grad()
    def evaluate(self, ck: dict, record: str | None = None) -> dict:
        import booster_mjlab.supermimic.modules.common as common
        from booster_mjlab.supermimic.runners.supermimic_runner import build_networks
        from booster_mjlab.supermimic.utils.rotations import mat_to_quat

        sim, mdp, motion, dev = self.sim, self.sim.mdp, self.sim.motion, self.device
        c = ck["cfg"]
        K, Hh = c["goal_frames"], c["hist"]
        common.AMP_DTYPE = torch.bfloat16 if c["amp_dtype"] == "bf16" else None
        pi = build_networks(sim, c, dev)[1]
        pi.load_state_dict(ck["pi"])
        pi.eval()
        C = sim.N
        sim.done_eval[:] = False
        sim.failed_eval[:] = False
        sim.reset(torch.arange(C, device=dev), motion.clip_start.clone())
        clip_end = motion.clip_end.clone()
        ret = torch.zeros(C, device=dev)
        hist = None
        traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
        while not bool(sim.done_eval.all()):
            S = sim.state()
            o = mdp.obs(S, sim.a_prev, sim.xi, sim.f)
            hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
            a = pi(hist, motion.goal(sim.f, K))
            alive = ~sim.done_eval.clone()
            if record:
                traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
                traj["q"].append(S[4].cpu()), traj["f"].append(sim.f.cpu()), traj["alive"].append(alive.cpu())
            out = sim.step(a)
            ret += torch.where(alive, out["r"], 0.0)
            sim.f = torch.minimum(sim.f, clip_end - 1)  # finished clips keep stepping; keep f+1 in range
        msr = mdp.max_step_reward
        norm = ret / ((motion.clip_len - 1).float() * msr)
        res = {
            "norm_return": norm.mean().item(),
            "norm_return_framew": (ret.sum() / ((motion.clip_len - 1).sum() * msr)).item(),
            "n_failed": int(sim.failed_eval.sum().item()),
            "failed": [motion.names[i] for i in sim.failed_eval.nonzero().flatten().tolist()],
            "per_clip": {motion.names[i]: round(norm[i].item(), 4) for i in range(C)},
        }
        if record:
            np.savez_compressed(record, names=np.array(motion.names), joint_names=np.array(sim.joint_names),
                                **{k: torch.stack(v).numpy() for k, v in traj.items()},
                                ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                                clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy())  # fmt: skip
        return res


def main():
    args = tyro.cli(EvalConfig, config=TYRO_FLAGS)
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        ev = Evaluator(args.task, args.motion_dir or ck["motion_dir"], args.device)
        res = ev.evaluate(ck)
        print(json.dumps(res, indent=1))
        return
    assert args.run_dir, "give --run-dir or --checkpoint"
    run_dir = os.path.realpath(args.run_dir)  # .../latest -> the run it points to now
    out_csv = os.path.join(run_dir, "eval.csv")
    done = set()
    if os.path.exists(out_csv):
        done = {int(line.split(",")[0]) for line in open(out_csv).read().strip().splitlines()[1:]}
    else:
        open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
    ev = None
    k = 0
    while True:
        ckpts = sorted(glob.glob(os.path.join(run_dir, "model_*.pt")), key=_iteration)
        for path in [p for p in ckpts if _iteration(p) not in done]:
            k += 1
            if (k - 1) % args.every:
                continue
            it = _iteration(path)
            ck = torch.load(path, map_location=args.device, weights_only=False)
            ev = ev or Evaluator(args.task, args.motion_dir or ck["motion_dir"], args.device)
            t0 = time.time()
            res = ev.evaluate(ck, path.replace(".pt", "_traj.npz"))
            line = (f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},"
                    f"{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'])}\n")  # fmt: skip
            open(out_csv, "a").write(line)
            json.dump(res, open(path.replace(".pt", "_eval.json"), "w"), indent=1)
            print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{ev.sim.N} "
                  f"{res['per_clip']} ({time.time() - t0:.0f}s)", flush=True)  # fmt: skip
            done.add(it)
        if not args.watch:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
