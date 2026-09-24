"""What does the AMP discriminator of a checkpoint reject? Scores per commanded speed + input sensitivity.

usage: disc_by_speed.py <task> <checkpoint.pt> [envs=32] [cmds=0,0.3,0.6,0.8,1.0,1.3] [steps=600]

Loads actor AND discriminator. For every commanded speed, runs `envs` robots (deterministic policy), records the
AMP observation each step and builds discriminator samples exactly like training (K consecutive frames; windows that
would need backfill are dropped). Scores:
D(expert) = sigmoid(logit), with the minibatch-std scalar computed over all policy samples of all speeds together
(the training batch is mixed too). Printed per speed: mean / p10 / p90 of D, style reward softplus(logit), and the
mean |d logit / d input| per AMP joint (which joints the discriminator is using against the policy). The same for
the reference samples (all clips and augmentations), and for the policy samples mirrored left/right.
"""

import sys
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import booster_mjlab.tasks.velocity.config.nemov3  # noqa: F401
from booster_mjlab.robots.nemov3.nemov3_constants import nemov3_mirror_maps


def main():
    task, ckpt = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    cmds = [float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,0.3,0.6,0.8,1.0,1.3").split(",")]
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 600

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = n
    tw = env_cfg.commands["twist"]
    tw.ranges.lin_vel_y = (0.0, 0.0)
    tw.ranges.ang_vel_z = (0.0, 0.0)
    tw.rel_standing_envs = 0.0
    tw.rel_forward_envs = 0.0
    agent_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(task))(venv, asdict(agent_cfg), device="cuda:0")
    runner.load(ckpt, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    amp = runner.alg.amp
    disc = amp.discriminator.eval()
    K, s = amp.num_amp_obs_steps, 1  # K consecutive frames per discriminator sample
    span = (K - 1) * s + 1
    amp_names = [n for n in env.scene["robot"].joint_names]
    term = env.command_manager.get_term("twist")
    joint_ids = env.observation_manager._group_obs_term_cfgs["amp"][0].params["asset_cfg"].joint_ids
    names = [amp_names[i] for i in joint_ids]
    idx, sign = nemov3_mirror_maps(names)  # mirror within the AMP joint set (every joint has its partner here)

    def windows(seq):  # seq [env, time, dim] -> [samples, K*dim]
        out = [seq[:, t - span + 1 : t + 1 : s].reshape(seq.shape[0], -1) for t in range(span + 50, seq.shape[1], 5)]
        return torch.cat(out)

    samples = {}
    for c in cmds:
        term.cfg.ranges.lin_vel_x = (c, c)
        seq = []
        with torch.inference_mode():
            obs, _ = venv.reset()
            for _ in range(steps):
                obs, *_ = venv.step(policy(obs))
                seq.append(obs["amp"].clone())
        samples[c] = windows(torch.stack(seq, 1))
    ref = amp.amp_loader.all_obs
    ref = ref[torch.randperm(len(ref), device=ref.device)[:20000]]

    def trunk(x):
        return disc.trunk(disc._normalize(x))

    def logits(x, s_scalar):
        h = trunk(x)
        return disc.linear(torch.cat([h, s_scalar.expand(len(h), 1)], -1)).squeeze(-1) if disc.use_minibatch_std else disc.linear(h).squeeze(-1)

    with torch.no_grad():
        allpol = torch.cat(list(samples.values()))
        s_pol = trunk(allpol).std(0, unbiased=False).mean().reshape(1, 1) if disc.use_minibatch_std else None
        s_ref = trunk(ref).std(0, unbiased=False).mean().reshape(1, 1) if disc.use_minibatch_std else None
    print(f"K={K} stride={s} span={span} | minibatch-std scalar: policy batch {s_pol.item() if s_pol is not None else 0:.3f}, "
          f"reference batch {s_ref.item() if s_ref is not None else 0:.3f}")
    print("AMP joints:", names)

    def report(tag, x, s_scalar):
        x = x.detach().clone().requires_grad_(True)
        lg = logits(x, s_scalar)
        g = torch.autograd.grad(lg.sum(), x)[0].abs().reshape(len(x), K, -1).mean((0, 1))
        d = torch.sigmoid(lg).detach().cpu().numpy()
        r = F.softplus(lg).detach().mean().item()
        print(f"{tag:>18s}: D mean {d.mean():.3f} p10 {np.percentile(d, 10):.3f} p90 {np.percentile(d, 90):.3f} "
              f"| style reward {r:.3f} | |dlogit/dx| " + " ".join(f"{v:.2f}" for v in g.cpu().numpy()), flush=True)

    report("reference", ref, s_ref)
    report("reference s_pol", ref, s_pol)
    for c, x in samples.items():
        report(f"policy cmd {c:.2f}", x, s_pol)
    sign_t = torch.tensor(sign, device=allpol.device)
    mirrored = (allpol.reshape(len(allpol), K, -1)[:, :, idx] * sign_t).reshape(len(allpol), -1)
    report("policy mirrored", mirrored, s_pol)
    report("policy all", allpol, s_pol)
    env.close()


if __name__ == "__main__":
    main()
