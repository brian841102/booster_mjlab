"""Why PPO symmetry *data augmentation* fails when resuming an asymmetric policy.

rsl_rl's augment_batch gives each mirrored sample (M o, M a) the old log-prob of the original (o, a). The PPO
importance ratio of a mirrored sample is then pi(M a | M o) / pi_old(a | o), which is ~1 only if the policy is
already symmetric. This script samples (o, a) from a stochastic rollout of a checkpoint and reports the
distribution of log pi(M a | M o) - log pi(a | o), plus the policy's symmetry loss MSE(mu(M o), M mu(o)).
usage: TASK=<task> mirror_ratio_check.py <onnx> <ckpt.pt> [steps=600]
"""
import os, sys, numpy as np, torch, onnxruntime as ort
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import booster_mjlab.tasks.velocity.config.nemov3  # noqa: F401
from booster_mjlab.tasks.velocity.mdp.nemov3_symmetry import mirror_actions, mirror_obs

onnx_path, ckpt = sys.argv[1], sys.argv[2]; steps = int(sys.argv[3]) if len(sys.argv) > 3 else 600
cfg = load_env_cfg(os.environ.get('TASK', 'Mjlab-Velocity-Flat-Amp-Clock-Arms-Booster-Nemov3'), play=True); cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
std = torch.load(ckpt, map_location='cpu', weights_only=False)['actor_state_dict']['distribution.std_param'].abs().numpy()
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']); nm = sess.get_inputs()[0].name
mu = lambda o: sess.run(None, {nm: o.astype(np.float32)})[0]
obs, _ = env.reset(); rng = np.random.default_rng(0); lr, sq = [], []
for t in range(steps):
    o = obs['actor'].cpu().numpy(); m = mu(o); a = m + std * rng.standard_normal(m.shape).astype(np.float32)
    if t >= 50:
        mo = mirror_obs(env, 'actor', torch.as_tensor(o)).numpy(); ma = mirror_actions(torch.as_tensor(a)).numpy(); mmo = mu(mo)
        lr.append(float((((a - m) ** 2 - (ma - mmo) ** 2) / (2 * std ** 2)).sum()))
        sq.append(float(((mmo - mirror_actions(torch.as_tensor(m)).numpy()) ** 2).mean()))
    obs, *_ = env.step(torch.as_tensor(a, device='cuda:0'))
lr = np.array(lr)
print(f'MR symmetry loss MSE(mu(Mo), M mu(o)) = {np.mean(sq):.3f} | action std mean {std.mean():.3f}')
print(f'MR log ratio of mirrored samples: mean {lr.mean():.1f}, median {np.median(lr):.1f}, p10 {np.percentile(lr,10):.1f}, p90 {np.percentile(lr,90):.1f} | '
      f'fraction with ratio in the PPO clip range [0.8, 1.2]: {np.mean((lr > np.log(0.8)) & (lr < np.log(1.2)))*100:.1f}% | ratio < 0.01: {np.mean(lr < np.log(0.01))*100:.0f}% | ratio > 100: {np.mean(lr > np.log(100))*100:.0f}%')
env.close()
