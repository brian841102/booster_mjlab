"""Permutation importance + temporal-coherence test for a trained AMP discriminator.
usage: disc_permutation.py <onnx> <ckpt.pt> <scale_mode: v0|v5|v6|cur> <tag> [task] [K]
Layout is derived from the env: per step [joint_pos(nj), joint_vel(nj), base_lin_vel(3), projected_gravity(3)]."""
import sys, numpy as np, torch, onnxruntime as ort
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import booster_mjlab.tasks.velocity.config.nemov3
from booster_mjlab.motion.motion_loader import MotionLoader
from booster_mjlab.amp.modules.discriminator import Discriminator
from booster_mjlab.amp.utils.default_joint_positions import _resolve_default_joint_positions
from booster_mjlab.tasks.velocity.config.nemov3.rl_cfg import nemov3_amp_ppo_runner_cfg

onnx_path, ckpt_path, mode, tag = sys.argv[1:5]
TASK = sys.argv[5] if len(sys.argv) > 5 else 'Mjlab-Velocity-Flat-Amp-Booster-Nemov3'
K = int(sys.argv[6]) if len(sys.argv) > 6 else 10
cfg = load_env_cfg(TASK, play=True); cfg.scene.num_envs = 1
if mode in ('v0', 'v5', 'v6'):
    from booster_mjlab.robots.nemov3 import nemov3_constants as C
    sc = C._build_action_scale(C.get_spec(), factor=1.0 if mode == 'v6' else 0.25)
    if mode == 'v5': sc.update({'right_hip_pitch_joint': .25, 'left_hip_pitch_joint': .25, 'right_calf_joint': .35, 'left_calf_joint': .35})
    cfg.actions['joint_pos'].scale = sc
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
env.command_manager.get_term('play_controls').push_enabled = True     # training has pushes
rc = nemov3_amp_ppo_runner_cfg()
class W: pass
w = W(); w.unwrapped = env
default_pose, joint_ids = _resolve_default_joint_positions(w)
jnames = [env.scene['robot'].joint_names[i].replace('_joint', '') for i in joint_ids]; nj = len(jnames)
BASE = int(__import__('os').environ.get('AMP_BASE', '1')); D = 2 * nj + (6 if BASE else 0)
loader = MotionLoader(dataset_root=rc.dataset_root, simulation_dt=env.cfg.sim.mujoco.timestep * env.cfg.decimation,
                      speed_factor=rc.speed_factor, num_amp_obs_steps=K, augmentations=rc.dataset_augmentations,
                      default_pose=default_pose, joint_indices=joint_ids, include_base_lin_vel=bool(BASE),
                      include_base_ang_vel=False, include_projected_gravity=bool(BASE), device='cuda:0')
exp_all, pw = loader.all_obs, loader.per_frame_weights
assert exp_all.shape[1] == K * D, (exp_all.shape, K, D)
disc = Discriminator(input_dim=K * D, hidden_layer_sizes=[256, 128], reward_scale=1.0, reward_clamp_epsilon=1e-4,
                     loss_type='bce', use_minibatch_std=True, empirical_normalization=True, device='cuda:0')
ck = torch.load(ckpt_path, map_location='cuda:0', weights_only=False)
disc.load_state_dict(ck['discriminator_state_dict']); disc.eval()
act_std = ck['actor_state_dict']['distribution.std_param'].cpu().numpy()

sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']); inn = sess.get_inputs()[0].name
obs, _ = env.reset(); A = []
for t in range(3200):
    a = sess.run(None, {inn: obs['actor'].cpu().numpy().astype(np.float32)})[0]
    a = a + act_std[None, :] * np.random.randn(*a.shape).astype(np.float32)        # stochastic policy like training
    obs, *_ = env.step(torch.as_tensor(a, device='cuda:0'))
    if t % 200 == 199: obs, _ = env.reset()          # match 4 s training episodes
    elif t % 200 >= 30: A.append(obs['amp'][0])
A = torch.stack(A); assert A.shape[1] == D
pol_all = torch.stack([A[i:i + K].reshape(-1) for i in range(len(A) - K + 1)])
env.close()

# feature groups: per joint type (pos / vel over both sides) + base velocity + gravity
groups = {}
for j in sorted({n.split('_', 1)[1] if n.startswith(('right_', 'left_')) else n for n in jnames}):
    idx = [i for i, n in enumerate(jnames) if n.endswith(j)]
    groups[f'pos {j}'] = idx; groups[f'vel {j}'] = [nj + i for i in idx]
if BASE: groups['base_lin_vel xyz'] = [2 * nj, 2 * nj + 1, 2 * nj + 2]; groups['proj_gravity xyz'] = [2 * nj + 3, 2 * nj + 4, 2 * nj + 5]
gcols = {g: torch.tensor([s * D + f for s in range(K) for f in fs], device='cuda:0') for g, fs in groups.items()}

B, NB = 2048, 8
gen = torch.Generator(device='cuda:0').manual_seed(0)
def batch():
    ip = torch.randint(len(pol_all), (B,), device='cuda:0', generator=gen)
    return pol_all[ip], exp_all[torch.multinomial(pw, B, replacement=True, generator=gen)]
def score(x):
    with torch.no_grad():
        lg = disc(x); return torch.sigmoid(lg).mean().item(), torch.nn.functional.softplus(lg).mean().item()
def avg(fn): return np.array([fn(*batch()) for _ in range(NB)]).mean(0)
def replaced(p, e, gs):
    x = p.clone()
    for g in gs: x[:, gcols[g]] = e[:, gcols[g]]
    return x
def time_shuffled(x):     # random frame order inside each window -> destroys temporal coherence, keeps per-frame marginals
    y = x.view(-1, K, D); perm = torch.stack([torch.randperm(K, device=x.device) for _ in range(y.shape[0])])
    return torch.gather(y, 1, perm[:, :, None].expand(-1, -1, D)).reshape(x.shape)

base = avg(lambda p, e: score(p)); ex = avg(lambda p, e: score(e))
print(f'RESULT[{tag}] K={K} D_feat={D} | expert: D={ex[0]:.3f} r/step={ex[1]:.3f} | policy: D={base[0]:.3f} r/step={base[1]:.3f}')
exs = avg(lambda p, e: score(time_shuffled(e))); pls = avg(lambda p, e: score(time_shuffled(p)))
print(f'RESULT[{tag}] TEMPORAL: expert windows time-shuffled D={exs[0]:.3f} (was {ex[0]:.3f}) | policy windows time-shuffled D={pls[0]:.3f} (was {base[0]:.3f})')
names = list(groups)
knock_in = {g: avg(lambda p, e, g=g: score(replaced(p, e, [g]))) for g in names}
print(f"RESULT[{tag}] {'group':22s} | fix ONLY this group: D (dR/step)")
for g in sorted(names, key=lambda g: -(knock_in[g][1] - base[1])):
    print(f'RESULT[{tag}] {g:22s} | D {knock_in[g][0]:.3f} ({knock_in[g][1]-base[1]:+.3f})')
order = sorted(names, key=lambda g: -(knock_in[g][1] - base[1])); cum = []
for g in order:
    cum.append(g); r = avg(lambda p, e: score(replaced(p, e, cum)))
    print(f'RESULT[{tag}]   cumulative +{g:20s} -> D {r[0]:.3f} r/step {r[1]:.3f}')
