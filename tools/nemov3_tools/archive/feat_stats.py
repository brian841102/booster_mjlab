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
D = 2 * nj + 6
loader = MotionLoader(dataset_root=rc.dataset_root, simulation_dt=env.cfg.sim.mujoco.timestep * env.cfg.decimation,
                      speed_factor=rc.speed_factor, num_amp_obs_steps=K, augmentations=rc.dataset_augmentations,
                      default_pose=default_pose, joint_indices=joint_ids, include_base_lin_vel=True,
                      include_base_ang_vel=False, include_projected_gravity=True, device='cuda:0')
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


names = [f'pos {n}' for n in jnames] + [f'vel {n}' for n in jnames] + ['blv_x','blv_y','blv_z','g_x','g_y','g_z']
E = exp_all.reshape(exp_all.shape[0], K, D)[:, -1, :]     # last frame of each expert window
P = A
print('RESULT feature            expert mean/std        policy mean/std')
for i, n in enumerate(names):
    print(f'RESULT {n:22s} {E[:, i].mean():+.3f}/{E[:, i].std():.3f}     {P[:, i].mean().item():+.3f}/{P[:, i].std().item():.3f}')
