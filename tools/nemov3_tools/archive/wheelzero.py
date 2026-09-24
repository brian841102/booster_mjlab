import sys, os, numpy as np, torch, onnxruntime as ort
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import booster_mjlab.tasks.velocity.config.nemov3
onnx_path, zero = sys.argv[1], sys.argv[2] == '1'; steps = int(sys.argv[3])
cfg = load_env_cfg(os.environ['TASK'], play=True); cfg.scene.num_envs = 1
tw = cfg.commands['twist']; tw.ranges.lin_vel_x = (1.0, 1.0); tw.ranges.lin_vel_y = (0.0, 0.0); tw.ranges.ang_vel_z = (0.0, 0.0); tw.rel_standing_envs = 0.0; tw.rel_forward_envs = 0.0
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0'); robot = env.scene['robot']
om = env.observation_manager; names = om._group_obs_term_names['actor']; dims = om._group_obs_term_dim['actor']
print('OBS', list(zip(names, dims)))
off = 0; sl = {}
for n, d in zip(names, dims): sl[n] = (off, off + int(np.prod(d))); off += int(np.prod(d))
jp0, jp1 = sl['joint_pos']; wheel_ids = [robot.joint_names.index('left_wheel_joint'), robot.joint_names.index('right_wheel_joint')]
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']); nm = sess.get_inputs()[0].name
ids = [robot.joint_names.index(n) for n in ('right_hip_pitch_joint', 'left_hip_pitch_joint', 'right_calf_joint', 'left_calf_joint')]
obs, _ = env.reset(); rows = []
for t in range(steps):
    o = obs['actor'].cpu().numpy().astype(np.float32).copy()
    if zero:
        for w in wheel_ids: o[0, jp0 + w] = 0.0
    a = sess.run(None, {nm: o})[0]
    obs, *_ = env.step(torch.as_tensor(a, device='cuda:0'))
    rows.append(np.r_[robot.data.joint_pos[0, ids].cpu().numpy(), robot.data.root_link_lin_vel_b[0, 0].item(), robot.data.joint_pos[0, wheel_ids[0]].item()])
X = np.array(rows); dt = env.step_dt
print('t(s) R_hip L_hip R_calf L_calf vx wheelangle')
for t in range(0, steps, 75): print(f'WZ {t*dt:5.1f} ' + ' '.join(f'{v:+7.2f}' for v in X[t]))
for a, b in ((100, 500), (500, 900), (900, min(1300, steps))):
    if b > a: x = X[a:b, 0]; y = X[a:b, 1]; print(f'WZ window {a*dt:.0f}-{b*dt:.0f}s: R_hip std {x.std():.3f} L_hip std {y.std():.3f} speed mean {X[a:b,4].mean():.2f} std {X[a:b,4].std():.2f}')
env.close()
