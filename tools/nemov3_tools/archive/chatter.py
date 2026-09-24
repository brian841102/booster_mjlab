import sys, numpy as np, torch, onnxruntime as ort
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import booster_mjlab.tasks.velocity.config.nemov3
onnx_path, ckpt, noise = sys.argv[1], sys.argv[2], float(sys.argv[3])
cfg = load_env_cfg('Mjlab-Velocity-Flat-Amp-Booster-Nemov3', play=True); cfg.scene.num_envs = 1
tw = cfg.commands['twist']; tw.ranges.lin_vel_x = (1.0, 1.0); tw.ranges.lin_vel_y = (0.0, 0.0); tw.ranges.ang_vel_z = (0.0, 0.0); tw.rel_standing_envs = 0.0
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0'); robot = env.scene['robot']
ids = [robot.joint_names.index(n) for n in ('right_hip_pitch_joint', 'left_hip_pitch_joint', 'right_calf_joint', 'left_calf_joint')]
std = torch.load(ckpt, map_location='cpu', weights_only=False)['actor_state_dict']['distribution.std_param'].numpy()
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']); nm = sess.get_inputs()[0].name
obs, _ = env.reset(); Q, V, ACT = [], [], []
for t in range(400):
    a = sess.run(None, {nm: obs['actor'].cpu().numpy().astype(np.float32)})[0]
    a = a + noise * std[None, :] * np.random.randn(*a.shape).astype(np.float32)
    obs, *_ = env.step(torch.as_tensor(a, device='cuda:0'))
    if t >= 100: Q.append(robot.data.joint_pos[0, ids].cpu().numpy()); V.append(robot.data.joint_vel[0, ids].cpu().numpy()); ACT.append(a[0][[22, 26, 25, 29]])
Q, V, ACT = map(np.array, (Q, V, ACT)); dt = env.step_dt
for i, n in enumerate(['R_hip', 'L_hip', 'R_calf', 'L_calf']):
    v = V[:, i]; fv = np.abs(np.fft.rfft(v - v.mean())); fr = np.fft.rfftfreq(len(v), dt)
    hi = np.sqrt((fv[fr > 5] ** 2).sum() * 2) / len(v); lo = np.sqrt((fv[fr <= 5] ** 2).sum() * 2) / len(v)
    da = np.diff(ACT[:, i]); print(f'CH {n}: pos std {Q[:, i].std():.3f}  vel std {v.std():.2f}  vel>5Hz {hi:.2f} vel<5Hz {lo:.2f}  action diff std {da.std():.3f}  act std {ACT[:, i].std():.3f}')
env.close()
