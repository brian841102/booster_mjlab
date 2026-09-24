import sys, numpy as np, torch, onnxruntime as ort
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import booster_mjlab.tasks.velocity.config.nemov3
onnx_path, cmd_speed = sys.argv[1], float(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 600
cfg = load_env_cfg(__import__('os').environ.get('TASK', 'Mjlab-Velocity-Flat-Amp-Booster-Nemov3'), play=True); cfg.scene.num_envs = 1
tw = cfg.commands['twist']; tw.ranges.lin_vel_x = (cmd_speed, cmd_speed); tw.ranges.lin_vel_y = (0.0, 0.0); tw.ranges.ang_vel_z = (0.0, 0.0)
tw.rel_standing_envs = 0.0; tw.rel_forward_envs = 0.0
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']); nm = sess.get_inputs()[0].name
robot = env.scene['robot']
names = ['right_hip_pitch_joint', 'left_hip_pitch_joint', 'right_calf_joint', 'left_calf_joint']
ids = [robot.joint_names.index(n) for n in names]
obs, _ = env.reset(); rows = []
for t in range(steps):
    a = sess.run(None, {nm: obs['actor'].cpu().numpy().astype(np.float32)})[0]
    obs, *_ = env.step(torch.as_tensor(a, device='cuda:0'))
    rows.append(np.r_[robot.data.joint_pos[0, ids].cpu().numpy(), robot.data.root_link_lin_vel_b[0, 0].item(), a[0][[robot.joint_names.index(n) if False else 0]] if False else 0])
X = np.array(rows)[:, :5]; dt = env.step_dt
np.save(sys.argv[4] if len(sys.argv) > 4 else '/tmp/leg_series.npy', X)
print('t(s)   R_hip  L_hip  R_calf L_calf  vx')
for t in range(0, steps, 10): print(f'{t*dt:5.2f} ' + ' '.join(f'{v:+6.2f}' for v in X[t]))
# band split: slow (<0.3Hz) vs fast (>0.3Hz)
for i, n in enumerate(names):
    x = X[100:, i]; sp = np.fft.rfft(x - x.mean()); fr = np.fft.rfftfreq(len(x), dt)
    fast = np.sqrt((np.abs(sp[fr > 0.3])**2).sum() * 2) / len(x); slow = np.sqrt((np.abs(sp[(fr <= 0.3) & (fr > 0)])**2).sum() * 2) / len(x)
    print(f'{n}: slow(<0.3Hz) std {slow:.3f}  fast(>0.3Hz) std {fast:.3f}')
env.close()
