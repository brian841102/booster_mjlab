"""Episodic gait metrics matching the 4 s training episodes: reset every 200 steps, analyse steps 25-200 of each episode.
usage: TASK=<task> [EP_STEPS=200] [PUSH=1] gait_episodic.py <onnx> <cmd_speed> [episodes=12] [tag]"""
import os, sys, numpy as np, torch, onnxruntime as ort
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import booster_mjlab.tasks.velocity.config.nemov3
onnx_path, cmd = sys.argv[1], float(sys.argv[2]); n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 12; tag = sys.argv[4] if len(sys.argv) > 4 else 'policy'
cfg = load_env_cfg(os.environ.get('TASK', 'Mjlab-Velocity-Flat-Amp-Booster-Nemov3'), play=True); cfg.scene.num_envs = 1
tw = cfg.commands['twist']; tw.ranges.lin_vel_x = (cmd, cmd); tw.ranges.lin_vel_y = (0.0, 0.0); tw.ranges.ang_vel_z = (0.0, 0.0); tw.rel_standing_envs = 0.0; tw.rel_forward_envs = 0.0
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0'); robot = env.scene['robot']; dt = env.step_dt
ids = [robot.joint_names.index(n) for n in ('right_hip_pitch_joint', 'left_hip_pitch_joint', 'right_calf_joint', 'left_calf_joint', 'joint_rShoulderPitch', 'joint_lShoulderPitch', 'joint_rElbow', 'joint_lElbow')]
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']); nm = sess.get_inputs()[0].name
if os.environ.get('PUSH') == '1': env.command_manager.get_term('play_controls').push_enabled = True     # training-time random pushes
EP = int(os.environ.get('EP_STEPS', 200)); rows = []; falls = 0; vsp = []
for ep in range(n_ep):
    obs, _ = env.reset(); q = []; up = 1.0; v = []
    for t in range(EP):
        a = sess.run(None, {nm: obs['actor'].cpu().numpy().astype(np.float32)})[0]
        obs, *_ = env.step(torch.as_tensor(a, device='cuda:0'))
        v.append(robot.data.root_link_lin_vel_b[0, 0].item()); q.append(robot.data.joint_pos[0, ids].cpu().numpy()); up = max(up, 0) if False else min(up, -robot.data.projected_gravity_b[0, 2].item())
    q = np.array(q)[25:]; falls += int(up < 0.5); vsp.append((np.mean(v[25:]), np.max(v)))
    out = []
    for (i, j) in ((0, 1), (2, 3), (4, 5), (6, 7), (0, 4)):        # hip R/L, calf R/L, shoulder R/L, elbow R/L, right hip vs right shoulder
        r, l = q[:, i] - q[:, i].mean(), q[:, j] - q[:, j].mean(); n = len(r); w = np.hanning(n); NF = 2048; fr = np.fft.rfftfreq(NF, dt); m = (fr > 0.4) & (fr < 3.0)   # zero-padded FFT: 0.01 Hz bins
        Fr, Fl = np.fft.rfft(r * w, NF), np.fft.rfft(l * w, NF); c = (Fr[m] * np.conj(Fl[m])).sum()
        fr0 = np.fft.rfftfreq(n, dt); m0 = (fr0 > 0.4) & (fr0 < 3.0); G1, G2 = np.fft.rfft(r * w), np.fft.rfft(l * w)     # rms from the un-padded FFT (Parseval)
        rms = np.sqrt(((abs(G1[m0]) ** 2).sum() + (abs(G2[m0]) ** 2).sum()) / 2 * 2) / n / w.mean()
        out += [rms, fr[m][np.argmax(abs(Fr[m]) + abs(Fl[m]))], np.degrees(np.angle(c)), abs(c) / np.sqrt((abs(Fr[m]) ** 2).sum() * (abs(Fl[m]) ** 2).sum() + 1e-12)]
    rows.append(out)
R = np.array(rows)
def ang_med(x): return np.degrees(np.angle(np.mean(np.exp(1j * np.radians(x)))))
def ang_res(x): return abs(np.mean(np.exp(1j * np.radians(x))))          # 1 = phases all identical across episodes
print(f'EP {tag} cmd {cmd} episodes {n_ep} x {EP} steps, falls {falls}, mean speed {np.mean([a for a, b in vsp]):.2f} m/s, max speed {np.max([b for a, b in vsp]):.2f} m/s')
for k, name in enumerate(('hip R/L', 'calf R/L', 'shoulder-pitch R/L', 'elbow R/L', 'R-hip vs R-shoulder')):
    b = 4 * k; print(f'   {name}: band-rms median {np.median(R[:, b]):.3f}  peak-freq median {np.median(R[:, b+1]):.2f} Hz (IQR {np.percentile(R[:, b+1], 25):.2f}-{np.percentile(R[:, b+1], 75):.2f})  '
                     f'phase circ-mean {ang_med(R[:, b+2]):+.0f} deg, consistency {ang_res(R[:, b+2]):.2f}  coherence median {np.median(R[:, b+3]):.2f}')
env.close()
