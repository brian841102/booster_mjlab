"""Band-limited (0.3-3 Hz) gait metrics on a deterministic rollout. usage: gait_band.py <onnx> <cmd_speed> [steps] [tag]"""
import sys, pickle, numpy as np, torch, onnxruntime as ort
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import booster_mjlab.tasks.velocity.config.nemov3
from booster_mjlab.robots.nemov3.nemov3_constants import NEMOV3_JOINT_ORDER as J
NAMES = {'hip': ('right_hip_pitch_joint', 'left_hip_pitch_joint'), 'calf': ('right_calf_joint', 'left_calf_joint')}
def band(R, L, dt, tag):
    out = []
    for k in NAMES:
        r, l = R[k] - R[k].mean(), L[k] - L[k].mean(); n = len(r); w = np.hanning(n)
        fr = np.fft.rfftfreq(n, dt); m = (fr > 0.3) & (fr < 3.0)
        Fr, Fl = np.fft.rfft(r * w), np.fft.rfft(l * w)
        ar, al = np.sqrt((abs(Fr[m])**2).sum() * 2) / n / w.mean(), np.sqrt((abs(Fl[m])**2).sum() * 2) / n / w.mean()   # band rms
        pk = fr[m][np.argmax(abs(Fr[m]) + abs(Fl[m]))]
        c = (Fr[m] * np.conj(Fl[m])).sum(); coh = abs(c) / np.sqrt((abs(Fr[m])**2).sum() * (abs(Fl[m])**2).sum() + 1e-12)
        out.append(f'{k}: band-rms R/L {ar:.3f}/{al:.3f}  peak {pk:.2f} Hz  R-L phase {np.degrees(np.angle(c)):+.0f} deg (180=anti-phase)  coherence {coh:.2f}')
    print(f'GB {tag}\n   ' + '\n   '.join(out))
if __name__ == '__main__':
    onnx_path, cmd_speed = sys.argv[1], float(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000; tag = sys.argv[4] if len(sys.argv) > 4 else 'policy'
    for n in ['skating01', 'skating02', 'skatingfast06']:
        d = pickle.load(open(f'assets/amp/nemov3_skating_straight/{n}.pkl', 'rb')); q = d['dof_pos']
        band({k: q[:, J.index(v[0])] for k, v in NAMES.items()}, {k: q[:, J.index(v[1])] for k, v in NAMES.items()}, 1.0 / d['fps'], 'REF ' + n)
    cfg = load_env_cfg(__import__('os').environ.get('TASK', 'Mjlab-Velocity-Flat-Amp-Booster-Nemov3'), play=True); cfg.scene.num_envs = 1
    tw = cfg.commands['twist']; tw.ranges.lin_vel_x = (cmd_speed, cmd_speed); tw.ranges.lin_vel_y = (0.0, 0.0); tw.ranges.ang_vel_z = (0.0, 0.0); tw.rel_standing_envs = 0.0; tw.rel_forward_envs = 0.0
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0'); robot = env.scene['robot']
    ids = {k: [robot.joint_names.index(v[0]), robot.joint_names.index(v[1])] for k, v in NAMES.items()}
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']); nm = sess.get_inputs()[0].name
    obs, _ = env.reset(); B = {k: [] for k in NAMES}; sp = []; up = []
    for t in range(steps):
        a = sess.run(None, {nm: obs['actor'].cpu().numpy().astype(np.float32)})[0]
        obs, *_ = env.step(torch.as_tensor(a, device='cuda:0'))
        if t >= 100:
            for k, (ir, il) in ids.items(): B[k].append(robot.data.joint_pos[0, [ir, il]].cpu().numpy())
            sp.append(robot.data.root_link_lin_vel_b[0, 0].item()); up.append(robot.data.projected_gravity_b[0, 2].item())
    band({k: np.array(v)[:, 0] for k, v in B.items()}, {k: np.array(v)[:, 1] for k, v in B.items()}, env.step_dt, f'POLICY[{tag}] cmd {cmd_speed} m/s, measured {np.mean(sp):.2f}, gravity_z min {np.min(up):.2f} (-1=upright)')
    env.close()
