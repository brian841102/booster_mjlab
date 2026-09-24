import sys, pickle, numpy as np, torch, onnxruntime as ort
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import booster_mjlab.tasks.velocity.config.nemov3
from booster_mjlab.robots.nemov3.nemov3_constants import NEMOV3_JOINT_ORDER as J

onnx_path, cmd_speed, tag = sys.argv[1], float(sys.argv[2]), sys.argv[3]
TASK = sys.argv[4] if len(sys.argv) > 4 else 'Mjlab-Velocity-Flat-Amp-Booster-Nemov3'
PAIRS = {'hip_pitch': ('right_hip_pitch_joint', 'left_hip_pitch_joint'), 'calf': ('right_calf_joint', 'left_calf_joint')}

def analyze(name, R, L, dt, extra=''):
    """R, L: time series of right/left joint (one env)."""
    out = []
    for lab, (r, l) in [('hip_pitch', (R['hip_pitch'], L['hip_pitch'])), ('calf', (R['calf'], L['calf']))]:
        rc, lc = r - r.mean(), l - l.mean()
        corr = float((rc * lc).sum() / (np.linalg.norm(rc) * np.linalg.norm(lc) + 1e-9))
        spec = np.abs(np.fft.rfft(rc)); fr = np.fft.rfftfreq(len(rc), dt); k = spec[1:].argmax() + 1
        out.append(f'{lab}: L/R corr {corr:+.2f} (−1=perfect alternation, +1=in sync)  amp(std R/L) {r.std():.3f}/{l.std():.3f}  dom.freq {fr[k]:.2f} Hz')
    print(f'{name}{extra}\n   ' + '\n   '.join(out))

# ---- reference ----
for n in ['skating01', 'skating02', 'skatingfast06']:
    d = pickle.load(open(f'assets/amp/nemov3_skating_straight/{n}.pkl', 'rb')); q = d['dof_pos']
    R = {k: q[:, J.index(v[0])] for k, v in PAIRS.items()}; L = {k: q[:, J.index(v[1])] for k, v in PAIRS.items()}
    analyze(f'REF {n}', R, L, 1.0 / d['fps'])

# ---- policy ----
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = 1
tw = cfg.commands['twist']
tw.ranges.lin_vel_x = (cmd_speed, cmd_speed); tw.ranges.lin_vel_y = (0.0, 0.0); tw.ranges.ang_vel_z = (0.0, 0.0)
tw.rel_standing_envs = 0.0; tw.rel_forward_envs = 0.0
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']); in_name = sess.get_inputs()[0].name
robot = env.scene['robot']
ids = {k: [robot.joint_names.index(v[0]), robot.joint_names.index(v[1])] for k, v in PAIRS.items()}
obs, _ = env.reset()
buf = {k: [] for k in PAIRS}; sp = []; term_cnt = 0; ph = []
clock = env.command_manager.get_term('gait_clock') if 'gait_clock' in env.command_manager.active_terms else None
def_hip = [robot.data.default_joint_pos[0, i].item() for i in ids['hip_pitch']]
for t in range(1400):
    a = sess.run(None, {in_name: obs['actor'].cpu().numpy().astype(np.float32)})[0]
    obs, rew, term, trunc, _ = env.step(torch.as_tensor(a, device='cuda:0'))
    term_cnt += int(term.sum().item())
    if t >= 200:
        for k, (ir, il) in ids.items(): buf[k].append(robot.data.joint_pos[0, [ir, il]].cpu().numpy())
        sp.append(robot.data.root_link_lin_vel_b[0, 0].item())
        if clock is not None: ph.append(clock.phase[0].item())
R = {k: np.array(v)[:, 0] for k, v in buf.items()}; L = {k: np.array(v)[:, 1] for k, v in buf.items()}
analyze(f'POLICY[{tag}] cmd={cmd_speed} m/s, measured {np.mean(sp):.2f} m/s, terminations {term_cnt}', R, L, env.step_dt)
env.close()

if clock is not None:
    ph = np.array(ph); sph = np.sin(ph)
    for lab, side_i, sign in (('R hip pitch vs +sin(phase)', 0, +1), ('L hip pitch vs -sin(phase)', 1, -1)):
        q = buf['hip_pitch']; q = np.array(q)[:, side_i]
        qc = q - q.mean(); c = float((qc * sign * sph).sum() / (np.linalg.norm(qc) * np.linalg.norm(sph - sph.mean()) + 1e-9))
        print(f'POLICY[{tag}] clock tracking {lab}: corr {c:+.2f} (1.0 = follows the gait clock exactly), amplitude (std*sqrt2) {q.std()*1.414:.3f} rad (target 0.3 at 1 m/s)')
