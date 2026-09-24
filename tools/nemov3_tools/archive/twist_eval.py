"""Pelvis-yaw / hip-roll / hip-yaw motion in policy rollouts (0.4-3 Hz band rms + total std) vs Fast06 reference std (0.124 / 0.073 / 0.11 rad).
usage: TASK=<task> [EP_STEPS=500] twist_eval.py <onnx> <cmd_speed> [episodes] [tag]"""
import os, sys, pickle, numpy as np, torch, onnxruntime as ort
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import booster_mjlab.tasks.velocity.config.nemov3
from booster_mjlab.robots.nemov3.nemov3_constants import NEMOV3_JOINT_ORDER as J
onnx_path, cmd = sys.argv[1], float(sys.argv[2]); n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 4; tag = sys.argv[4] if len(sys.argv) > 4 else 'policy'; EP = int(os.environ.get('EP_STEPS', 500))
NAMES = ['joint_pelvisYaw', 'right_hip_roll_joint', 'left_hip_roll_joint', 'right_hip_yaw_joint', 'left_hip_yaw_joint']
d = pickle.load(open('assets/amp/nemov3_skating_straight/skatingfast06.pkl', 'rb')); q = d['dof_pos']
print('TW REF fast06 std (rad): ' + ', '.join(f"{n.replace('_joint','')} {q[:, J.index(n)].std():.3f}" for n in NAMES))
cfg = load_env_cfg(os.environ.get('TASK', 'Mjlab-Velocity-Flat-Amp-Clock-Arms-Booster-Nemov3'), play=True); cfg.scene.num_envs = 1
tw = cfg.commands['twist']; tw.ranges.lin_vel_x = (cmd, cmd); tw.ranges.lin_vel_y = (0.0, 0.0); tw.ranges.ang_vel_z = (0.0, 0.0); tw.rel_standing_envs = 0.0; tw.rel_forward_envs = 0.0
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0'); robot = env.scene['robot']; dt = env.step_dt
ids = [robot.joint_names.index(n) for n in NAMES]; sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']); nm = sess.get_inputs()[0].name
S = []; B = []
for ep in range(n_ep):
    obs, _ = env.reset(); Q = []
    for t in range(EP):
        a = sess.run(None, {nm: obs['actor'].cpu().numpy().astype(np.float32)})[0]; obs, *_ = env.step(torch.as_tensor(a, device='cuda:0'))
        if t >= 100: Q.append(robot.data.joint_pos[0, ids].cpu().numpy())
    Q = np.array(Q); S.append(Q.std(axis=0)); n = len(Q); w = np.hanning(n); fr = np.fft.rfftfreq(n, dt); m = (fr > 0.4) & (fr < 3.0)
    B.append([np.sqrt((abs(np.fft.rfft((Q[:, i] - Q[:, i].mean()) * w))[m] ** 2).sum() * 2) / n / w.mean() for i in range(len(NAMES))])
S = np.mean(S, axis=0); B = np.mean(B, axis=0)
print(f'TW {tag} cmd {cmd}: total std (rad) ' + ', '.join(f"{n.replace('_joint','')} {s:.3f}" for n, s in zip(NAMES, S)) + ' | band rms ' + ', '.join(f'{b:.3f}' for b in B))
env.close()
