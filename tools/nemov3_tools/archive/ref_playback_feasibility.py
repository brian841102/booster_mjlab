"""Physical feasibility of the Fast06 lift: play the reference joint trajectory back as PD targets (actions = (q_ref - default)/scale)
with the wheels rolling at the reference speed, starting from the reference root pose/velocity, and see whether the simulated robot
lifts the wheels (~8.5 cm) and stays upright.  usage: ref_playback_feasibility.py [start_frame] [wheel_sign]"""
import os, sys, pickle, numpy as np, torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import booster_mjlab.tasks.velocity.config.nemov3
from booster_mjlab.robots.nemov3.nemov3_constants import NEMOV3_JOINT_ORDER as J
start = int(sys.argv[1]) if len(sys.argv) > 1 else 380; wsign = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
d = pickle.load(open('assets/amp/nemov3_skating_straight/skatingfast06.pkl', 'rb')); q = d['dof_pos']; rp = d['root_pos']; rq = np.array([np.asarray(x) for x in d['root_rot']]); fps = d['fps']
npz = np.load('assets/amp_sources/nemov3_skating/data/npz/Robot_Model0530_V2_SkatingFast06_static_prefix_trim_transition_v1.npz', allow_pickle=True); bn = [str(x) for x in npz['body_names']]
hR = npz['body_pos_w'][:, bn.index('right_wheel_link'), 2] - 0.075; hL = npz['body_pos_w'][:, bn.index('left_wheel_link'), 2] - 0.075; g = min(hR.min(), hL.min())
cfg = load_env_cfg('Mjlab-Velocity-Flat-Amp-Clock-Arms-Booster-Nemov3', play=True); cfg.scene.num_envs = 1
tw = cfg.commands['twist']; tw.ranges.lin_vel_x = (1.0, 1.0); tw.rel_standing_envs = 0.0
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0'); robot = env.scene['robot']; jp = env.action_manager.get_term('joint_pos'); wv = env.action_manager.get_term('wheel_vel')
names = list(jp.target_names); scale = jp.scale if torch.is_tensor(jp.scale) else torch.tensor(jp.scale, device='cuda:0'); offset = jp.offset if torch.is_tensor(jp.offset) else torch.tensor(jp.offset, device='cuda:0')
idx = [J.index(n) for n in names]; wid = [robot.joint_names.index(n) for n in ('right_wheel_joint', 'left_wheel_joint')]
env.reset()
vel = np.gradient(rp, 1.0 / fps, axis=0)[start]; angv = np.zeros(3)
rot = rq[start]; wxyz = torch.tensor([rot[3], rot[0], rot[1], rot[2]], dtype=torch.float32, device='cuda:0')[None]
pos = torch.tensor([[0.0, 0.0, rp[start, 2]]], dtype=torch.float32, device='cuda:0')
robot.write_root_link_pose_to_sim(torch.cat([pos, wxyz], dim=-1)); robot.write_root_link_velocity_to_sim(torch.tensor(np.r_[vel, angv], dtype=torch.float32, device='cuda:0')[None])
jq = torch.tensor(q[start], dtype=torch.float32, device='cuda:0')[None]; robot.write_joint_state_to_sim(jq, torch.zeros(1, 32, device='cuda:0'))
speed = float(np.linalg.norm(vel[:2])); wheel_raw = wsign * speed / 0.075 / float(wv.scale if not torch.is_tensor(wv.scale) else wv.scale.mean())
print(f'RP start frame {start}, ref speed {speed:.2f} m/s, wheel action {wheel_raw:.2f}; ref wheel heights at start R {100*(hR[start]-g):.1f} L {100*(hL[start]-g):.1f} cm; ref peak in next 1.5 s R {100*(hR[start:start+90].max()-g):.1f} L {100*(hL[start:start+90].max()-g):.1f} cm')
ids = [robot.body_names.index('right_wheel_link'), robot.body_names.index('left_wheel_link')]; H = []; U = []; V = []
leg = [robot.joint_names.index(n) for n in ('right_hip_pitch_joint', 'right_calf_joint', 'left_hip_pitch_joint', 'left_calf_joint')]; LN = ('R_hipP', 'R_calf', 'L_hipP', 'L_calf'); ERR = []
rz = npz['body_pos_w'][:, bn.index('link_leg2pelvis'), 2]
for t in range(75):
    fr = min(start + int(round(t * fps / 50.0)), len(q) - 1)
    qa = torch.tensor(q[fr][idx], dtype=torch.float32, device='cuda:0'); a = torch.zeros(1, 32, device='cuda:0')
    a[0, :30] = (qa - offset.reshape(-1)) / scale.reshape(-1); a[0, 30:] = wheel_raw
    env.step(a)
    h = (robot.data.body_link_pos_w[0, ids, 2] - 0.075).cpu().numpy() * 100; H.append(h); U.append(-robot.data.projected_gravity_b[0, 2].item()); V.append(robot.data.root_link_lin_vel_b[0, 0].item())
    act = robot.data.joint_pos[0, leg].cpu().numpy(); tgt = np.array([q[fr][J.index(n)] for n in ('right_hip_pitch_joint', 'right_calf_joint', 'left_hip_pitch_joint', 'left_calf_joint')]); ERR.append(np.abs(act - tgt))
    if t % 8 == 0: print(f'      leg joints act-tgt (rad): ' + ', '.join(f'{n} {a:+.2f}/{b:+.2f}' for n, a, b in zip(LN, act, tgt)) + f' | pelvis z sim {robot.data.root_link_pos_w[0,2].item():.3f} ref {rz[fr]:.3f}')
    if t % 8 == 0: print(f'   t={t*0.02:.2f}s frame {fr}: sim h_R,h_L {h[0]:.1f},{h[1]:.1f} cm (ref {100*(hR[fr]-g):.1f},{100*(hL[fr]-g):.1f}) | upright {U[-1]:.2f} | fwd speed {V[-1]:.2f}')
ERR = np.array(ERR); print('RP mean |joint tracking error| (rad) over the first 0.5 s: ' + ', '.join(f'{n} {e:.3f}' for n, e in zip(LN, ERR[:25].mean(axis=0))))
H = np.array(H); print(f'RP result: sim max wheel height R {H[:,0].max():.1f} L {H[:,1].max():.1f} cm | min upright {min(U):.2f} (1=upright, <0.5 = fallen) | mean fwd speed {np.mean(V):.2f}')
env.close()
