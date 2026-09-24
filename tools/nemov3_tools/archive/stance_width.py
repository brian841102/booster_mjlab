"""Stance width (lateral wheel separation in the base frame) and hip-roll distribution: references vs a policy.
usage: TASK=<task> stance_width.py <onnx> [episodes]   (policy rollouts at 0.6 / 1.0 / 1.3 m/s, EP_STEPS default 500)"""
import os, sys, pickle, numpy as np, torch, onnxruntime as ort
from scipy.spatial.transform import Rotation
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import booster_mjlab.tasks.velocity.config.nemov3
from booster_mjlab.robots.nemov3.nemov3_constants import NEMOV3_JOINT_ORDER as J, HOME_KEYFRAME
DEF = {'right_hip_roll_joint': HOME_KEYFRAME.joint_pos.get('right_hip_roll_joint', 0.0), 'left_hip_roll_joint': HOME_KEYFRAME.joint_pos.get('left_hip_roll_joint', 0.0)}
def stats(x): return f'mean {x.mean():+.3f} p5 {np.percentile(x,5):+.3f} p95 {np.percentile(x,95):+.3f} min {x.min():+.3f} max {x.max():+.3f}'
CL = {'skating01': 'Robot_Model0530_V2_Skating01_IK_control.npz', 'skating02': 'Robot_Model0530_V2_Skating02_IK_control.npz', 'skatingfast06': 'Robot_Model0530_V2_SkatingFast06_static_prefix_trim_transition_v1.npz'}
for n, fn in CL.items():
    p = pickle.load(open(f'assets/amp/nemov3_skating_straight/{n}.pkl', 'rb')); q = p['dof_pos']
    z = np.load(f'assets/amp_sources/nemov3_skating/data/npz/{fn}', allow_pickle=True); bn = [str(x) for x in z['body_names']]
    qq = z['body_quat_w'][:, bn.index('link_leg2pelvis')]; R = Rotation.from_quat(np.c_[qq[:, 1:], qq[:, 0]])
    root = z['body_pos_w'][:, bn.index('link_leg2pelvis')]; yr = R.inv().apply(z['body_pos_w'][:, bn.index('right_wheel_link')] - root)[:, 1]; yl = R.inv().apply(z['body_pos_w'][:, bn.index('left_wheel_link')] - root)[:, 1]
    print(f'SW REF {n}: wheel track {stats(np.abs(yl - yr))} m | hip roll R {stats(q[:, J.index("right_hip_roll_joint")])} | L {stats(q[:, J.index("left_hip_roll_joint")])}')
if len(sys.argv) > 1:
    onnx_path = sys.argv[1]; n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 3; EP = int(os.environ.get('EP_STEPS', 500))
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']); nm = sess.get_inputs()[0].name
    cfg = load_env_cfg(os.environ.get('TASK', 'Mjlab-Velocity-Flat-Amp-Clock-Arms-Booster-Nemov3'), play=True); cfg.scene.num_envs = 1
    tw = cfg.commands['twist']; tw.ranges.lin_vel_y = (0.0, 0.0); tw.ranges.ang_vel_z = (0.0, 0.0); tw.rel_standing_envs = 0.0; tw.rel_forward_envs = 0.0
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0'); robot = env.scene['robot']; term = env.command_manager.get_term('twist')
    ir, il = [robot.joint_names.index(n) for n in ('right_hip_roll_joint', 'left_hip_roll_joint')]; wr, wl = [robot.body_names.index(n) for n in ('right_wheel_link', 'left_wheel_link')]
    for c in (0.6, 1.0, 1.3):
        term.cfg.ranges.lin_vel_x = (c, c); T = []; RR = []; RL = []
        for ep in range(n_ep):
            obs, _ = env.reset()
            for t in range(EP):
                a = sess.run(None, {nm: obs['actor'].cpu().numpy().astype(np.float32)})[0]; obs, *_ = env.step(torch.as_tensor(a, device='cuda:0'))
                if t >= 100:
                    from mjlab.utils.lab_api.math import quat_apply_inverse
                    d = quat_apply_inverse(robot.data.root_link_quat_w, robot.data.body_link_pos_w[:, wr] - robot.data.body_link_pos_w[:, wl])[0]
                    T.append(abs(d[1].item())); RR.append(robot.data.joint_pos[0, ir].item()); RL.append(robot.data.joint_pos[0, il].item())
        print(f'SW POLICY cmd {c}: wheel track {stats(np.array(T))} m | hip roll R {stats(np.array(RR))} | L {stats(np.array(RL))}', flush=True)
    env.close()
