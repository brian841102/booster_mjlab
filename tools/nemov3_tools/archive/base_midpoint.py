"""Offset of the base (root link) from the midpoint of the two wheel centers, in the heading (yaw-only) frame.
Reference clips (from the source npz world poses) and, optionally, a policy at several speeds.
usage: TASK=<task> [EP_STEPS=500] base_midpoint.py [onnx] [episodes] [cmds comma-sep]"""
import os, sys, numpy as np
from scipy.spatial.transform import Rotation
CL = {'skating01': 'Robot_Model0530_V2_Skating01_IK_control.npz', 'skating02': 'Robot_Model0530_V2_Skating02_IK_control.npz',
      'skatingfast06': 'Robot_Model0530_V2_SkatingFast06_static_prefix_trim_transition_v1.npz'}
ROOT = os.environ.get('ROOT_BODY', 'link_pelvisYaw')
def fmt(dx, dy, dist, tag):
    print(f'BM {tag}: dx (base ahead of wheel midpoint) mean {dx.mean()*100:+.1f} cm std {dx.std()*100:.1f} | dy (base left of midpoint) mean {dy.mean()*100:+.1f} cm std {dy.std()*100:.1f} | '
          f'|d| mean {dist.mean()*100:.1f} cm p95 {np.percentile(dist,95)*100:.1f} | reward exp(-d^2/0.05^2) mean {np.exp(-dist**2/0.05**2).mean():.2f}', flush=True)
for n, fn in CL.items():
    z = np.load(f'assets/amp_sources/nemov3_skating/data/npz/{fn}', allow_pickle=True); bn = [str(x) for x in z['body_names']]
    q = z['body_quat_w'][:, bn.index(ROOT)]; yaw = Rotation.from_quat(np.c_[q[:, 1:], q[:, 0]]).as_euler('ZYX')[:, 0]
    root = z['body_pos_w'][:, bn.index(ROOT)]; mid = 0.5 * (z['body_pos_w'][:, bn.index('right_wheel_link')] + z['body_pos_w'][:, bn.index('left_wheel_link')])
    d = root[:, :2] - mid[:, :2]; c, s = np.cos(-yaw), np.sin(-yaw); dx = c * d[:, 0] - s * d[:, 1]; dy = s * d[:, 0] + c * d[:, 1]
    fmt(dx, dy, np.hypot(dx, dy), f'REF {n} (root body {ROOT})')
if len(sys.argv) > 1:
    import torch, onnxruntime as ort
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
    import booster_mjlab.tasks.velocity.config.nemov3
    onnx_path = sys.argv[1]; n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 3; cmds = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else '0.6,1.0,1.3').split(',')]; EP = int(os.environ.get('EP_STEPS', 500))
    cfg = load_env_cfg(os.environ.get('TASK', 'Mjlab-Velocity-Flat-Amp-Clock-Arms-Booster-Nemov3'), play=True); cfg.scene.num_envs = 1
    tw = cfg.commands['twist']; tw.ranges.lin_vel_y = (0.0, 0.0); tw.ranges.ang_vel_z = (0.0, 0.0); tw.rel_standing_envs = 0.0; tw.rel_forward_envs = 0.0
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0'); robot = env.scene['robot']; term = env.command_manager.get_term('twist')
    print(f'BM sim root body = {robot.body_names[0]}')
    w = [robot.body_names.index('right_wheel_link'), robot.body_names.index('left_wheel_link')]
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']); nm = sess.get_inputs()[0].name
    for c in cmds:
        term.cfg.ranges.lin_vel_x = (c, c); DX, DY = [], []
        for ep in range(n_ep):
            obs, _ = env.reset()
            for t in range(EP):
                a = sess.run(None, {nm: obs['actor'].cpu().numpy().astype(np.float32)})[0]; obs, *_ = env.step(torch.as_tensor(a, device='cuda:0'))
                if t >= 100:
                    mid = robot.data.body_link_pos_w[:, w].mean(dim=1); d = robot.data.root_link_pos_w - mid; d[:, 2] = 0
                    db = quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w), d)[0]; DX.append(db[0].item()); DY.append(db[1].item())
        DX, DY = np.array(DX), np.array(DY); fmt(DX, DY, np.hypot(DX, DY), f'POLICY cmd {c}')
    env.close()
