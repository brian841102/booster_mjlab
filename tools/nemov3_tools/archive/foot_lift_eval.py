"""Wheel lift height in policy rollouts vs the Fast06 reference (8.5 cm peak, ~0.55 s per lift, ~40% of time above 2 cm).
usage: TASK=<task> [EP_STEPS=500] foot_lift_eval.py <onnx> <cmd_speed> [episodes] [tag]
h = wheel-center z - 0.075 (wheel bottom height); a lift event = contiguous segment with h > 2 cm."""
import os, sys, numpy as np, torch, onnxruntime as ort
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import booster_mjlab.tasks.velocity.config.nemov3
onnx_path, cmd = sys.argv[1], float(sys.argv[2]); n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 6; tag = sys.argv[4] if len(sys.argv) > 4 else 'policy'
EP = int(os.environ.get('EP_STEPS', 500))
cfg = load_env_cfg(os.environ.get('TASK', 'Mjlab-Velocity-Flat-Amp-Clock-Arms-Booster-Nemov3'), play=True); cfg.scene.num_envs = 1
tw = cfg.commands['twist']; tw.ranges.lin_vel_x = (cmd, cmd); tw.ranges.lin_vel_y = (0.0, 0.0); tw.ranges.ang_vel_z = (0.0, 0.0); tw.rel_standing_envs = 0.0; tw.rel_forward_envs = 0.0
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0'); robot = env.scene['robot']; sensor = env.scene['feet_ground_contact']; dt = env.step_dt
ids = [robot.body_names.index('right_wheel_link'), robot.body_names.index('left_wheel_link')]
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']); nm = sess.get_inputs()[0].name
peaks, durs, frac2, frac5, air_frac, speeds = [], [], [], [], [], []
for ep in range(n_ep):
    obs, _ = env.reset(); H = []; A = []; V = []
    for t in range(EP):
        a = sess.run(None, {nm: obs['actor'].cpu().numpy().astype(np.float32)})[0]
        obs, *_ = env.step(torch.as_tensor(a, device='cuda:0'))
        if t >= 100:
            H.append((robot.data.body_link_pos_w[0, ids, 2] - 0.075).clamp_min(0).cpu().numpy()); A.append((sensor.data.found[0] == 0).cpu().numpy()); V.append(robot.data.root_link_lin_vel_b[0, 0].item())
    H = np.array(H); A = np.array(A); speeds.append(np.mean(V))
    frac2.append((H > 0.02).mean()); frac5.append((H > 0.05).mean()); air_frac.append(A.mean())
    for f in range(2):
        above = H[:, f] > 0.02; idx = np.flatnonzero(np.diff(above.astype(int)) != 0) + 1
        for sg in np.split(np.arange(len(above)), idx):
            if above[sg[0]] and len(sg) > 2: peaks.append(H[sg, f].max() * 100); durs.append(len(sg) * dt)
pk = np.array(peaks) if peaks else np.array([0.0])
print(f'FL {tag} cmd {cmd} eps {n_ep}x{EP}: mean speed {np.mean(speeds):.2f} | lift events {len(peaks)} ({len(peaks)/(n_ep*2*(EP-100)*dt):.2f}/s per foot) | peak height cm: mean {pk.mean():.1f} p50 {np.percentile(pk,50):.1f} p90 {np.percentile(pk,90):.1f} max {pk.max():.1f} | duration/event {np.mean(durs) if durs else 0:.2f} s | time above 2 cm {np.mean(frac2)*100:.0f}%  above 5 cm {np.mean(frac5)*100:.0f}% | contact-lost {np.mean(air_frac)*100:.0f}%   [Fast06 ref: peak 8.5 cm, 0.55 s/event, above 2 cm ~40%]')
env.close()
