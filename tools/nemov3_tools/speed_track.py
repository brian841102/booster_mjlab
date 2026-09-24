"""Forward-speed tracking across models/commands (clean, deterministic, same env for every model).
usage: TASK=<task> speed_track.py <episodes> <steps> <cmds comma-sep> tag=onnx [tag=onnx ...]
Drops the first 100 steps (2 s) of each episode (start-up transient); reports mean speed, bias and RMSE of (v - cmd)."""
import os, sys, numpy as np, torch, onnxruntime as ort
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import booster_mjlab.tasks.velocity.config.nemov3
n_ep, steps = int(sys.argv[1]), int(sys.argv[2]); cmds = [float(x) for x in sys.argv[3].split(',')]; models = dict(a.split('=', 1) for a in sys.argv[4:])
cfg = load_env_cfg(os.environ.get('TASK', 'Mjlab-Velocity-Flat-Amp-Clock-Arms-Booster-Nemov3'), play=True); cfg.scene.num_envs = 1
tw = cfg.commands['twist']; tw.ranges.lin_vel_x = (1.0, 1.0); tw.ranges.lin_vel_y = (0.0, 0.0); tw.ranges.ang_vel_z = (0.0, 0.0); tw.rel_standing_envs = 0.0; tw.rel_forward_envs = 0.0
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0'); robot = env.scene['robot']; term = env.command_manager.get_term('twist')
print('ST model   ' + ' '.join(f'| cmd {c:.1f}: mean  bias  rmse ' for c in cmds) + '| avg|bias|  avg rmse')
for tag, path in models.items():
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider']); nm = sess.get_inputs()[0].name
    row = []; b_all = []; r_all = []
    for c in cmds:
        term.cfg.ranges.lin_vel_x = (c, c); vs = []
        for ep in range(n_ep):
            obs, _ = env.reset(); v = []
            for t in range(steps):
                a = sess.run(None, {nm: obs['actor'].cpu().numpy().astype(np.float32)})[0]
                obs, *_ = env.step(torch.as_tensor(a, device='cuda:0')); v.append(robot.data.root_link_lin_vel_b[0, 0].item())
            vs += v[100:]
        vs = np.array(vs); bias = vs.mean() - c; rmse = np.sqrt(((vs - c) ** 2).mean()); b_all.append(abs(bias)); r_all.append(rmse)
        row.append(f'| {vs.mean():5.2f} {bias:+5.2f} {rmse:5.2f}     ')
    print(f'ST {tag:14s}' + ' '.join(row) + f'| {np.mean(b_all):.3f}     {np.mean(r_all):.3f}', flush=True)
env.close()
