"""Measure wheel (foot) ground-site height over each reference clip: how high does the rear foot lift?
Uses the source npz world-frame body poses (assets/amp_sources/nemov3_skating/data/npz) and the same
wheel-ground site as the simulator (local (0,0,-0.075) in the wheel link frame)."""
import glob, numpy as np
from scipy.spatial.transform import Rotation
CLIPS = {'skating01': 'Robot_Model0530_V2_Skating01_IK_control.npz', 'skating02': 'Robot_Model0530_V2_Skating02_IK_control.npz',
         'skatingfast06': 'Robot_Model0530_V2_SkatingFast06_static_prefix_trim_transition_v1.npz'}
SITE_LOCAL = np.array([0.0, 0.0, -0.075])
out = {}
for name, fn in CLIPS.items():
    d = np.load(f'assets/amp_sources/nemov3_skating/data/npz/{fn}', allow_pickle=True); bn = [str(x) for x in d['body_names']]; fps = float(d['fps'][0])
    site = {}
    for side in ('right', 'left'):
        i = bn.index(f'{side}_wheel_link'); q = d['body_quat_w'][:, i]        # wxyz
        R = Rotation.from_quat(np.c_[q[:, 1:], q[:, 0]]); site[side] = d['body_pos_w'][:, i] + R.apply(SITE_LOCAL)
    zr, zl = site['right'][:, 2], site['left'][:, 2]; ground = min(zr.min(), zl.min())
    root = d['body_pos_w'][:, bn.index('link_leg2pelvis')]; T = len(zr)
    vx = np.gradient(root[:, 0], 1.0 / fps)
    print(f'MEAS {name}: frames {T}, fps {fps:.0f}, duration {T/fps:.2f} s, root z {root[:,2].mean():.3f}, mean fwd speed {vx.mean():.2f} m/s')
    print(f'   ground reference (clip-min of both wheel sites) = {ground:.4f} m')
    for side, z in (('right', zr), ('left', zl)):
        h = z - ground
        print(f'   {side:5s} foot height above ground: min {h.min()*100:5.1f} | p50 {np.percentile(h,50)*100:5.1f} | p90 {np.percentile(h,90)*100:5.1f} | p95 {np.percentile(h,95)*100:5.1f} | max {h.max()*100:5.1f} cm ; time above 1 cm {np.mean(h>0.01)*100:4.0f}%, above 2 cm {np.mean(h>0.02)*100:4.0f}%, above 5 cm {np.mean(h>0.05)*100:4.0f}%')
    hd = np.maximum(zr, zl) - np.minimum(zr, zl)
    print(f'   |R-L| height difference: p50 {np.percentile(hd,50)*100:.1f} | p95 {np.percentile(hd,95)*100:.1f} | max {hd.max()*100:.1f} cm')
    # per-foot peak lift per swing: local maxima of height
    for side, z in (('right', zr), ('left', zl)):
        h = z - ground; pk = [i for i in range(1, T - 1) if h[i] > h[i-1] and h[i] >= h[i+1] and h[i] > 0.02]
        print(f'   {side:5s} lift peaks (>2 cm): {len(pk)} peaks, heights cm {[round(h[i]*100, 1) for i in pk][:12]}')
    out[name] = (zr - ground, zl - ground, fps)
np.savez('/tmp/foot_lift_series.npz', **{f'{k}_{s}': v[i] for k, v in out.items() for i, s in enumerate(('r', 'l'))})


# ---- Wheel-center based foot height (physically the wheel bottom = center z - wheel radius, independent of foot tilt) ----
print("MEAS ---- wheel-center based foot height h = z_center - 0.075 (what the reward uses) ----")
R_WHEEL = 0.075
for name, fn in CLIPS.items():
    d = np.load(f'assets/amp_sources/nemov3_skating/data/npz/{fn}', allow_pickle=True); bn = [str(x) for x in d['body_names']]; fps = float(d['fps'][0])
    h = {s: d['body_pos_w'][:, bn.index(f'{s}_wheel_link'), 2] - R_WHEEL for s in ('right', 'left')}
    g = min(h['right'].min(), h['left'].min()); root = d['body_pos_w'][:, bn.index('link_leg2pelvis')]
    vx = np.convolve(np.gradient(root[:, 0], 1.0 / fps), np.ones(30) / 30, mode='same'); T = len(root)
    print(f'MEAS {name}: lift above clip-min ground: max R {(h["right"].max()-g)*100:.1f} cm, L {(h["left"].max()-g)*100:.1f} cm')
    for side in ('right', 'left'):
        hh = h[side] - g; other = 'left' if side == 'right' else 'right'
        above = hh > 0.02; idx = np.flatnonzero(np.diff(above.astype(int)) != 0) + 1; segs = [sg for sg in np.split(np.arange(T), idx) if above[sg[0]] and len(sg) > 2]
        if not segs: print(f'   {side}: no lift events above 2 cm'); continue
        rows = [(hh[sg].max() * 100, len(sg) / fps, vx[sg[np.argmax(hh[sg])]]) for sg in segs]; a = np.array(rows)
        print(f'   {side}: {len(a)} lift events (>2 cm); peak height cm mean {a[:,0].mean():.1f} [{a[:,0].min():.1f}, {a[:,0].max():.1f}]; time above 2 cm per event {a[:,1].mean():.2f} s; speed at peak {a[:,2].mean():.2f} m/s; fraction of time above 2 cm {above.mean()*100:.0f}%')
