#!/usr/bin/env python3
"""Analyse a wheel bench test (see README.md): what the wheel really does for a commanded velocity.

usage: uv run --with numpy python wheel_bench_analyze.py <left.jsonl>        (output of wheel_bench_probe.py --stage full)
       uv run --with rosbags --with numpy python wheel_bench_analyze.py <rosbag dir> [--wheel left|right] [--t0 SECONDS]
       uv run --with numpy python wheel_bench_analyze.py --selftest

A .jsonl file is read directly (the test starts at its "test_start" marker). A rosbag directory is read with
/wheel_commands and /robot_joint_states (the test starts at the first /wheel_commands message; --t0 overrides, in seconds
since the first message). Reports, for the driven wheel:
  hold    velocity noise / creep of the stopped wheel and the static effort
  ramp    command at which the wheel starts (deadband) on the way up and at which it stops on the way down
  steps   dead time, 10-90% rise time, overshoot and steady gain for each +-1/2/4 rad/s step
  chirp   delay (ms) and gain between command and velocity around 1, 2, 4, 6, 8, 10 Hz
and writes wheel_bench_result.json next to the bag, to calibrate the simulated wheel (deadband, delay, first-order time
constant, gain) and to compare with the 5.8 Hz standing jitter measured on the hardware.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wheel_bench_profile as P  # noqa: E402

FS = 1000.0  # analysis grid (Hz)
MOVING = 0.2  # rad/s: "the wheel is turning"


def grid(t_end):
    return np.arange(0.0, t_end, 1.0 / FS)


def zoh(t, t_cmd, cmd):
    """Zero-order hold of the command messages on the analysis grid."""
    idx = np.clip(np.searchsorted(t_cmd, t, side="right") - 1, 0, len(t_cmd) - 1)
    return np.where(t >= t_cmd[0], cmd[idx], 0.0)


def analyse(t_cmd, cmd, t_fb, vel, effort=None):
    """All times in seconds since the start of the test. Returns a dict of results."""
    t = grid(min(P.DURATION, t_fb[-1]))
    c = zoh(t, t_cmd, cmd)
    v = np.interp(t, t_fb, vel)
    e = np.interp(t, t_fb, effort) if effort is not None and len(effort) == len(t_fb) else None
    out = {}

    m = (t >= P.HOLD[0] + 1.0) & (t < P.HOLD[1])
    out["hold"] = {"vel_mean": float(v[m].mean()), "vel_std": float(v[m].std()),
                   "effort_mean": None if e is None else float(e[m].mean()),
                   "effort_std": None if e is None else float(e[m].std())}

    ramp = {}
    for name, (a, b), sign in (("pos", P.RAMP_POS, 1.0), ("neg", P.RAMP_NEG, -1.0)):
        mid = 0.5 * (a + b)
        up = (t >= a) & (t < mid)
        down = (t >= mid) & (t < b)
        moving_up = up & (sign * v > MOVING)
        moving_down = down & (sign * v > MOVING)
        start_cmd = float(abs(c[moving_up][0])) if moving_up.any() else None
        stop_cmd = float(abs(c[moving_down][-1])) if moving_down.any() else None
        ramp[name] = {"start_cmd": start_cmd, "stop_cmd": stop_cmd}
    out["ramp"] = ramp

    steps = []
    for level, t_on, t_off in P.step_windows():
        pre = (t >= t_on - 0.8) & (t < t_on - 0.1)
        base = v[pre].mean()
        fin = v[(t >= t_off - 0.7) & (t < t_off - 0.05)].mean() - base
        seg = (t >= t_on - 0.1) & (t < t_off)
        d = (v[seg] - base) / fin if abs(fin) > 1e-3 else np.zeros(seg.sum())
        d = np.convolve(d, np.ones(20) / 20, mode="same")  # 20 ms average: noise must not trigger the crossing
        ts = t[seg]

        def cross(frac):
            above = d >= frac
            ok = [k for k in range(len(above) - 20) if above[k:k + 20].all()]  # stays above for 20 ms
            return float(ts[ok[0]] - t_on) if ok else None

        t10, t90 = cross(0.1), cross(0.9)
        steps.append({"level": level, "gain": float(fin / level), "dead_time_s": t10,
                      "rise_10_90_s": None if t10 is None or t90 is None else t90 - t10,
                      "overshoot": float(d[10:-10].max() - 1.0) if d.size > 20 else None})
    out["steps"] = steps

    chirp = {}
    for f in (1.0, 2.0, 4.0, 6.0, 8.0, 10.0):
        tc = P.CHIRP[0] + np.log(f / P.CHIRP_F0) * P.CHIRP_SWEEP / np.log(P.CHIRP_F1 / P.CHIRP_F0)
        half = max(1.0, 4.0 / f) / 2
        w = (t >= tc - half) & (t < tc + half)
        cc, vv = c[w] - c[w].mean(), v[w] - v[w].mean()
        best, lag = -2.0, 0
        for L in range(0, 151):  # velocity lags the command by 0..150 ms
            r = np.dot(cc[: len(cc) - L], vv[L:]) / (np.linalg.norm(cc[: len(cc) - L]) * np.linalg.norm(vv[L:]) + 1e-12)
            if r > best:
                best, lag = r, L
        gain = float(np.dot(cc[: len(cc) - lag], vv[lag:]) / (np.dot(cc[: len(cc) - lag], cc[: len(cc) - lag]) + 1e-12))
        chirp[f"{f:g}Hz"] = {"delay_ms": lag, "corr": float(best), "gain": gain}
    out["chirp"] = chirp
    return out


def report(res, label):
    h = res["hold"]
    print(f"== {label}")
    print(f"hold: vel mean {h['vel_mean']:+.3f} std {h['vel_std']:.3f} rad/s; effort mean {h['effort_mean']} std {h['effort_std']}")
    print(f"ramp: start cmd + {res['ramp']['pos']['start_cmd']} / - {res['ramp']['neg']['start_cmd']} rad/s; "
          f"stop cmd + {res['ramp']['pos']['stop_cmd']} / - {res['ramp']['neg']['stop_cmd']} rad/s")
    print("steps:  level  gain  dead time  rise 10-90  overshoot")
    for s in res["steps"]:
        f = lambda x, n=3: "  n/a" if x is None else f"{x:{n + 3}.{n}f}"
        print(f"      {s['level']:+5.1f} {s['gain']:6.2f} {f(s['dead_time_s'])} s {f(s['rise_10_90_s'])} s {f(s['overshoot'], 2)}")
    print("chirp:  freq  delay(ms)  gain  corr")
    for k, c in res["chirp"].items():
        print(f"      {k:>6s} {c['delay_ms']:8d} {c['gain']:6.2f} {c['corr']:5.2f}")


def read_bag(path, wheel, t0_override):
    from rosbags.highlevel import AnyReader
    cmd_rows, fb_rows, t_first = [], [], None
    with AnyReader([Path(path)]) as r:
        for c, t, raw in r.messages(connections=[c for c in r.connections if c.topic in ("/wheel_commands", "/robot_joint_states")]):
            m = r.deserialize(raw, c.msgtype)
            ts = t * 1e-9
            if t_first is None:
                t_first = ts
            if c.topic == "/wheel_commands":
                cmd_rows.append((ts, dict(zip(m.name, m.velocity))))
            else:
                j = {n: i for i, n in enumerate(m.name)}
                fb_rows.append((ts, m.velocity, m.effort, j))
    if not cmd_rows:
        sys.exit("no /wheel_commands in the bag")
    names = ["left_wheel_joint", "right_wheel_joint"]
    if wheel is None:  # the wheel that got a nonzero command
        wheel = max(("left", "right"), key=lambda w: max(abs(r[1].get(f"{w}_wheel_joint", 0.0)) for r in cmd_rows))
    key = f"{wheel}_wheel_joint"
    assert key in names
    t_start = t_first + t0_override if t0_override is not None else cmd_rows[0][0]
    t_cmd = np.array([r[0] - t_start for r in cmd_rows]); cmd = np.array([r[1].get(key, 0.0) for r in cmd_rows])
    t_fb = np.array([r[0] - t_start for r in fb_rows])
    vel = np.array([r[1][r[3][key]] for r in fb_rows]); eff = np.array([r[2][r[3][key]] if len(r[2]) else 0.0 for r in fb_rows])
    keep = t_fb >= -1.0
    return wheel, t_cmd, cmd, t_fb[keep], vel[keep], eff[keep]


def read_jsonl(path):
    cmds, fbs, t_start = [], [], None
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        t = r["monotonic_ns"] * 1e-9
        if r["kind"] == "marker" and r.get("name") == "test_start":
            t_start = t
        elif r["kind"] == "command":
            cmds.append((t, r["velocity"]))
        elif r["kind"] == "feedback":
            fbs.append((t, r["velocity"], r["effort"]))
    if t_start is None:
        sys.exit("no test_start marker: the file is not from --stage full (or the run stopped before the profile)")
    t_cmd = np.array([c[0] - t_start for c in cmds]); cmd = np.array([c[1] for c in cmds])
    t_fb = np.array([f[0] - t_start for f in fbs]); vel = np.array([f[1] for f in fbs]); eff = np.array([f[2] for f in fbs])
    keep = (t_fb >= -1.0)
    return t_cmd[t_cmd >= 0], cmd[t_cmd >= 0], t_fb[keep], vel[keep], eff[keep]


def simulate(deadband=0.4, delay=0.035, tau=0.04, gain=0.9, noise=0.05, seed=0):
    """Synthetic wheel (dead time + first order + deadband + noise) to test the analysis."""
    rng = np.random.default_rng(seed)
    t_cmd = np.arange(0, P.DURATION, 0.02)
    cmd = np.array([P.command(x) for x in t_cmd])
    t = np.arange(-1.0, P.DURATION, 0.005)
    v = np.zeros_like(t)
    for i in range(1, len(t)):
        c = cmd[np.clip(np.searchsorted(t_cmd, t[i] - delay, side="right") - 1, 0, len(cmd) - 1)] if t[i] - delay >= 0 else 0.0
        target = 0.0 if abs(c) < deadband else gain * c
        v[i] = v[i - 1] + (target - v[i - 1]) * 0.005 / (tau + 0.005)
    return t_cmd, cmd, t, v + rng.normal(0, noise, len(t))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", nargs="?")
    ap.add_argument("--wheel", choices=("left", "right"))
    ap.add_argument("--t0", type=float)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        t_cmd, cmd, t, v = simulate()
        res = analyse(t_cmd, cmd, t, v)
        report(res, "synthetic wheel: deadband 0.4 rad/s, dead time 35 ms, tau 40 ms, gain 0.9, noise 0.05")
        return
    if a.bag.endswith(".jsonl"):
        wheel = Path(a.bag).stem
        t_cmd, cmd, t_fb, vel, eff = read_jsonl(a.bag)
        out_path = Path(a.bag).with_suffix(".result.json")
    else:
        wheel, t_cmd, cmd, t_fb, vel, eff = read_bag(a.bag, a.wheel, a.t0)
        out_path = Path(a.bag, "wheel_bench_result.json")
    res = analyse(t_cmd, cmd, t_fb, vel, eff)
    report(res, f"{wheel} wheel, {a.bag}")
    out_path.write_text(json.dumps({"wheel": wheel, **res}, indent=1))
    print(f"written {out_path}")


if __name__ == "__main__":
    main()
