"""Training curves from a train log, averaged per window of iterations: is the run still improving?

usage: train_curve.py <train.log> [window=100] [metric substrings, comma separated]

Default metrics: AMP task / style reward (the user's stop rule: stop when both stop growing), episode length,
linear-velocity tracking, and every Episode_Reward/wheel_* term. Prints one row per window (mean over the window).
"""

import re
import sys
from collections import defaultdict

import numpy as np

DEFAULT = (
    "Episode_Metrics/Amp/task_reward",
    "Episode_Metrics/Amp/style_reward",
    "Mean episode length",
    "Episode_Reward/track_linear_velocity",
    "Episode_Reward/wheel_",
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
ITER = re.compile(r"Learning iteration (\d+)/")
ROW = re.compile(r"^\s*([A-Za-z_/ ]+?):\s+(-?[\d.]+(?:e-?\d+)?)\s*$")


def main():
    path = sys.argv[1]
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    wanted = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else DEFAULT
    it = None
    rows = defaultdict(lambda: defaultdict(list))
    with open(path, errors="replace") as f:
        for line in f:
            line = ANSI.sub("", line)
            m = ITER.search(line)
            if m:
                it = int(m.group(1))
                continue
            m = ROW.match(line)
            if m and it is not None and any(w in m.group(1) for w in wanted):
                rows[it // window][m.group(1).strip()].append(float(m.group(2)))
    names = sorted({k for r in rows.values() for k in r}, key=lambda k: [w in k for w in wanted].index(True))
    short = [n.split("/")[-1][:14] for n in names]
    print("iters      " + " ".join(f"{s:>14s}" for s in short))
    for b in sorted(rows):
        vals = [np.mean(rows[b][n]) if rows[b][n] else float("nan") for n in names]
        print(f"{b * window:5d}-{(b + 1) * window - 1:<5d}" + " ".join(f"{v:14.4f}" for v in vals))


if __name__ == "__main__":
    main()
