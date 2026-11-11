"""Train one Pino v1 experiment from scratch and check it every 400 iterations (autonomous loop, user 2026-11-03).

usage: auto_exp.py <task> <run-name> [max_iterations=2000] [eval speeds, e.g. 0,0.3,0.6,0.9]

Starts the training, and at each multiple of 400 iterations reads the mean episode length (mean of the last 10
iterations from the training log). From >= 480 on (the robots survive the whole 10 s episode) it runs gait_eval.py on
that checkpoint; on PASS it stops the training and renders the 480p preview video (forward preset); a run that ends without PASS gets the video of its last checkpoint. A training log that stops growing for 10 min
counts as hung and the training is killed. Everything is
appended to /tmp/pinov1_auto_<run-name>.log; the last line is ``AUTO DONE <pass|fail> <run dir>``.
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv/bin/python")
task, run_name = sys.argv[1], sys.argv[2]
max_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
eval_speeds = [sys.argv[4]] if len(sys.argv) > 4 else []
log_path = Path(f"/tmp/pinov1_auto_{run_name}.log")
train_log = Path(f"/tmp/pinov1_train_{run_name}.log")
LOG_ROOT = ROOT / "logs/rsl_rl/pinov1_velocity_amp"


def say(msg: str) -> None:
    with log_path.open("a") as f:
        f.write(msg + "\n")


def episode_length(iteration: int, window: int = 10) -> float | None:
    """Mean of "Mean episode length" over the ``window`` iterations ending at ``iteration``."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", train_log.read_text(errors="ignore"))
    blocks = re.split(r"Learning iteration (\d+)/\d+", text)
    values = {int(blocks[i]): blocks[i + 1] for i in range(1, len(blocks) - 1, 2)}
    lengths = []
    for it in range(iteration - window + 1, iteration + 1):
        m = re.search(r"Mean episode length:\s*([\d.]+)", values.get(it, ""))
        if m:
            lengths.append(float(m.group(1)))
    return sum(lengths) / len(lengths) if lengths else None


def run_dir() -> Path | None:
    dirs = sorted(LOG_ROOT.glob(f"*_{run_name}"))
    return dirs[-1] if dirs else None


cmd = [PY, "-m", "booster_mjlab.scripts.train", task, "--env.scene.num-envs", "4096", "--agent.max-iterations",
       str(max_iter), "--agent.save-interval", "100", "--agent.experiment-name", "pinov1_velocity_amp",
       "--agent.run-name", run_name, "--agent.logger", "tensorboard"]
say(f"AUTO start {run_name}: {' '.join(cmd[2:])}")
train = subprocess.Popen(cmd, cwd=ROOT, stdout=train_log.open("w"), stderr=subprocess.STDOUT)
checked: set[int] = set()


def check(it: int, ckpt: Path) -> bool:
    """Episode-length gate, then the gait evaluation; True (and the video rendered) when the goal is met."""
    checked.add(it)
    length = episode_length(it)
    say(f"AUTO iter {it}: mean episode length {length}")
    if length is None or length < 480:
        return False
    out = subprocess.run([PY, "tools/pinov1_tools/gait_eval.py", task, str(ckpt), *eval_speeds], cwd=ROOT, capture_output=True,
                         text=True)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith(("GE", "GF", "REST", "PASS", "=="))]
    say("\n".join(lines) if lines else "gait_eval failed:\n" + out.stderr[-1500:])
    if not any(ln.startswith("PASS 1") for ln in lines):
        return False
    train.terminate()
    video = ckpt.parent / "videos" / f"{run_name}_model{it}_forward_480p_leftfront.mp4"
    video.parent.mkdir(exist_ok=True)
    subprocess.run([PY, "tools/nemov3_tools/record_video.py", task, str(ckpt), str(video), "forward", "854", "480",
                    f"{run_name}@{it}"], cwd=ROOT, env={**os.environ, "MUJOCO_GL": "egl"}, capture_output=True)
    say(f"AUTO PASS at iteration {it}; video {video}")
    return True


def sweep(final: bool = False) -> bool:
    d = run_dir()
    if d is None:
        return False
    its = list(range(400, max_iter, 400)) + ([max_iter - 1] if final else [])
    for it in its:
        ckpt = d / f"model_{it}.pt"
        if it not in checked and ckpt.exists():
            time.sleep(5)  # let the file finish writing
            if check(it, ckpt):
                return True
    return False


STALL_S = 600  # the training log not growing for this long = hung (Exp 11 hung at iteration 1303 for 12 h)
passed = False
while train.poll() is None and not passed:
    time.sleep(20)
    passed = sweep()
    if time.time() - train_log.stat().st_mtime > STALL_S:
        say(f"AUTO STALL: training log unchanged for {STALL_S} s, stopping the training")
        train.kill()
        break
if not passed:
    train.wait()
    passed = sweep(final=True)
if not passed:
    # A preview video for every result (user 2026-11-03): the last checkpoint of the run.
    d = run_dir()
    ckpts = sorted(d.glob("model_*.pt"), key=lambda c: int(c.stem.split("_")[1])) if d else []
    if ckpts:
        video = d / "videos" / f"{run_name}_{ckpts[-1].stem}_forward_480p_leftfront.mp4"
        video.parent.mkdir(exist_ok=True)
        subprocess.run([PY, "tools/nemov3_tools/record_video.py", task, str(ckpts[-1]), str(video), "forward", "854",
                        "480", f"{run_name}@{ckpts[-1].stem.split('_')[1]}"], cwd=ROOT,
                       env={**os.environ, "MUJOCO_GL": "egl"}, capture_output=True)
        say(f"AUTO video {video}")
say(f"AUTO DONE {'pass' if passed else 'fail'} {run_dir()}")
