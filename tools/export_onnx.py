"""Export a trained checkpoint (.pt) to ONNX with metadata. Works for nemov3 / pinov1 (any registered task).

usage:
  .venv/bin/python tools/export_onnx.py --task <task-id> --checkpoint <run_dir>/model_3000.pt [--out PATH] [--device cpu]

Default output: <run_dir>/exported/exp<N>_<ckpt stem>.onnx (N parsed from the run dir name, else just <ckpt stem>.onnx).
Metadata (joint names, PD, default pose, action scale, observation terms...) is attached like training-time export.
"""

import argparse
import re
from dataclasses import asdict
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

import booster_mjlab.tasks.velocity.config.nemov3  # noqa: F401
import booster_mjlab.tasks.velocity.config.pinov1  # noqa: F401
from booster_mjlab.rl.exporter_utils import (
    attach_base_metadata,
    attach_exclusion_metadata,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="registered task id the checkpoint was trained with")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    if a.task not in list_tasks():
        raise SystemExit(f"unknown task: {a.task}")
    ckpt = a.checkpoint.resolve()
    if a.out is None:
        m = re.search(r"exp\d+", ckpt.parent.name)
        a.out = ckpt.parent / "exported" / (f"{m.group(0)}_{ckpt.stem}.onnx" if m else f"{ckpt.stem}.onnx")
    out = a.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    env_cfg = load_env_cfg(a.task, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(a.task)
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=env_cfg, device=a.device), clip_actions=agent_cfg.clip_actions
    )
    runner = load_runner_cls(a.task)(env, asdict(agent_cfg), device=a.device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=a.device)
    runner.export_policy_to_onnx(str(out.parent), out.name)
    attach_base_metadata(env.unwrapped, str(out))
    attach_exclusion_metadata(env.unwrapped, out)

    import onnxruntime as ort

    s = ort.InferenceSession(str(out))
    print("saved", out)
    print("inputs ", [(i.name, i.shape) for i in s.get_inputs()])
    print("outputs", [(o.name, o.shape) for o in s.get_outputs()])


if __name__ == "__main__":
    main()
