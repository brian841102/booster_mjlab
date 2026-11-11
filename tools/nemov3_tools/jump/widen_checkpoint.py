"""Seed a jump run from a skating checkpoint: append zero-weighted input columns for the new observation terms.

The jump task appends ``jump_command`` (2 values) at the end of the actor and the critic observations. The new columns
of the first layer get zero weights and the observation normalizer gets mean 0 / variance 1 there, so the seeded policy
acts exactly like the source until training moves those weights. The discriminator is unchanged (same AMP features).
Optimizer state is dropped: start the run with ``--agent.resume True --agent.warm-start True``.

With ``min_std`` (J02+) the exploration std of the action slots in ``std_slots`` (default: hip pitch and calf of both
legs, slots 10 / 13 / 14 / 17 of the 20-action fixed-head-waist robot) is raised to at least that value: Exp 90 ends
with ~0.11-0.13 there, too little to stumble on a two-leg push-off.

usage: widen_checkpoint.py <source model_*.pt> <output model_*.pt> [n_new=2] [min_std] [std_slots=10,13,14,17]
"""

import sys
from pathlib import Path

import torch

src, out = Path(sys.argv[1]), Path(sys.argv[2])
n_new = int(sys.argv[3]) if len(sys.argv) > 3 else 2
ckpt = torch.load(src, map_location="cpu", weights_only=False)
for net in ("actor_state_dict", "critic_state_dict"):
    sd = ckpt[net]
    w = sd["mlp.0.weight"]
    sd["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], n_new)], dim=1)
    for key, fill in (("obs_normalizer._mean", 0.0), ("obs_normalizer._var", 1.0), ("obs_normalizer._std", 1.0)):
        v = sd[key]
        sd[key] = torch.cat([v, torch.full((v.shape[0], n_new), fill)], dim=1)
    print(f"{net}: input {w.shape[1]} -> {sd['mlp.0.weight'].shape[1]}")
if len(sys.argv) > 4:
    min_std = float(sys.argv[4])
    slots = [int(v) for v in (sys.argv[5] if len(sys.argv) > 5 else "10,13,14,17").split(",")]
    std = ckpt["actor_state_dict"]["distribution.std_param"]
    print("std before", std[slots].numpy().round(3))
    std[slots] = std[slots].clamp_min(min_std)
    print("std after ", std[slots].numpy().round(3))
ckpt.pop("optimizer_state_dict", None)
ckpt["iter"] = 0
# mjlab restores the env step counter from the checkpoint on load (mjlab/rl/runner.py); the source's (72024 after Exp 90's
# 3000 iterations) would put every step-based schedule of the new run at its end (J06's feed-forward anneal began at 0).
ckpt["infos"] = {"env_state": {"common_step_counter": 0}}
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(ckpt, out)
print("saved", out, "keys", sorted(ckpt))
