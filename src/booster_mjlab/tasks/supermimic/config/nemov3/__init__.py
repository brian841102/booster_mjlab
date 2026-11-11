from mjlab.tasks.registry import register_mjlab_task

from booster_mjlab.supermimic.runners import SuperMimicRunner

from .env_cfgs import nemov3_supermimic_env_cfg
from .rl_cfg import nemov3_supermimic_runner_cfg

# exp1: the supermimic_vibed exp5 setup (three dance clips, hardware PD, no sim-to-real).
register_mjlab_task(
    task_id="Mjlab-SuperMimic-Flat-Booster-Nemov3",
    env_cfg=nemov3_supermimic_env_cfg(),
    play_env_cfg=nemov3_supermimic_env_cfg(play=True),
    rl_cfg=nemov3_supermimic_runner_cfg(),
    runner_cls=SuperMimicRunner,
)
# exp2+: + AMP Exp 90's sim-to-real randomization, wheel speed clamp +-25 rad/s, all of it in the world model's xi;
# exp3+: the world model also sees the previous 3 actions (the delayed commands).
register_mjlab_task(
    task_id="Mjlab-SuperMimic-Flat-Sim2Real-Booster-Nemov3",
    env_cfg=nemov3_supermimic_env_cfg(sim2real=True),
    play_env_cfg=nemov3_supermimic_env_cfg(play=True, sim2real=True),
    rl_cfg=nemov3_supermimic_runner_cfg(wm_action_hist=3),  # exp3+ (exp2 had 0)
    runner_cls=SuperMimicRunner,
)
