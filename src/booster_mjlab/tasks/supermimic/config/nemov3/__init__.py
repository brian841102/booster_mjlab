from mjlab.tasks.registry import register_mjlab_task

from booster_mjlab.supermimic.runners import SuperMimicRunner

from .env_cfgs import nemov3_supermimic_env_cfg
from .rl_cfg import nemov3_supermimic_runner_cfg

register_mjlab_task(
    task_id="Mjlab-SuperMimic-Flat-Booster-Nemov3",
    env_cfg=nemov3_supermimic_env_cfg(),
    play_env_cfg=nemov3_supermimic_env_cfg(play=True),
    rl_cfg=nemov3_supermimic_runner_cfg(),
    runner_cls=SuperMimicRunner,
)
