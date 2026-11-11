"""SuperMimic runner configuration for Nemov3 (the settings of the reference run exp5, 3 dance clips)."""

from booster_mjlab.supermimic.config import SuperMimicRunnerCfg


def nemov3_supermimic_runner_cfg(wm_action_hist: int = 0) -> SuperMimicRunnerCfg:
    """``wm_action_hist``: 3 for the -Sim2Real- task, whose actuator delay (<= 50 ms = 2.5 control steps) makes the
    executed command an older action (tools/nemov3_finetune_supermimic.md, 排查 1)."""
    return SuperMimicRunnerCfg(
        experiment_name="nemov3_supermimic",
        seed=0,
        max_iterations=12_000,
        save_interval=1000,
        save_interval_late=500,
        save_late_from=3000,
        wm_action_hist=wm_action_hist,
    )
