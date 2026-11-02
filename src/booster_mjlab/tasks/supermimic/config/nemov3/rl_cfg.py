"""SuperMimic runner configuration for Nemov3 (the settings of the reference run exp5, 3 dance clips)."""

from booster_mjlab.supermimic.config import SuperMimicRunnerCfg


def nemov3_supermimic_runner_cfg() -> SuperMimicRunnerCfg:
    return SuperMimicRunnerCfg(
        experiment_name="nemov3_supermimic",
        seed=0,
        max_iterations=12_000,
        save_interval=1000,
        save_interval_late=500,
        save_late_from=3000,
    )
