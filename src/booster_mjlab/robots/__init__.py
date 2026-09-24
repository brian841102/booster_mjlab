from booster_mjlab.robots.booster_k1.k1_constants import (
    K1_ACTION_SCALE,
    get_k1_robot_cfg,
)
from booster_mjlab.robots.booster_k1.k1_parallel_constants import (
    K1_PARALLEL_ACTION_SCALE,
    get_k1_parallel_robot_cfg,
)
from booster_mjlab.robots.nemov3.nemov3_constants import (
    NEMOV3_ACTION_SCALE,
    NEMOV3_WHEEL_ACTION_SCALE,
    get_nemov3_robot_cfg,
)

__all__ = (
    "K1_ACTION_SCALE",
    "K1_PARALLEL_ACTION_SCALE",
    "NEMOV3_ACTION_SCALE",
    "NEMOV3_WHEEL_ACTION_SCALE",
    "get_k1_robot_cfg",
    "get_k1_parallel_robot_cfg",
    "get_nemov3_robot_cfg",
)
