"""Nemov3 (wheeled-legged) velocity tracking environment configuration.

Flat terrain, AMP-only, straight-line-forward first pass: the twist command
is pinned to 0-1.3 m/s forward with no lateral or yaw component, matching
the only motion data currently prepared for this robot (see
``assets/amp_sources/nemov3_skating`` / ``assets/amp/nemov3_skating_straight``).

Differences from the K1 config this is adapted from
(``tasks/velocity/config/k1/env_cfgs.py``):

* No rough-terrain variant / no ``foot_height_scan`` terrain-clearance
  sensor: Nemov3's "feet" are driven wheels that stay in continuous ground
  contact, so there is no biped swing phase to track clearance for.
* The K1 reward terms that assume a discrete swing/stance cycle
  (``air_time``, ``foot_clearance``, ``foot_swing_height``, ``foot_slip``,
  ``soft_landing``) are dropped rather than re-pointed at the wheels: a
  wheel-mounted site is never stationary during "contact" the way a foot
  sole is under pure rolling, so the slip/landing formulations would
  misread normal wheel spin as a fault. ``foot_contact``/
  ``foot_contact_forces`` (binary/force ground-contact signal) are kept.
* Two action terms instead of one: the MJCF's own ``<position>`` actuators
  (everything but the wheels) go through ``JointPositionActionCfg`` as
  usual, but the wheels are ``<velocity>`` actuators in the XML, so they get
  a second ``JointVelocityActionCfg`` term.
"""

from __future__ import annotations

import dataclasses
import re

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from booster_mjlab.robots import (
    NEMOV3_ACTION_SCALE,
    NEMOV3_WHEEL_ACTION_SCALE,
    get_nemov3_robot_cfg,
)
from booster_mjlab.robots.nemov3.nemov3_constants import (
    NEMOV3_DELAY_GROUPS,
    NEMOV3_DEPLOY_PD_GAINS,
    NEMOV3_POSITION_ACTUATOR_NAME_EXPR,
    NEMOV3_POSITION_JOINT_NAMES,
    NEMOV3_WHEEL_JOINT_NAMES,
)
from booster_mjlab.tasks.velocity.mdp.motor_dynamics import (
    MotorDynamicsJointPositionActionCfg,
    MotorDynamicsJointVelocityActionCfg,
    ActuatorDelayCfg,
    MotorTargetDynamicsCfg,
    sample_actuator_delay,
)
from booster_mjlab.tasks.velocity.mdp.rewards import torso_forward_lean
from booster_mjlab.tasks.velocity.mdp.stance import hip_roll_excess
from booster_mjlab.tasks.velocity.mdp.wheel_feet import (
    reward_param_schedule,
    reward_weight_schedule,
    single_wheel_lift,
    wheel_air_time,
    wheel_clearance,
    wheel_height_excess,
    wheel_jump,
    wheel_lift,
    wheel_single_stance,
    wheel_support_switch,
    wheel_swing_height,
)
from booster_mjlab.tasks.velocity.mdp.gait_clock import GaitClockCommandCfg
from booster_mjlab.tasks.velocity.mdp.play_controls import (
    PlayControlsCommandCfg,
    push_if_enabled,
)
from booster_mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TRUNK_BODY = "link_torso"
WHEEL_LINK_NAMES = ("left_wheel_link", "right_wheel_link")
WHEEL_COLLISION_GEOMS = ("left_wheel_collision", "right_wheel_collision")

# Gait clock (clock=True). Frequency from the reference clips: all three have a ~1.35 s stride period
# (0.74 Hz).
GAIT_FREQUENCY_HZ = 0.74


# Hip pitch / calf action scale (rad per action unit). The 0.25*forcerange/kp
# heuristic gives only ~0.09-0.1 for these joints (K1: 0.212 hip pitch, 0.35
# knee) so the reference stride is ~3-9 action units away; with these it is
# ~1-3. Alone it did not create a gait (Exp 5/6), but it is what makes
# "keep swinging" representable for the AMP reward to pull towards.
LEG_SWING_ACTION_SCALE = {
    "right_hip_pitch_joint": 0.25,
    "left_hip_pitch_joint": 0.25,
    "right_calf_joint": 0.35,
    "left_calf_joint": 0.35,
}

# Whether the policy/critic observe the absolute wheel angles. They must not:
# a wheel angle grows without bound with distance travelled (~13 rad/s at
# 1 m/s), so it is an unintended clock/odometer input -- the policy only ever
# sees angles up to (training episode length x speed / wheel radius), and once
# an evaluation rollout outlasts that range the gait dies (Exp 18: hip swing std
# 0.33 -> 0.13 -> 0.07 over 2-10 / 10-18 / 18-26 s; feeding zeros for the wheel
# angles kept it at 0.37 in every window). Wheel *velocity* stays observed.
# Set True only to replay checkpoints trained before Exp 19.
NEMOV3_OBSERVE_WHEEL_ANGLE = False

# Foot-lift variant (foot_lift=True): K1-style swing-height / air-time rewards
# adapted to wheels (mdp/wheel_feet.py). The target height comes from the
# reference data: in SkatingFast06 (the only clip with foot lifting) each foot's
# wheel bottom (wheel-center z - 0.075 m) rises to 8.5 cm (8.4~8.5 cm over all
# 13+13 lifts), for ~0.55 s per lift, at ~0.98 m/s; Skating01/02 never lift
# more than 0.3 cm (measured with tools/nemov3_tools/archive/measure_foot_lift.py). The
# terms fade in over commanded speed 0.6 -> 1.0 m/s (Skating01 at 0.6 m/s does
# not lift, Fast06 at ~1.0 m/s does) rather than switching on at one speed.
# Weights: air_time / swing_height as K1; clearance half of K1's -2.0 because
# the target is ~1.4x K1's 6 cm.
FOOT_LIFT_TARGET_M = 0.085
FOOT_LIFT_RAMP_SPEED = (0.6, 1.0)
FOOT_LIFT_WEIGHTS = {"air_time": 0.1, "clearance": -1.0, "swing_height": -0.25}
# "shaped" variant (Exp 26): the positive lift reward needs a much larger weight than K1's
# air_time 0.1: it is multiplied by (1 - style weight) = 0.3 in the AMP mix, and Exp 25 showed
# 0.1 is invisible next to the style / tracking terms.
# Exp 26 used lift 1.0 / swing_height -0.25 / linear height / min air time 0.05 s: the wheels never left the ground (wheel_lift stayed 0.0003).
# Exp 27: lift 2.0 (concave sqrt height so the first cm already pays), swing_height -1.0 (so micro-hops that land far below the target are net-neutral,
# proper lifts are strongly positive), min air time 0.02 s (one control step).
FOOT_LIFT_SHAPED_WEIGHTS = {"lift": 2.0, "swing_height": -1.0}

# Narrow-stance variant (narrow_stance=True): squared-hinge penalty on hip roll above
# HIP_ROLL_LIMIT_RAD. Data (tools/nemov3_tools/archive/stance_width.py): Skating01/02 keep hip roll
# <= 0.141 rad (wheel track 0.215-0.242 m); Fast06 spreads to 0.26 rad (track up to 0.31 m)
# and the Exp 23 policy drifts to the same wide stance at 1.3 m/s (left hip roll mean 0.155,
# p95 0.198; track p95 0.293), where the user saw it jump into a Fast06-like large-amplitude style.
# WARNING for future foot-lift training: Fast06's foot lifting needs hip roll up to ~0.25 rad,
# so this reward may have to be removed / relaxed then -- it can block freedom the lift needs.
HIP_ROLL_LIMIT_RAD = 0.14
HIP_ROLL_SPREAD_WEIGHT = -100.0

# Foot-lift skating (lift=..., docs/nemov3_foot_lift_plan.md; tools/nemov3_finetune.md Exp 46-77, section 抬脚).
# Hip-roll upper limit (user, 2026-09-25): relaxed from 0.14 to 0.2 rad, still short of Fast06's ~0.26 (reached by
# the lifted leg only; the support leg stays <= 0.11), to keep the sideways swing small.
FOOT_LIFT_HIP_ROLL_LIMIT_RAD = 0.2

# Single-wheel feasibility probe (single_wheel=True, Exp 46): can the robot ride on the right wheel with the left one
# held >= SINGLE_WHEEL_TARGET_M up, and at which speeds? Trained without style reward; forward speed 0 -
# SINGLE_WHEEL_MAX_SPEED m/s, no turning.
SINGLE_WHEEL_LIFT_WEIGHT = 3.0
SINGLE_WHEEL_TARGET_M = 0.05
SINGLE_WHEEL_MAX_SPEED = 1.2

# Contact-timing rewards after 2607.10815's Push Glide ([B]; user: "接触时序奖励可以学B"): hop penalty, single-stance
# pay, support-switch pay. One policy for all speeds, the rewards only above LIFT_RAMP_SPEED[0] m/s (user: "你可以让奖励在
# 0.6米每秒速度以上才引入"), faded in up to LIFT_RAMP_SPEED[1] to avoid a style switch at one speed. Fast06 (steady part):
# peak 8.3-8.5 cm, 0.37-0.42 s with the other wheel > 3 cm higher, strictly alternating.
LIFT_RAMP_SPEED = (0.6, 1.0)
LIFT_TARGET_HEIGHT_M = 0.05  # single-stance pay saturates here; the discriminator's height feature shapes the rest
LIFT_MAX_SINGLE_STANCE_S = 0.6
LIFT_SWITCH_MIN_HEIGHT_M = 0.03
STEPS_PER_ITERATION = 24
# Per recipe (env lift=...): (start weight, end weight, first iteration, last iteration) of each term's linear weight
# schedule, and the air-time range of a lift the support switch pays.
#   "early" (Exp 49): with driven wheels a lift has no task benefit, so it must be found while the policy still explores
#     -- [B]'s weak start (Exp 47 / 48) never lifted; strong single-stance pay from the start, +5 per switch.
#   "natural" (Exp 50 from scratch, Exp 51 resumed from Exp 49; after the user watched Exp 49: "抬脚节奏太快了 ... 原地来回
#     跺脚的感觉很不自然 ... 离地六公分感觉就差不多了"): switch +1 only after >= 0.35 s, lift pay handed over to the style
#     sooner, a penalty above 8 cm -- neither found a lift / kept one (Exp 51 dropped its lifts within 100 iterations).
#   "staged" (Exp 52+): learn first, then shape -- iterations 0-400 the Exp 49 recipe, 400-1200 a linear move to the
#     natural targets: switch 5 -> 1 and its minimum air time 0.15 -> 0.35 s (LIFT_SWITCH_MIN_AIR_TIME_STAGED), forward
#     lean 0 -> 1.0, the > 8 cm bound 0 -> -1.0.
#   "staged_end" (Exp 59, resumed runs): the staged end values held constant (a resumed env restarts its step counter,
#     which would replay the schedules from their start).
LIFT_WEIGHT_SCHEDULES = {
    "early": {
        "wheel_single_stance": (3.0, 1.5, 0, 1500),
        "wheel_support_switch": (5.0, 5.0, 0, 1),
        "wheel_jump": (-2.5, -1.8, 0, 900),
    },
    "natural": {
        "wheel_single_stance": (3.0, 0.8, 0, 1000),
        "wheel_support_switch": (1.0, 1.0, 0, 1),
        "wheel_jump": (-2.5, -1.8, 0, 900),
    },
    "staged": {
        "wheel_single_stance": (3.0, 1.5, 0, 1200),
        "wheel_support_switch": (5.0, 1.0, 400, 1200),
        "wheel_jump": (-2.5, -1.8, 0, 900),
        "wheel_height_excess": (0.0, -1.0, 400, 1000),  # end = LIFT_HEIGHT_EXCESS_WEIGHT
        "torso_forward_lean": (0.0, 1.0, 400, 1000),
    },
}
LIFT_WEIGHT_SCHEDULES["staged_end"] = {
    name: (end, end, 0, 1) for name, (_, end, _, _) in LIFT_WEIGHT_SCHEDULES["staged"].items()
}
LIFT_SWITCH_AIR_TIME_S = {"early": (0.15, 0.9), "natural": (0.35, 1.0), "staged": (0.15, 1.0), "staged_end": (0.35, 1.0)}
LIFT_SWITCH_MIN_AIR_TIME_STAGED = (0.15, 0.35, 400, 1200)
LIFT_HEIGHT_LIMIT_M = 0.08
LIFT_HEIGHT_EXCESS_WEIGHT = -1.0

# Forward lean (lean=True; user: "正常滑冰应该是身体略微倾斜"): Fast06 leans the torso forward 10.8 deg (std 1.4) at
# ~0.9 m/s; Skating01/02 carry no torso orientation (constant 0 in the data), so the discriminator cannot teach it (and
# must not see it). Target = LEAN_TARGET_RAD * min(|cmd| / LEAN_FULL_SPEED, 1): 0 when standing (stand straight).
LEAN_TARGET_RAD = 0.175
LEAN_FULL_SPEED = 1.0
LEAN_STD_RAD = 0.1
LEAN_WEIGHT = 1.0

# Payload uncertainty of the upper body (trunk_mass_com=True, Exp 88; flamingo_rl_train's old recipe "base mass +-1 kg,
# COM +-3 cm", which it applies to its base body -- here on the torso, the user's "躯干", since link_leg2pelvis weighs
# only 0.39 kg): a point mass of -1..+1 kg added at the torso's COM (inertia unchanged) and its COM moved by -3..+3 cm
# per axis, on top of the trunk_inertia event; drawn once per environment at startup.
TRUNK_MASS_ADD_KG = (-1.0, 1.0)
TRUNK_COM_OFFSET_M = (-0.03, 0.03)

# Relaxed upright tolerance while moving (upright_std_moving, Exp 59; user: "会不会有可能是水平惩罚太重导致学不出判别器步态"):
# the upright reward exp(-|g_xy|^2 / std^2) with the moving std 0.5 costs 14 % at Fast06's 11 deg lean (weight 1.0,
# x0.3 task share ~0.04 / step), about what the discriminator gains from the lean -- so the lean may be cancelled.
# With 1.0 an 11 deg lean costs ~4 %; standing keeps its std (stand straight).
UPRIGHT_STD_MOVING_RELAXED = 1.0

# Diagonal arm swing (arm_yaw=True; user: "手臂摆动自然带动上身 ... 左脚在前的时候对应胳膊往左前方摆动"): in Fast06 the arm
# opposite the leading foot swings forward and across mostly through shoulder yaw (std 0.21 rad), with pelvis yaw
# +-0.13 rad. Shoulder yaw leaves the upper_body_posture penalty and both get a larger action scale; the discriminator
# scores them (config/nemov3/__init__.py, NEMOV3_AMP_SKATE_JOINT_NAMES).
ARM_YAW_ACTION_SCALE = {
    "joint_lShoulderYaw": 0.25,
    "joint_rShoulderYaw": 0.25,
    "joint_pelvisYaw": 0.2,
}

# Lift tempo on the gait clock (lift_clock, Exp 77a / 77a2; user: "符合Fast06的节奏 ... 10%误差内可接受"): Fast06's stride
# cycle is 1.33 s -- two ~0.4 s lifts with ~0.27 s on both wheels in between (59 % single support) -- while the lifts
# alone ran at a 1.03 s cycle (Exp 76). The observed gait clock (GAIT_FREQUENCY_HZ 0.74 = 1.35 s) gives each wheel a lift
# window: the right wheel's single-stance pay only in the first LIFT_CLOCK_WINDOW of the cycle (0.41 s), the left
# wheel's half a cycle later, both wheels down in between; ramped in over 400-1200 (LIFT_CLOCK_ALPHA). lift_clock=True
# (Exp 77a) dropped every lift while the window was ramped in -- a lift at a random clock phase keeps only ~30 % of its
# pay; lift_clock="compensated" (Exp 77a2) pays in-window lifts x 1 / LIFT_CLOCK_WINDOW, so such a lift keeps its
# expected pay and lifts timed to the clock earn up to 3.3x more. Weighting the pay by the air time instead
# (Exp 61-66 / 75 / 77b) slowed the tempo by lengthening each lift and swung the arms on the same side.
LIFT_CLOCK_WINDOW = 0.30
LIFT_CLOCK_ALPHA = (0.0, 1.0, 400, 1200)
LIFT_CLOCK_GAIN_COMPENSATED = 1.0 / LIFT_CLOCK_WINDOW

# Turning variant (turn=True): the yaw-rate command is sampled from [-TURN_YAW_RATE_MAX, TURN_YAW_RATE_MAX]
# rad/s, i.e. left and right turns (user request: turning at 0~1.3 rad/s; Exp 41+). The straight-line tasks keep
# the yaw-rate command at 0.
TURN_YAW_RATE_MAX = 1.3

# Symmetric-default variant (symmetric_default=True): HOME_KEYFRAME's hip yaw is -0.09 (left) /
# -0.06 (right), the per-clip mean of the unmirrored references. Joint positions are observed
# relative to the default and actions are offsets from it, so this asymmetry breaks the
# left/right mirror symmetry the PPO symmetry augmentation relies on (verified: it is the only
# term of mirror(obs(s)) - obs(mirror(s)) that is not 0, tools/nemov3_tools/symmetry_check.py).
# With mirrored reference clips the mean is -0.075 on both sides.
SYMMETRIC_HIP_YAW_DEFAULT = {"left_hip_yaw_joint": -0.075, "right_hip_yaw_joint": -0.075}

# Arm-swing variant (arms=True): shoulder pitch / elbow become AMP-driven, so
# they get a larger action scale (the 0.25*forcerange/kp heuristic gives only
# 0.1 rad/unit) and a default pose at the reference mean (clip-equal mean of
# Skating01/02/Fast06: shoulder pitch +0.05, elbow -0.35).
ARM_SWING_ACTION_SCALE = {
    "joint_rShoulderPitch": 0.25,
    "joint_lShoulderPitch": 0.25,
    "joint_rElbow": 0.25,
    "joint_lElbow": 0.25,
}
ARM_DEFAULT_POS = {
    "joint_rShoulderPitch": 0.05,
    "joint_lShoulderPitch": 0.05,
    "joint_rElbow": -0.35,
    "joint_lElbow": -0.35,
}

# Training episode length. Every episode starts from a reference pose+velocity
# (reset_from_pose_pool), where "keep swinging" vs "freeze" differ strongly in
# discriminator reward. With 20 s episodes only the first few steps of ~1000
# are near the reference manifold; the rest is the frozen-legs optimum. Short
# episodes multiply the density of that informative signal.
NEMOV3_TRAIN_EPISODE_LENGTH_S = 4.0


def nemov3_flat_env_cfg(
    play: bool = False,
    clock: bool = False,
    arms: bool = False,
    foot_lift: bool | str = False,
    narrow_stance: bool = False,
    symmetric_default: bool = False,
    turn: bool = False,
    hip_roll_limit: float = HIP_ROLL_LIMIT_RAD,
    single_wheel: bool = False,
    lift: str | None = None,
    lean: bool = False,
    arm_yaw: bool = False,
    upright_std_moving: float | None = None,
    lift_clock: bool | str = False,
    fixed_joints: tuple[str, ...] = (),
    deploy_pd: bool = False,
    actuator_delay_steps: int = 0,
    actuator_delay: ActuatorDelayCfg | None = None,
    motor_dynamics: MotorTargetDynamicsCfg | None = None,
    trunk_mass_com: bool | str = False,
    backward: float = 0.0,
) -> ManagerBasedRlEnvCfg:
    """Create Nemov3 flat-terrain velocity tracking configuration.

    With ``backward=V`` (> 0) the forward command is sampled from [-V, 1.3] m/s instead of [0, 1.3] (Exp 90: Exp 83/85/88
    never saw a negative command and spread the legs to 40+ cm when driven backward). No extra reward.

    With ``turn=True`` the yaw-rate command is sampled from [-TURN_YAW_RATE_MAX, TURN_YAW_RATE_MAX] rad/s (left
    and right turns) instead of being pinned to 0.

    With ``narrow_stance=True`` adds a hip-roll upper-limit penalty so the feet do not spread, above
    ``hip_roll_limit`` (see the note at ``HIP_ROLL_LIMIT_RAD``; foot-lift training uses FOOT_LIFT_HIP_ROLL_LIMIT_RAD).

    With ``single_wheel=True`` adds the single-wheel feasibility reward (ride on the right wheel with the left one
    held up) and limits the forward command to SINGLE_WHEEL_MAX_SPEED.

    With ``lift`` ("early" / "natural" / "staged" / "staged_end", see LIFT_WEIGHT_SCHEDULES) adds the [B]-style
    contact-timing rewards (hop penalty, single stance, support switch), active only above LIFT_RAMP_SPEED[0], and
    except for "early" a lift-height bound. ``lift_clock`` (True or "compensated"; needs ``clock``) times the lifts by
    the gait clock (LIFT_CLOCK_*). With ``lean=True`` adds a speed-dependent forward-lean target for the torso (LEAN_*;
    its weight follows the staged schedules). With ``arm_yaw=True`` (needs ``arms``) frees shoulder yaw from the
    upper-body posture penalty and enlarges the shoulder / pelvis yaw action scale. ``upright_std_moving`` overrides
    the tilt tolerance of the ``upright`` reward while moving (standing keeps its own), see UPRIGHT_STD_MOVING_RELAXED.

    ``fixed_joints`` (e.g. NEMOV3_FIXED_HEAD_WAIST_JOINTS) are welded in the model for hardware that does not actuate
    them: they leave the actions and observations, and reward terms drop them. The AMP clips must drop their columns
    too (dataset transform drop_fixed_head_waist_joints; config/nemov3/__init__.py does that for such tasks).

    Sim-to-real (Exp 81+): ``deploy_pd=True`` uses the hardware's PD gains (NEMOV3_DEPLOY_PD_GAINS: the arms are far
    softer than the XML's kp 50); ``actuator_delay_steps`` = N delays the actuator commands by a lag of 0..N physics
    steps (5 ms each), one per environment and joint group (arms / legs / wheels), drawn at reset; ``actuator_delay``
    (ActuatorDelayCfg) instead sets the groups / range, optionally with a second, motor-target-style lag on top; ``motor_dynamics`` runs the joint-position and wheel-velocity targets through
    MotorTargetDynamics (per-episode motor rate hold, transport delay, first-order low-pass; mdp/motor_dynamics.py).
    ``trunk_mass_com=True`` randomizes the torso payload (TRUNK_MASS_ADD_KG, TRUNK_COM_OFFSET_M); ``"mass"`` only the
    mass. Both are set relative to the XML values, so they replace the trunk_inertia event's mass / COM draws (+-10 %,
    +-5 cm per axis) for the torso; with ``"mass"`` its COM draw stays.

    With ``foot_lift=True`` (or ``"k1"``) adds K1-style air-time / clearance /
    swing-height rewards for the wheels, targeting the foot lift measured in
    Fast06 (Exp 25: no lift learned). ``foot_lift="shaped"`` instead adds a
    positive height-weighted, rear-foot-gated lift reward plus the landing
    peak-height penalty (Exp 26).

    With ``arms=True`` shoulder pitch and elbow are released from the
    ``upper_body_posture`` penalty (so the AMP discriminator, which must be
    given those joints, is their only style pressure), get a larger action
    scale and a default pose at the reference mean.

    With ``clock=True`` only adds a free-running phase clock (sin/cos, 0.74 Hz,
    random phase per episode) to the actor/critic observations -- no reward. It
    gives a memoryless MLP policy a time signal to build a periodic gait from,
    while the *style* (amplitudes, phases, shapes) is still shaped only by the
    AMP discriminator.
    """
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {
        "robot": get_nemov3_robot_cfg(
            fixed_joints=fixed_joints,
            pd_gains=NEMOV3_DEPLOY_PD_GAINS if deploy_pd else (),
            actuator_delay_steps=(
                actuator_delay.buffer_steps(cfg.sim.mujoco.timestep) if actuator_delay else actuator_delay_steps
            ),
        )
    }
    position_joints = tuple(name for name in NEMOV3_POSITION_JOINT_NAMES if name not in fixed_joints)

    # Flat terrain only. K1's own buffer sizes are fine here (verified at
    # num_envs=4096, 10 iterations, no NaN, no OOM) -- an earlier NaN was
    # mistakenly attributed to undersized njmax/nconmax and "fixed" by
    # reusing the source XML's large njmax=2000/nconmax=400, but the real
    # cause was the `limb_inertia` domain-randomization event corrupting
    # near-degenerate inertia tensors (see below); the buffer size was never
    # the problem. The large buffers just ate GPU memory nemov3 doesn't
    # need: at 137 geoms (32 of them non-trivial meshes for collision) they
    # limited num_envs to ~1024-1536 before OOMing, vs. 4096 at K1's sizes.
    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 60
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = 50
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Sensors. Reuses the generic "feet_ground_contact"/"non_foot_ground_contact"
    # names the base mdp's observation/reward terms already look for, pointed
    # at the wheel bodies instead of foot bodies. No terrain-height scan (see
    # module docstring), so foot_height_scan is dropped from the scene and
    # its obs/reward terms below.
    wheels_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_wheel_link|right_wheel_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    non_wheel_ground_cfg = ContactSensorCfg(
        name="non_foot_ground_contact",
        primary=ContactMatch(
            mode="body",
            entity="robot",
            pattern=r".*",
            exclude=WHEEL_LINK_NAMES,
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        # link_pelvisYaw's subtree covers pelvis/torso/arms/head but not the
        # legs+wheels branch, mirroring K1's "Trunk" self-collision pattern.
        primary=ContactMatch(mode="subtree", pattern="link_pelvisYaw", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="link_pelvisYaw", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (wheels_ground_cfg, non_wheel_ground_cfg, self_collision_cfg)

    # Actions: position control for everything except the wheels, velocity
    # control for the wheels (matches the MJCF's own actuator types).
    pos_action_cls, vel_action_cls, motor = JointPositionActionCfg, JointVelocityActionCfg, {}
    if motor_dynamics is not None:
        pos_action_cls, vel_action_cls = MotorDynamicsJointPositionActionCfg, MotorDynamicsJointVelocityActionCfg
        motor = {"motor_dynamics": motor_dynamics}
    cfg.actions["joint_pos"] = pos_action_cls(
        entity_name="robot",
        actuator_names=position_joints,
        scale=NEMOV3_ACTION_SCALE,
        use_default_offset=True,
        **motor,
    )
    cfg.actions["joint_pos"].scale = {**NEMOV3_ACTION_SCALE, **LEG_SWING_ACTION_SCALE}
    cfg.actions["wheel_vel"] = vel_action_cls(
        entity_name="robot",
        actuator_names=NEMOV3_WHEEL_JOINT_NAMES,
        scale=NEMOV3_WHEEL_ACTION_SCALE,
        use_default_offset=True,
        **motor,
    )

    cfg.viewer.body_name = TRUNK_BODY

    # Command: forward only, 0-1.3 m/s; yaw rate 0 (straight line) unless turn=True (turning). No
    # lateral, no heading-seeking behavior, and no curriculum widening (the base config's "command_vel"
    # curriculum progressively opens up lateral and yaw ranges, which we deliberately don't want here).
    # Of the envs, rel_standing_envs (10 %) stand and rel_forward_envs (10 %, base config) go straight
    # with zero yaw rate, so the straight-line gait keeps being trained when turning is on.
    assert cfg.commands is not None
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, mdp.UniformVelocityCommandCfg)
    twist_cmd.heading_command = False
    twist_cmd.rel_standing_envs = 0.1
    twist_cmd.ranges.lin_vel_x = (0.0 - backward, 1.3)
    twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
    twist_cmd.ranges.ang_vel_z = (-TURN_YAW_RATE_MAX, TURN_YAW_RATE_MAX) if turn else (0.0, 0.0)
    twist_cmd.ranges.heading = None

    # Events.
    # pd_gains randomization only supports position-mode actuators (not the
    # wheels' velocity-mode XmlActuator), so restrict it to the position group.
    cfg.events["pd_gains"].params["asset_cfg"] = SceneEntityCfg(
        "robot", actuator_names=NEMOV3_POSITION_ACTUATOR_NAME_EXPR
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = WHEEL_COLLISION_GEOMS
    cfg.events["trunk_inertia"].params["asset_cfg"].body_names = (TRUNK_BODY,)
    # Dropped rather than re-pointed: unlike K1's ~25 fairly uniform limb
    # bodies, several of Nemov3's real (non-decorative) bodies have very
    # thin/near-degenerate inertia tensors (e.g. left/right_calf_link and
    # link_*LinearOut have a principal-moment ratio as low as ~0.04-0.07,
    # versus K1's more uniform limbs). pseudo_inertia's default alpha/t
    # perturbation range can push those non-positive-definite, which
    # silently NaNs the whole mass-matrix factorization (qLD/qfrc_bias) on
    # the very first reset -- found by bisecting which startup DR event
    # produced NaN AMP/critic obs, then confirmed by excluding first the
    # ~36-microgram head/eye and arm-linkage decorations (which fixed it
    # sometimes) and then the LinearOut bodies too (still recurred with a
    # different body on a later run). Excluding bodies one at a time chased
    # a moving target across restarts (the perturbation is randomly
    # sampled), so this is dropped outright rather than narrowed further;
    # re-enabling it for a curated, per-body-safe subset with tighter
    # ranges is a reasonable follow-up but needs its own validation pass.
    cfg.events.pop("limb_inertia", None)

    # Rewards: point body-name params at the trunk; drop the swing/clearance
    # family (see module docstring) along with the terrain-clearance obs
    # terms and the now-dangling curriculum entries for those rewards.
    cfg.rewards["upright"].params["asset_cfg"].body_names = (TRUNK_BODY,)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (TRUNK_BODY,)
    cfg.rewards["upper_body_posture"].params["asset_cfg"].joint_names = (
        r"head_.*_joint",
        r"joint_[rl]Shoulder.*",
        r"joint_[rl]Elbow",
    )
    cfg.rewards["upper_body_posture"].params["std_standing"] = {
        r"head_.*_joint": 0.05,
        r"joint_[rl]Shoulder.*": 0.05,
        r"joint_[rl]Elbow": 0.05,
    }
    cfg.rewards["upper_body_posture"].params["std_walking"] = {
        r"head_.*_joint": 0.05,
        r"joint_[rl]Shoulder.*": 0.15,
        r"joint_[rl]Elbow": 0.15,
    }
    cfg.rewards["upper_body_posture"].params["std_running"] = {
        r"head_.*_joint": 0.05,
        r"joint_[rl]Shoulder.*": 0.2,
        r"joint_[rl]Elbow": 0.35,
    }
    if arms:
        # HOME_KEYFRAME is a module-level object shared by every variant, so
        # replace (don't mutate) this variant's copy of the init state.
        robot_cfg = cfg.scene.entities["robot"]
        robot_cfg.init_state = dataclasses.replace(
            robot_cfg.init_state,
            joint_pos={**robot_cfg.init_state.joint_pos, **ARM_DEFAULT_POS},
        )
        assert isinstance(cfg.actions["joint_pos"], JointPositionActionCfg)
        cfg.actions["joint_pos"].scale = {
            **cfg.actions["joint_pos"].scale,
            **ARM_SWING_ACTION_SCALE,
        }
        posture = cfg.rewards["upper_body_posture"].params
        posture["asset_cfg"].joint_names = (
            r"head_.*_joint",
            r"joint_[rl]ShoulderRoll",
            r"joint_[rl]ShoulderYaw",
        )
        posture["std_standing"] = {
            r"head_.*_joint": 0.05,
            r"joint_[rl]ShoulderRoll": 0.05,
            r"joint_[rl]ShoulderYaw": 0.05,
        }
        posture["std_walking"] = {
            r"head_.*_joint": 0.05,
            r"joint_[rl]ShoulderRoll": 0.15,
            r"joint_[rl]ShoulderYaw": 0.15,
        }
        posture["std_running"] = {
            r"head_.*_joint": 0.05,
            r"joint_[rl]ShoulderRoll": 0.2,
            r"joint_[rl]ShoulderYaw": 0.2,
        }
        if arm_yaw:
            cfg.actions["joint_pos"].scale = {**cfg.actions["joint_pos"].scale, **ARM_YAW_ACTION_SCALE}
            posture["asset_cfg"].joint_names = (r"head_.*_joint", r"joint_[rl]ShoulderRoll")
            for key in ("std_standing", "std_walking", "std_running"):
                posture[key] = {k: v for k, v in posture[key].items() if "ShoulderYaw" not in k}

    if upright_std_moving is not None:
        cfg.rewards["upright"].params["std_walking"] = upright_std_moving
        cfg.rewards["upright"].params["std_running"] = upright_std_moving

    if lean:
        cfg.rewards["torso_forward_lean"] = RewardTermCfg(
            func=torso_forward_lean,
            weight=LEAN_WEIGHT,
            params={
                "command_name": "twist",
                "target_rad": LEAN_TARGET_RAD,
                "full_speed": LEAN_FULL_SPEED,
                "std": LEAN_STD_RAD,
                "asset_cfg": SceneEntityCfg("robot", body_names=(TRUNK_BODY,)),
            },
        )

    if symmetric_default:
        robot_cfg = cfg.scene.entities["robot"]
        robot_cfg.init_state = dataclasses.replace(
            robot_cfg.init_state,
            joint_pos={**robot_cfg.init_state.joint_pos, **SYMMETRIC_HIP_YAW_DEFAULT},
        )

    if narrow_stance:
        cfg.rewards["hip_roll_spread"] = RewardTermCfg(
            func=hip_roll_excess,
            weight=HIP_ROLL_SPREAD_WEIGHT,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_hip_roll_joint",)),
                "limit": hip_roll_limit,
            },
        )

    if lift:
        wheels = SceneEntityCfg("robot", body_names=WHEEL_LINK_NAMES)
        schedules = LIFT_WEIGHT_SCHEDULES[lift]
        switch_air = LIFT_SWITCH_AIR_TIME_S[lift]
        staged = lift in ("staged", "staged_end")
        common = {
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "ramp_lo": LIFT_RAMP_SPEED[0],
            "ramp_hi": LIFT_RAMP_SPEED[1],
        }

        def schedule(name, w0, w1, it0, it1):
            cfg.curriculum[f"{name}_weight"] = CurriculumTermCfg(
                func=reward_weight_schedule,
                params={
                    "term_name": name,
                    "start_weight": w0,
                    "end_weight": w1,
                    "start_step": it0 * STEPS_PER_ITERATION,
                    "end_step": it1 * STEPS_PER_ITERATION,
                },
            )

        def param_schedule(term, param, v0, v1, it0, it1):
            cfg.curriculum[f"{term}_{param}"] = CurriculumTermCfg(
                func=reward_param_schedule,
                params={
                    "term_name": term,
                    "param": param,
                    "start_value": v0,
                    "end_value": v1,
                    "start_step": it0 * STEPS_PER_ITERATION,
                    "end_step": it1 * STEPS_PER_ITERATION,
                },
            )

        lift_terms = {
            "wheel_jump": (wheel_jump, {}),
            "wheel_single_stance": (
                wheel_single_stance,
                {"asset_cfg": wheels, "target_height": LIFT_TARGET_HEIGHT_M, "max_time": LIFT_MAX_SINGLE_STANCE_S},
            ),
            "wheel_support_switch": (
                wheel_support_switch,
                {
                    "asset_cfg": wheels,
                    "min_height": LIFT_SWITCH_MIN_HEIGHT_M,
                    "t_min": switch_air[0],
                    "t_max": switch_air[1],
                },
            ),
        }
        for name, (func, params) in lift_terms.items():
            w0, w1, it0, it1 = schedules[name]
            cfg.rewards[name] = RewardTermCfg(func=func, weight=w0, params={**common, **params})
            schedule(name, w0, w1, it0, it1)
        if lift_clock:
            assert clock, "lift_clock needs the gait clock (clock=True)"
            stance = cfg.rewards["wheel_single_stance"].params
            stance["clock_window"] = LIFT_CLOCK_WINDOW
            stance["clock_alpha"] = LIFT_CLOCK_ALPHA[0]
            if lift_clock == "compensated":
                stance["clock_gain"] = LIFT_CLOCK_GAIN_COMPENSATED
            param_schedule("wheel_single_stance", "clock_alpha", *LIFT_CLOCK_ALPHA)
        if lift != "early":
            cfg.rewards["wheel_height_excess"] = RewardTermCfg(
                func=wheel_height_excess,
                weight=LIFT_HEIGHT_EXCESS_WEIGHT,
                params={"asset_cfg": wheels, "limit": LIFT_HEIGHT_LIMIT_M, "scale": 0.02},
            )
        if staged:
            # Terms outside the lift set (height bound, forward lean) get their weight schedules here as well.
            for name in ("wheel_height_excess", "torso_forward_lean"):
                if name in cfg.rewards:
                    w0, w1, it0, it1 = schedules[name]
                    cfg.rewards[name].weight = w0
                    schedule(name, w0, w1, it0, it1)
            t_min = LIFT_SWITCH_MIN_AIR_TIME_STAGED if lift == "staged" else (switch_air[0], switch_air[0], 0, 1)
            param_schedule("wheel_support_switch", "t_min", *t_min)

    if single_wheel:
        twist_cmd.ranges.lin_vel_x = (0.0, SINGLE_WHEEL_MAX_SPEED)
        cfg.rewards["single_wheel_lift"] = RewardTermCfg(
            func=single_wheel_lift,
            weight=SINGLE_WHEEL_LIFT_WEIGHT,
            params={
                "target_height": SINGLE_WHEEL_TARGET_M,
                "lifted_cfg": SceneEntityCfg("robot", body_names=("left_wheel_link",)),
                "support_cfg": SceneEntityCfg("robot", body_names=("right_wheel_link",)),
            },
        )

    if foot_lift == "shaped":
        wheels = SceneEntityCfg("robot", body_names=WHEEL_LINK_NAMES)
        ramp_lo, ramp_hi = FOOT_LIFT_RAMP_SPEED
        common = {
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "ramp_lo": ramp_lo,
            "ramp_hi": ramp_hi,
            "target_height": FOOT_LIFT_TARGET_M,
            "asset_cfg": wheels,
        }
        cfg.rewards["wheel_lift"] = RewardTermCfg(
            func=wheel_lift, weight=FOOT_LIFT_SHAPED_WEIGHTS["lift"], params=dict(common)
        )
        cfg.rewards["wheel_swing_height"] = RewardTermCfg(
            func=wheel_swing_height,
            weight=FOOT_LIFT_SHAPED_WEIGHTS["swing_height"],
            params=dict(common),
        )
    elif foot_lift:
        wheels = SceneEntityCfg("robot", body_names=WHEEL_LINK_NAMES)
        ramp_lo, ramp_hi = FOOT_LIFT_RAMP_SPEED
        common = {
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "ramp_lo": ramp_lo,
            "ramp_hi": ramp_hi,
        }
        cfg.rewards["wheel_air_time"] = RewardTermCfg(
            func=wheel_air_time, weight=FOOT_LIFT_WEIGHTS["air_time"], params=dict(common)
        )
        cfg.rewards["wheel_clearance"] = RewardTermCfg(
            func=wheel_clearance,
            weight=FOOT_LIFT_WEIGHTS["clearance"],
            params={**common, "target_height": FOOT_LIFT_TARGET_M, "asset_cfg": wheels},
        )
        cfg.rewards["wheel_swing_height"] = RewardTermCfg(
            func=wheel_swing_height,
            weight=FOOT_LIFT_WEIGHTS["swing_height"],
            params={**common, "target_height": FOOT_LIFT_TARGET_M, "asset_cfg": wheels},
        )

    for reward_name in (
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "soft_landing",
    ):
        cfg.rewards.pop(reward_name, None)

    for obs_group in cfg.observations.values():
        obs_group.terms.pop("foot_height", None)
        obs_group.terms.pop("foot_air_time", None)

    assert cfg.curriculum is not None
    cfg.curriculum.pop("terrain_levels", None)
    cfg.curriculum.pop("soft_landing_weight", None)
    cfg.curriculum.pop("command_vel", None)

    if not NEMOV3_OBSERVE_WHEEL_ANGLE:
        for group_name in ("actor", "critic"):
            cfg.observations[group_name].terms["joint_pos"].params["asset_cfg"] = (
                SceneEntityCfg("robot", joint_names=position_joints)
            )

    if clock:
        cfg.commands["gait_clock"] = GaitClockCommandCfg(
            resampling_time_range=(1.0e9, 1.0e9), frequency_hz=GAIT_FREQUENCY_HZ
        )
        for group in cfg.observations.values():
            group.terms["gait_clock"] = ObservationTermCfg(
                func=mdp.generated_commands, params={"command_name": "gait_clock"}
            )

    cfg.episode_length_s = NEMOV3_TRAIN_EPISODE_LENGTH_S
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.terminations.pop("illegal_contact", None)
        # Random velocity kicks (incl. roll/pitch/yaw rates) every 1.5-4 s are
        # a training-time robustness perturbation. Kept in play but gated by
        # a "Push disturbance" checkbox in the Viser viewer, off by default.
        cfg.commands["play_controls"] = PlayControlsCommandCfg(
            resampling_time_range=(1.0e9, 1.0e9)
        )
        push_robot = cfg.events["push_robot"]
        push_robot.func = push_if_enabled
        push_robot.params["command_name"] = "play_controls"
        # viser's joystick GUI needs max > 0 on every axis (slider min is 0.1); its sliders span +-max.
        twist_cmd.ranges.lin_vel_y = (0.0, 0.1)
        if not turn:
            twist_cmd.ranges.ang_vel_z = (0.0, 0.1)

    if fixed_joints:
        _drop_fixed_joints(cfg, position_joints)

    if actuator_delay_steps:
        cfg.events["actuator_delay"] = EventTermCfg(
            func=sample_actuator_delay,
            mode="reset",
            params={"groups": NEMOV3_DELAY_GROUPS, "max_lag": actuator_delay_steps},
        )
    if trunk_mass_com:
        torso = SceneEntityCfg("robot", body_names=(TRUNK_BODY,))
        cfg.events["trunk_mass"] = EventTermCfg(
            func=mdp.dr.body_mass,
            mode="startup",
            params={"asset_cfg": torso, "ranges": TRUNK_MASS_ADD_KG, "operation": "add"},
        )
        if trunk_mass_com != "mass":
            cfg.events["trunk_com"] = EventTermCfg(
                func=mdp.dr.body_com_offset,
                mode="startup",
                params={"asset_cfg": torso, "ranges": TRUNK_COM_OFFSET_M, "operation": "add"},
            )

    if actuator_delay is not None:
        assert not actuator_delay_steps, "actuator_delay replaces actuator_delay_steps"
        cfg.events["actuator_delay"] = EventTermCfg(
            func=sample_actuator_delay,
            mode="reset",
            params={
                "groups": actuator_delay.groups,
                "max_lag": actuator_delay.max_lag,
                "cmd_groups": actuator_delay.cmd_groups,
                "cmd_delay_range_s": actuator_delay.cmd_delay_range_s,
            },
        )

    return cfg


def _drop_fixed_joints(cfg: ManagerBasedRlEnvCfg, position_joints: tuple[str, ...]) -> None:
    """Drop joint-name keys / patterns that match no remaining position joint (the welded joints) from the action
    scale and the upper-body posture reward, which list joints by name."""

    def active(pattern: str) -> bool:
        return any(re.fullmatch(pattern, name) for name in position_joints)

    action = cfg.actions["joint_pos"]
    action.scale = {k: v for k, v in action.scale.items() if active(k)}
    posture = cfg.rewards["upper_body_posture"].params
    posture["asset_cfg"].joint_names = tuple(p for p in posture["asset_cfg"].joint_names if active(p))
    for key in ("std_standing", "std_walking", "std_running"):
        posture[key] = {k: v for k, v in posture[key].items() if active(k)}
