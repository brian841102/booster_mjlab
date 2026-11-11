"""SuperMimic motion-tracking environment configuration.

The mjlab env provides the scene (robot + flat ground), the physics settings and the startup domain randomization
(robot friction, torso CoM); ``SuperMimicEnv`` drives it directly: it sets the joint targets, steps the physics and
resets episodes to reference states. Rewards, terminations and observations are the analytic heads of
``mdp/tracking.py`` -- the world model is trained through them -- so the reward / termination / observation managers
stay empty. Every randomized quantity the world model must know is in its sim context xi: [friction, joint offsets,
CoM offset], and with ``SuperMimicCfg.sim2real`` also the torso mass / inertia, PD gains, actuator delays and ground
contact (``mdp/tracking.py``, xi layout).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from booster_mjlab.tasks.velocity.mdp.motor_dynamics import ActuatorDelayCfg
from booster_mjlab.terrains.contact import terrain_collisions


@dataclass
class SuperMimicSim2RealCfg:
    """Sim-to-real additions (the ``-Sim2Real-`` tasks). The startup randomization (PD gains, torso mass / inertia,
    ground contact) is set as mjlab events by the task; ``SuperMimicEnv`` reads the drawn values back into xi."""

    actuator_delay: ActuatorDelayCfg | None = None
    """Per-episode command lag per actuator group (AMP Exp 85+ merged delay); the lag of every joint goes into xi."""
    wheel_speed_limit: float | None = None
    """Clamp of the wheel speed target [rad/s] (the hardware runner clamps at 25)."""
    push_interval_s: tuple[float, float] | None = None
    """Random horizontal root velocity kicks every U(interval) s; the pushed transition ends world-model windows."""
    push_vel_xy: float = 0.2
    """Kick: uniform +- per horizontal axis [m/s]."""
    mass_range: tuple[float, float] = (-1.0, 1.0)
    """Torso mass offsets the task draws [kg] (xi normalization)."""
    kp_range: tuple[float, float] = (0.8, 1.2)
    kd_range: tuple[float, float] = (0.8, 1.2)
    """PD gain scales the task draws (xi normalization)."""
    solref_ranges: tuple[tuple[float, float], ...] = ((0.006, 0.03), (0.95, 1.05))
    solimp_ranges: tuple[tuple[float, float], ...] = ((0.88, 0.92), (0.94, 0.99), (0.003, 0.01))
    """Ground contact solref (2) / solimp (first 3) ranges the task draws (xi normalization)."""


@dataclass
class SuperMimicCfg:
    """What to track and how it is scored (set per robot in ``config/<robot>/env_cfgs.py``)."""

    motion_dir: str = ""
    """Directory of clip CSVs; every clip in it is trained."""
    csv_joint_names: tuple[str, ...] = ()
    """Joint column order of the clip CSVs."""
    tracked_bodies: tuple[str, ...] = ()
    """Bodies of the reward and the critic; the first is the root."""
    velocity_joints: tuple[str, ...] = ()
    """Velocity-controlled joints (wheels): the action is a speed target, their angle is not observed or offset."""
    action_scale: float = 0.5
    """rad per action unit (position joints): target = default + offset + scale * a."""
    velocity_action_scale: float = 10.0
    """rad/s per action unit (velocity joints)."""
    lift_points: tuple[tuple[str, tuple[float, float, float]], ...] = ()
    """(tracked body, offset in its frame) points whose world height gets its own reward term, e.g. wheel centres:
    lifting a wheel 5 cm barely moves the averaged body-position error, so without it a policy keeps both wheels down."""
    lift_std: float = 0.03
    slip_points: tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...] = ()
    """(tracked body, wheel centre in its frame, wheel axle in its frame): the wheel-slip penalty's wheels."""
    wheel_radius: float = 0.075
    w_wheel_slip: float = 0.0
    """Per (m/s)^2 of wheel-centre velocity along the axle while the wheel is on the ground (sideways sliding; a
    rolling wheel has none). 0 = off (exp1-3); tools/nemov3_finetune_supermimic.md, 排查 2 / exp4."""
    length_scale: float = 1.0
    """Robot size relative to G1; scales the position reward widths."""
    term_height: float = 0.25
    term_tilt: float = 0.8
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    w_lift_floor: float = -10.0
    lift_floor_on: float = 0.02
    soft_limit_factor: float = 0.9
    max_episode_steps: int = 500
    reset_noise: bool = True
    """Perturb the reference state an episode starts from (training only)."""
    friction_range: tuple[float, float] = (0.3, 1.2)
    """Robot geom friction per env (``robot_friction`` startup event); 1.0 in evaluation."""
    friction_geoms: tuple[str, ...] = (".*",)
    """Robot geoms whose friction is randomized (one shared coefficient per env)."""
    joint_offset_range: float = 0.01
    """Per-episode joint target offsets (uniform +-, position joints), observed as a calibration error."""
    com_body: str = ""
    com_range: tuple[float, float, float] = (0.025, 0.05, 0.05)
    """Torso CoM offset per env (``torso_com`` startup event)."""
    velocity_limits: tuple[tuple[str, float], ...] = ()
    """(joint regex, rad/s) hardware joint speed limits: not simulated, written into the deployment contract."""
    joint_vel_noise: float = 0.5
    """Observation noise on the joint velocities (uniform +-, rad/s)."""
    sim2real: SuperMimicSim2RealCfg | None = None


@dataclass(kw_only=True)
class SuperMimicEnvCfg(ManagerBasedRlEnvCfg):
    supermimic: SuperMimicCfg = field(default_factory=SuperMimicCfg)
    eval_mode: bool = False
    """Evaluation: friction 1.0, no randomization, no reset noise, episodes end only at the clip end or a fall."""


def make_supermimic_env_cfg(play: bool = False) -> SuperMimicEnvCfg:
    """Robot-agnostic part; the robot entity, robot geoms and the SuperMimicCfg are set per robot."""
    cfg = SuperMimicEnvCfg(
        decimation=4,
        scene=SceneCfg(
            num_envs=4096,
            terrain=TerrainEntityCfg(terrain_type="plane", collisions=terrain_collisions()),
        ),
        # 200 Hz physics, 50 Hz control; the solver settings of the Isaac Lab / Newton MuJoCo-Warp original
        sim=SimulationCfg(
            njmax=500,
            nconmax=128,
            mujoco=MujocoCfg(timestep=0.005, iterations=100, ls_iterations=50, tolerance=1e-6),
        ),
        events={
            "robot_friction": EventTermCfg(
                mode="startup",
                func=dr.geom_friction,
                params={
                    "asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
                    "operation": "abs",
                    "ranges": (0.3, 1.2),
                    "shared_random": True,  # one coefficient per env, xi[0]
                },
            ),
            "torso_com": EventTermCfg(
                mode="startup",
                func=dr.body_com_offset,
                params={"asset_cfg": SceneEntityCfg("robot", body_names=()), "operation": "add", "ranges": {}},
            ),
        },
        episode_length_s=1e6,  # episodes are ended by SuperMimicEnv
        eval_mode=play,
    )
    if play:
        cfg.events = {}
        cfg.supermimic.reset_noise = False
    return cfg
