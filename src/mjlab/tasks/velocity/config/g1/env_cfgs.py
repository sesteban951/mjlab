"""Unitree G1 velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import HOME_KEYFRAME
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


#########################################################################
# VANILLA FLAT CONFIG
#########################################################################


def unitree_g1_flat_env_cfg_custom(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_flat_env_cfg(play=play)

  # Remove base_lin_vel from actor — not available on real hardware.
  del cfg.observations["actor"].terms["base_lin_vel"]

  # Add single-leg gait phase clock to actor and critic (2 values).
  desired_period = 1.0  # seconds per gait cycle
  for group in ("actor", "critic"):
    cfg.observations[group].terms["gait_phase"] = ObservationTermCfg(
      func=mdp.phase,
      params={"period": desired_period},
    )

  site_names = ("left_foot", "right_foot")

  # Replace foot_slip with contact_no_vel (3D velocity, no command gating).
  del cfg.rewards["foot_slip"]
  cfg.rewards["contact_no_vel"] = RewardTermCfg(
    func=mdp.reward_contact_no_vel,
    weight=-0.2,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
    },
  )

  # Penalize foot tilt during stance (enforce flat foot contact).
  cfg.rewards["flat_foot"] = RewardTermCfg(
    func=mdp.flat_foot_contact,
    weight=-0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg(
        "robot",
        body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
      ),
    },
  )

  # Encourage feet to lift off the ground when moving.
  cfg.rewards["air_time"].weight = 0.1
  cfg.rewards["air_time"].params["threshold_max"] = 0.6

  # Reward correct foot contact timing with gait phase (unitree_rl_gym style).
  cfg.rewards["gait_phase_contact"] = RewardTermCfg(
    func=mdp.gait_phase_contact,
    weight=0.18,
    params={
      "sensor_name": "feet_ground_contact",
      "period": desired_period,
      "offset": 0.5,
      "stance_ratio": 0.55,
    },
  )

  # Override velocity curriculum for reasonable walking speeds.
  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": 0, "lin_vel_x": (-0.5, 0.5), "ang_vel_z": (-0.25, 0.25)},
        {"step": 5000 * 24, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.50, 0.50)},
        {"step": 10000 * 24, "lin_vel_x": (-1.5, 1.5), "ang_vel_z": (-0.75, 0.75)},
        {"step": 15000 * 24, "lin_vel_x": (-2.0, 2.0), "ang_vel_z": (-1.0, 1.0)},
      ],
    },
  )

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 1.5)
    twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-0.75, 0.75)

  return cfg


#########################################################################
# UNITREE RL GYM FLAT CONFIG
#########################################################################


def unitree_g1_flat_env_cfg_unitree_rl_gym(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """G1 flat velocity config matching unitree_rl_gym rewards/terminations.

  Closely replicates the observation, reward, termination, and command
  structure from unitree_rl_gym's G1RoughCfg on flat terrain.
  """
  cfg = unitree_g1_flat_env_cfg(play=play)

  # Use HOME_KEYFRAME: unitree leg defaults + custom arm/waist defaults.
  cfg.scene.entities["robot"].init_state = HOME_KEYFRAME

  site_names = ("left_foot", "right_foot")
  gait_period = 0.8
  gait_offset = 0.5

  ##
  # Observations
  ##

  # Actor: remove base_lin_vel (not available on hardware).
  del cfg.observations["actor"].terms["base_lin_vel"]

  # Add single-leg gait phase clock to actor and critic (2 values).
  for group in ("actor", "critic"):
    cfg.observations[group].terms["gait_phase"] = ObservationTermCfg(
      func=mdp.phase,
      params={"period": gait_period},
    )

  ##
  # Rewards — match unitree_rl_gym G1 scales and functions.
  # All weights are raw (the framework handles dt multiplication).
  ##

  cfg.rewards = {
    "tracking_lin_vel": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": 0.5},
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=0.5,
      params={"command_name": "twist", "std": 0.5},
    ),
    "ang_vel_xy": RewardTermCfg(
      func=mdp.reward_ang_vel_xy,
      weight=-0.05,
    ),
    "orientation": RewardTermCfg(
      func=envs_mdp.flat_orientation_l2,
      weight=-1.0,
    ),
    "dof_acc": RewardTermCfg(
      func=envs_mdp.joint_acc_l2,
      weight=-2.5e-7,
    ),
    "dof_vel": RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=-1e-3,
    ),
    "action_rate": RewardTermCfg(
      func=envs_mdp.action_rate_l2,
      weight=-0.01,
    ),
    "dof_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-5.0,
    ),
    "alive": RewardTermCfg(
      func=envs_mdp.is_alive,
      weight=0.15,
    ),
    "contact": RewardTermCfg(
      func=mdp.gait_phase_contact,
      weight=0.18,
      params={
        "sensor_name": "feet_ground_contact",
        "period": gait_period,
        "offset": gait_offset,
        "stance_ratio": 0.55,
      },
    ),
    "contact_no_vel": RewardTermCfg(
      func=mdp.reward_contact_no_vel,
      weight=-0.2,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      },
    ),
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": {".*": 0.05},
        "std_walking": {
          # Lower body.
          r".*hip_pitch.*": 0.3,
          r".*hip_roll.*": 0.15,
          r".*hip_yaw.*": 0.15,
          r".*knee.*": 0.35,
          r".*ankle_pitch.*": 0.25,
          r".*ankle_roll.*": 0.1,
          # Waist.
          r".*waist_yaw.*": 0.2,
          r".*waist_roll.*": 0.08,
          r".*waist_pitch.*": 0.1,
          # Arms.
          r".*shoulder_pitch.*": 0.15,
          r".*shoulder_roll.*": 0.15,
          r".*shoulder_yaw.*": 0.1,
          r".*elbow.*": 0.15,
          r".*wrist.*": 0.3,
        },
        "std_running": {
          # Lower body.
          r".*hip_pitch.*": 0.5,
          r".*hip_roll.*": 0.2,
          r".*hip_yaw.*": 0.2,
          r".*knee.*": 0.6,
          r".*ankle_pitch.*": 0.35,
          r".*ankle_roll.*": 0.15,
          # Waist.
          r".*waist_yaw.*": 0.3,
          r".*waist_roll.*": 0.08,
          r".*waist_pitch.*": 0.2,
          # Arms.
          r".*shoulder_pitch.*": 0.5,
          r".*shoulder_roll.*": 0.2,
          r".*shoulder_yaw.*": 0.15,
          r".*elbow.*": 0.35,
          r".*wrist.*": 0.3,
        },
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
  }

  ##
  # Terminations — match unitree_rl_gym: separate roll/pitch limits.
  ##

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "bad_orientation": TerminationTermCfg(
      func=mdp.bad_orientation_rpy,
      params={"roll_limit": 0.8, "pitch_limit": 1.0},
    ),
  }

  ##
  # Commands — unitree uses 10s fixed resampling, no curriculum.
  ##

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.resampling_time_range = (10.0, 10.0)
  twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
  twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  ##
  # Events — adjust push robot to match unitree (5s interval, 1.5 m/s).
  ##

  cfg.events["push_robot"] = EventTermCfg(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(5.0, 5.0),
    params={
      "velocity_range": {
        "x": (-1.5, 1.5),
        "y": (-1.5, 1.5),
      },
    },
  )

  # No velocity or terrain curriculum.
  cfg.curriculum = {}

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 1.5)
    twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
