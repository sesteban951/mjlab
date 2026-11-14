"""Unitree Go2 custom velocity environment configuration.

Vendored from unitree_rl_mjlab; uses the self-contained velocity_custom base + mdp,
the same base the G1 custom velocity env is built on.
"""

from mjlab.asset_zoo.robots.unitree_go2.go2_constants import get_go2_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RayCastSensorCfg
from mjlab.tasks.velocity_custom import mdp
from mjlab.tasks.velocity_custom.mdp import (
  BodyPoseCommandCfg,
  UniformVelocityCommandCfg,
)
from mjlab.tasks.velocity_custom.velocity_env_cfg import make_velocity_env_cfg


def unitree_go2_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  # Solver is capped at 10/20 iterations for throughput; MuJoCo Warp 3.16 would
  # otherwise print a line every step a world hits that cap.
  cfg.sim.warn_solver_overflow = False

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Set raycast sensor frame to Go2 base_link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  # The vendored base raised action_rate_l2 to -0.2 for the G1; Unitree tuned the
  # quadruped at -0.05, so restore that here.
  cfg.rewards["action_rate_l2"].weight = -0.05

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
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


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = unitree_go2_rough_env_cfg(play=play)

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
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


##
# Velocity + body pose (height, roll, pitch).
##

# Measured on the Go2 model with the feet under the hips: the default stance holds
# the base at 0.295 m, and the legs reach 0.12 to 0.42 m without anything but the
# feet touching the floor. Commands stay inside that with margin for swing.
GO2_NOMINAL_HEIGHT = 0.295
GO2_LEG_HEIGHT_LIMITS = (0.13, 0.40)
GO2_FOOT_HALF_LENGTH = 0.193
GO2_FOOT_HALF_WIDTH = 0.115


def unitree_go2_pose_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Go2 flat velocity tracking plus commanded base height, roll and pitch.

  The actor sees six commands: the twist (vx, vy, wz) and the body pose (height,
  roll, pitch). Roll and pitch apply only while the twist is zero; height applies
  while walking too, with the lowest heights reserved for slow speeds.
  """
  cfg = unitree_go2_flat_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  # Posing happens while standing, so stand far more often than the base 5%.
  twist_cmd.rel_standing_envs = 0.3

  cfg.commands["body_pose"] = BodyPoseCommandCfg(
    entity_name="robot",
    twist_command_name="twist",
    resampling_time_range=(3.0, 8.0),
    nominal_height=GO2_NOMINAL_HEIGHT,
    height_range=(0.15, 0.37),
    roll_range=(-0.35, 0.35),
    pitch_range=(-0.4, 0.4),
    rel_nominal_envs=0.25,
    walking_height_range=(0.15, 0.34),
    crouch_speed_range=(0.6, 2.0),
    fast_min_height=0.24,
    leg_height_limits=GO2_LEG_HEIGHT_LIMITS,
    foot_half_length=GO2_FOOT_HALF_LENGTH,
    foot_half_width=GO2_FOOT_HALF_WIDTH,
  )

  # Observations: the pose command right after the twist; base height for the
  # critic only.
  pose_obs = ObservationTermCfg(
    func=mdp.generated_commands, params={"command_name": "body_pose"}
  )
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    items = list(terms.items())
    at = [k for k, _ in items].index("command") + 1
    cfg.observations[group].terms = dict(
      items[:at] + [("body_pose", pose_obs)] + items[at:]
    )
  cfg.observations["critic"].terms["base_height"] = ObservationTermCfg(
    func=mdp.base_height
  )

  # Rewards: track the pose, and stop the base terms from fighting it.
  cfg.rewards["track_base_height"] = RewardTermCfg(
    func=mdp.track_base_height,
    weight=1.0,
    params={"command_name": "body_pose", "std": 0.04},
  )
  cfg.rewards["track_body_attitude"] = RewardTermCfg(
    func=mdp.track_body_attitude,
    weight=1.0,
    params={"command_name": "body_pose", "std": 0.2},
  )
  # Flat-body penalty becomes a penalty on tilt away from the commanded attitude.
  cfg.rewards["body_orientation_l2"] = RewardTermCfg(
    func=mdp.body_orientation_target_l2,
    weight=-1.0,
    params={"command_name": "body_pose"},
  )
  # stand_still pins every joint to default while standing, which is exactly when
  # the robot poses.
  cfg.rewards.pop("stand_still", None)
  # Posture keeps only the hips near default; thighs and calves set the pose.
  hip_std = {r".*_hip_joint": 0.15}
  pose_params = cfg.rewards["pose"].params
  pose_params["asset_cfg"] = SceneEntityCfg("robot", joint_names=(r".*_hip_joint",))
  pose_params["std_standing"] = hip_std
  pose_params["std_walking"] = hip_std
  pose_params["std_running"] = {r".*_hip_joint": 0.2}
  cfg.rewards["pose"].weight = 0.5
  # Swing clearance scales with the commanded height (0.09 m at nominal).
  clearance = cfg.rewards["foot_clearance"]
  clearance.func = mdp.feet_clearance_height_scaled
  clearance.params = {
    "pose_command_name": "body_pose",
    "height_ratio": 0.3,
    "command_name": "twist",
    "command_threshold": 0.1,
    "asset_cfg": clearance.params["asset_cfg"],
  }

  return cfg
