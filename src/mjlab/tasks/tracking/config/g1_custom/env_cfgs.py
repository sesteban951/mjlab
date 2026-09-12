"""Unitree G1 custom flat tracking environment configuration.

Applies the unitree_rl_mjlab HOME initial pose, the shared custom DR base
(see ``custom_dr.add_custom_g1_dr``), and randomized actuator command delay
(training only) on top of mjlab's tracking environment.
"""

from mjlab.asset_zoo.robots.unitree_g1.custom_dr import (
  add_custom_g1_actuator_delay,
  add_custom_g1_dr,
)
from mjlab.asset_zoo.robots.unitree_g1.custom_rewards import (
  add_custom_g1_action_rate_split,
)
from mjlab.asset_zoo.robots.unitree_g1.g1_constants_mode11 import (
  G1_MODE11_ACTION_SCALE,
  G1_MODE11_ARTICULATION,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

# Initial pose matching unitree_rl_mjlab's HOME keyframe (mjlab's stock 29-DoF G1
# inits from a knees-bent pose with different arm angles).
UNITREE_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.8),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
  },
  joint_vel={".*": 0.0},
)


def unitree_g1_custom_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a customizable Unitree G1 (29-DoF) flat terrain tracking config.

  Starts from the stock G1 tracking configuration, applies the unitree_rl_mjlab
  HOME initial pose, and layers on the shared custom DR base.
  """
  cfg = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation, play=play
  )

  # Use unitree_rl_mjlab's HOME initial pose instead of mjlab's knees-bent pose.
  cfg.scene.entities["robot"].init_state = UNITREE_HOME_KEYFRAME

  # Match the lab robot's mode_machine 11 hardware. Mode 11 reuses g1.xml unchanged; only the
  # actuator table moves hip pitch from the 14.3:1 to the 22.5:1 gearbox (7520_22), which also
  # changes its PD gains and action scale (0.548 -> 0.351). Applied before the actuator delay
  # below so the delay is baked onto the mode-11 actuators. Stock G1 tasks are unaffected.
  cfg.scene.entities["robot"].articulation = G1_MODE11_ARTICULATION
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_MODE11_ACTION_SCALE

  # Apply the shared custom DR base
  add_custom_g1_dr(cfg)

  # Split the single action-rate penalty into limb (arms+legs) and waist groups.
  add_custom_g1_action_rate_split(cfg)

  # Add actuator command delay
  if not play:
    add_custom_g1_actuator_delay(cfg, min_delay_sec=0.0, max_delay_sec=0.04)

  return cfg
