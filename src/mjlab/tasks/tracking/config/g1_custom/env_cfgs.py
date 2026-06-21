"""Unitree G1 custom flat tracking environment configuration.

Applies the unitree_rl_mjlab parameters (HOME initial pose and wider base-COM
randomization) on top of mjlab's tracking environment.
"""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
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

  Starts from the stock G1 tracking configuration and applies the
  unitree_rl_mjlab parameters (initial pose and base-COM randomization).
  """
  cfg = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation, play=play
  )

  # Use unitree_rl_mjlab's HOME initial pose instead of mjlab's knees-bent pose.
  cfg.scene.entities["robot"].init_state = UNITREE_HOME_KEYFRAME

  # Widen base-COM x randomization to match unitree_rl_mjlab (mjlab uses +/-0.025).
  cfg.events["base_com"].params["ranges"][0] = (-0.05, 0.05)

  return cfg
