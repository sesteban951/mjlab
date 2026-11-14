"""Unitree Go2 custom flat tracking environment configuration.

Ports TrackerLab SuperMimic's Go2 gait-imitation task
(https://github.com/Renforce-Dynamics/trackerlab-supermimic) onto mjlab's
BeyondMimic tracking MDP, trained with mjlab's PPO instead of SuperMimic's world
model. SuperMimic's tracking reward is BeyondMimic's, so the reward terms and
weights carry over unchanged; the Go2 differences are:

- Actuators: SuperMimic's DC motors (kp 25, kd 0.5, 23.5 Nm, 30 rad/s), a
  hips-at-zero default pose and a uniform 0.25 action scale.
- Tracked bodies: the base plus every hip, thigh and calf (13 bodies), anchored
  on the base.
- Tighter kernels for the smaller robot: body position std 0.15 m, body linear
  velocity std 0.5 m/s, anchor position std 0.2 m.
- Joint-limit penalty -2, no end-effector termination, and a 0.16 m
  anchor-height termination.
- Not from SuperMimic: self-collisions are enabled with a small -1 penalty
  (10 N threshold), for future motions that may bring the legs together.
- SuperMimic's smaller reference-state-init noise.
- Not from SuperMimic: the shared custom Go2 domain randomization
  (``unitree_go2.custom_dr``), mirroring the custom G1 envs: 0-40 ms command
  delay, armature x0.95-1.05, base mass/inertia x0.95-1.05 and size-scaled
  pushes, on top of SuperMimic's COM, encoder and friction DR. Like SuperMimic,
  PD gains are not randomized.
- Actor observations without anchor position or base linear velocity, matching
  SuperMimic's proprioceptive "relative orientation" observation.
"""

from mjlab.asset_zoo.robots.unitree_go2.custom_dr import (
  add_custom_go2_actuator_delay,
  add_custom_go2_dr,
)
from mjlab.asset_zoo.robots.unitree_go2.go2_constants_supermimic import (
  GO2_SUPERMIMIC_ACTION_SCALE,
  get_go2_supermimic_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

GO2_ANCHOR_BODY = "base_link"
GO2_TRACKED_BODIES = (
  "base_link",
  *(
    f"{leg}_{link}"
    for leg in ("FL", "FR", "RL", "RR")
    for link in ("hip", "thigh", "calf")
  ),
)
GO2_FOOT_GEOMS = tuple(f"{leg}_foot_collision" for leg in ("FL", "FR", "RL", "RR"))


def unitree_go2_custom_flat_tracking_env_cfg(
  has_state_estimation: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the SuperMimic-style Unitree Go2 flat terrain tracking config."""
  cfg = make_tracking_env_cfg()

  # The tracking task caps the solver at 10/20 iterations for throughput; MuJoCo
  # Warp 3.16 would otherwise print a line every step a world hits that cap.
  cfg.sim.warn_solver_overflow = False

  cfg.scene.entities = {"robot": get_go2_supermimic_robot_cfg()}
  # Self-collision sensor for the penalty below (not in SuperMimic).
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=GO2_ANCHOR_BODY, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=GO2_ANCHOR_BODY, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO2_SUPERMIMIC_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = GO2_ANCHOR_BODY
  motion_cmd.body_names = GO2_TRACKED_BODIES
  # SuperMimic's Go2 reference-state-init noise.
  motion_cmd.pose_range = {
    "x": (-0.02, 0.02),
    "y": (-0.02, 0.02),
    "z": (-0.005, 0.005),
    "roll": (-0.05, 0.05),
    "pitch": (-0.05, 0.05),
    "yaw": (-0.1, 0.1),
  }
  motion_cmd.velocity_range = {
    "x": (-0.2, 0.2),
    "y": (-0.1, 0.1),
    "z": (-0.1, 0.1),
    "roll": (-0.2, 0.2),
    "pitch": (-0.2, 0.2),
    "yaw": (-0.3, 0.3),
  }
  motion_cmd.joint_position_range = (-0.05, 0.05)

  # Domain randomization: the shared custom Go2 base (SuperMimic's COM, encoder
  # bias and friction and no gain randomization, plus armature, base mass and
  # size-scaled pushes). Pushes are off in play mode, as in the G1 tracking task.
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = GO2_FOOT_GEOMS
  add_custom_go2_dr(cfg, push=not play)

  # Command delay, training only (same 0-40 ms as the custom G1 envs).
  if not play:
    add_custom_go2_actuator_delay(cfg, min_delay_sec=0.0, max_delay_sec=0.04)

  # Rewards: BeyondMimic's terms and weights with SuperMimic's Go2 kernels.
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.2
  cfg.rewards["motion_body_pos"].params["std"] = 0.15
  cfg.rewards["motion_body_lin_vel"].params["std"] = 0.5
  cfg.rewards["joint_limit"].weight = -2.0
  # SuperMimic has no self-collision term. Keep a small one (the G1 uses -10) so
  # future motions that bring the legs together are discouraged, not forbidden.
  cfg.rewards["self_collisions"].weight = -1.0

  # Terminations: anchor height and tilt only.
  cfg.terminations["anchor_pos"].params["threshold"] = 0.16
  cfg.terminations.pop("ee_body_pos", None)

  cfg.viewer.body_name = GO2_ANCHOR_BODY
  cfg.viewer.distance = 1.5

  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
