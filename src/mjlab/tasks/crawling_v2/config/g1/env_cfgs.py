"""Unitree G1 crawling v2: reference-free actor + a commanded belly rest at low speed.

Builds on crawling_v1 (reference-free actor: proprioception + phase + speed command; asymmetric
critic keeps the full reference; library drives the imitation reward). v2 adds a "stop" regime:
when the sampled forward speed is below the stop threshold the env holds a low-effort PRONE rest
instead of crawling. Per stopped env the command freezes the phase (holds the current frame), the
imitation rewards + reference-relative terminations are gated off, and rest rewards (low torso
height, stillness, low actuator effort) drive the robot onto its belly. Getting back up on resume
is a learned transition to watch during training.
"""

from dataclasses import replace
from pathlib import Path

import mjlab
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.crawling_v2 import mdp
# Dotted imports (not `from ...mdp import rewards`) so these resolve to the crawling_v2 submodules
# and not the `mjlab.envs.mdp.{rewards,terminations}` attributes the mdp star-import shadows in.
import mjlab.tasks.crawling_v2.mdp.rewards as crawl_rewards
import mjlab.tasks.crawling_v2.mdp.terminations as crawl_terms
from mjlab.tasks.crawling_v2.mdp.commands import LibraryMotionCommandCfg
from mjlab.tasks.tracking.config.g1_custom_contactrich.env_cfgs import (
  unitree_g1_custom_contactrich_flat_tracking_env_cfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# The converted tracking-format crawl library (produced by scripts/library_to_npz.py).
MOTION_DIR = str(Path(mjlab.MJLAB_SRC_PATH).parent.parent / "trajectories" / "library_tracking")

# Commanded forward-speed range. Includes 0: speeds below STOP_SPEED_THRESHOLD are trained as a
# belly rest (phase frozen, imitation off, rest rewards on) rather than a slow crawl.
SPEED_RANGE = (0.0, 0.40)
STOP_SPEED_THRESHOLD = 0.08  # commanded speed below this means "stop / belly rest"
# In play mode, pin a single speed so the reference ghost shows one clean gait (set 0.0 to see rest).
PLAY_SPEED = 0.28
# Belly-rest reward targets (only active for stopped envs).
REST_TORSO_HEIGHT = 0.20  # target anchor (torso) height when prone [m]
REST_HEIGHT_STD = 0.15


def unitree_g1_crawling_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree G1 (29-DoF) flat crawling config with a reference-free actor."""
  cfg = unitree_g1_custom_contactrich_flat_tracking_env_cfg(
    has_state_estimation=False, play=play
  )

  # --- swap the single-clip motion command for the speed-indexed library command ---
  # Copy the (already play-adjusted) MotionCommandCfg fields into a LibraryMotionCommandCfg.
  old = cfg.commands["motion"]
  cfg.commands["motion"] = LibraryMotionCommandCfg(
    entity_name=old.entity_name,
    resampling_time_range=old.resampling_time_range,
    debug_vis=old.debug_vis,
    pose_range=old.pose_range,
    velocity_range=old.velocity_range,
    joint_position_range=old.joint_position_range,
    adaptive_kernel_size=old.adaptive_kernel_size,
    adaptive_lambda=old.adaptive_lambda,
    adaptive_uniform_ratio=old.adaptive_uniform_ratio,
    adaptive_alpha=old.adaptive_alpha,
    sampling_mode=old.sampling_mode,
    anchor_body_name=old.anchor_body_name,
    body_names=old.body_names,
    viz=old.viz,
    motion_dir=MOTION_DIR,
    speed_command_range=((PLAY_SPEED, PLAY_SPEED) if play else SPEED_RANGE),
    stop_speed_threshold=STOP_SPEED_THRESHOLD,
    # motion_file is unused by the library loader, but train/play's tracking-task guard
    # (isinstance MotionCommandCfg) requires an existing path -> point it at the motion dir.
    motion_file=MOTION_DIR,
  )

  # --- observations: tell the policy the commanded speed (+ a phase clock), no noise ---
  speed_obs = ObservationTermCfg(func=mdp.commanded_speed, params={"command_name": "motion"})
  phase_obs = ObservationTermCfg(func=mdp.motion_phase, params={"command_name": "motion"})
  for group in ("actor", "critic"):
    cfg.observations[group].terms["commanded_speed"] = replace(speed_obs)
    cfg.observations[group].terms["motion_phase"] = replace(phase_obs)

  # --- v1: strip the target-motion reference from the ACTOR only (keep it in the critic) ---
  # The actor now sees only proprioception + phase + commanded speed; the critic keeps the full
  # reference (asymmetric). The library still drives learning via the imitation reward.
  for ref_term in ("command", "motion_anchor_ori_b"):
    cfg.observations["actor"].terms.pop(ref_term, None)

  # Restore a reference-free orientation signal: projected gravity = the robot's own IMU tilt
  # (roll/pitch), replacing the orientation lost with motion_anchor_ori_b. Yaw is left to reward.
  cfg.observations["actor"].terms["projected_gravity"] = ObservationTermCfg(
    func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
  )

  # --- reward: light direct forward-speed tracking to hit the (continuous) commanded speed ---
  cfg.rewards["forward_speed"] = RewardTermCfg(
    func=mdp.forward_speed_tracking,
    weight=0.3,
    params={"command_name": "motion", "std": 0.1, "axis": 1},
  )

  # --- v2 belly rest: below the stop threshold, hold a low-effort prone pose instead of crawling.
  # (1) gate imitation rewards off for stopped envs (the frozen crawl pose stops paying out),
  # (2) gate the reference-relative terminations off (lowering to the belly must not end the
  #     episode), (3) add rest rewards driving a still, low, low-effort hold. Phase-freeze and the
  #     is_stopped mask live in the command (commands.py).
  imitation_terms = (
    "motion_global_root_pos", "motion_global_root_ori", "motion_body_pos",
    "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel",
  )
  for name in imitation_terms:
    cfg.rewards[name].func = crawl_rewards.gate_moving(cfg.rewards[name].func)
  for name in ("anchor_pos", "anchor_ori", "ee_body_pos"):
    cfg.terminations[name].func = crawl_terms.gate_moving(cfg.terminations[name].func)

  cfg.rewards["rest_height"] = RewardTermCfg(
    func=crawl_rewards.rest_base_height,
    weight=2.0,
    params={"command_name": "motion", "target_height": REST_TORSO_HEIGHT, "std": REST_HEIGHT_STD},
  )
  cfg.rewards["rest_stillness"] = RewardTermCfg(
    func=crawl_rewards.rest_stillness, weight=-0.5, params={"command_name": "motion"}
  )
  cfg.rewards["rest_effort"] = RewardTermCfg(
    func=crawl_rewards.rest_effort, weight=-1e-5, params={"command_name": "motion"}
  )

  return cfg
