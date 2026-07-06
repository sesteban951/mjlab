"""Unitree G1 crawling (velocity-conditioned gait-library tracking) environment config.

Starts from the contact-rich G1 tracking env (whole-body friction, custom DR, HOME init,
no state estimation) and swaps its single-clip motion command for the speed-indexed gait-library
command, adding the commanded-speed / phase observations and a small forward-speed reward.
"""

from dataclasses import replace
from pathlib import Path

import mjlab
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.crawling_v0 import mdp
from mjlab.tasks.crawling_v0.mdp.commands import LibraryMotionCommandCfg
from mjlab.tasks.tracking.config.g1_custom_contactrich.env_cfgs import (
  unitree_g1_custom_contactrich_flat_tracking_env_cfg,
)

# The converted tracking-format crawl library (produced by scripts/library_to_npz.py).
MOTION_DIR = str(Path(mjlab.MJLAB_SRC_PATH).parent.parent / "trajectories" / "library_tracking")

# Commanded forward-speed range (matches the library span 0.08-0.40 m/s).
SPEED_RANGE = (0.08, 0.40)
# In play mode, pin a single speed so the reference ghost shows one clean gait.
PLAY_SPEED = 0.28


def unitree_g1_crawling_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree G1 (29-DoF) flat crawling gait-library tracking config."""
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

  # --- reward: light direct forward-speed tracking to hit the (continuous) commanded speed ---
  cfg.rewards["forward_speed"] = RewardTermCfg(
    func=mdp.forward_speed_tracking,
    weight=0.3,
    params={"command_name": "motion", "std": 0.1, "axis": 1},
  )

  return cfg
