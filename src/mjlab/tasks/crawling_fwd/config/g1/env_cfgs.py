"""Unitree G1 crawling (forward + idle stop): twist-conditioned gait-library tracking.

Contact-rich G1 tracking env + a twist-indexed gait-library command with a REFERENCE-FREE actor.
Each library clip is a periodic crawl gait labelled with a planar twist ``[vx, vy, wz]``; the command
samples a target twist and snaps to the nearest clip. A fraction of resamples (``REL_STATIC_ENVS``,
mirroring the velocity task's standing envs) command a zero twist, which snaps to the static idle
clip so the robot holds the idle pose. Twist resampling is timer-driven and keeps the shared phase
clock (no mid-episode teleport), so the robot physically transitions between twists; RSI happens
only at episode reset. The imitation reward tracks the selected clip (that teaches the crawl) and
the critic keeps the full reference (asymmetric), but the deployed actor sees only proprioception +
phase + the commanded twist.
"""

from dataclasses import replace
from pathlib import Path

import mjlab
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.crawling_fwd import mdp
from mjlab.tasks.crawling_fwd.mdp.commands import LibraryMotionCommandCfg
from mjlab.tasks.tracking.config.g1_custom_contactrich.env_cfgs import (
  unitree_g1_custom_contactrich_flat_tracking_env_cfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# The converted tracking-format crawl library (produced by scripts/library_to_npz.py).
MOTION_DIR = str(
  Path(mjlab.MJLAB_SRC_PATH).parent.parent
  / "trajectories"
  / "crawl_ff_loop_180_R_001__A229_tracking"
)

# Commanded twist [(vx_lo, vx_hi), (vy..), (wz..)]. Moving commands span the forward library
# (~0.11-0.26 m/s); vy/wz stay 0 until the grid adds those clips. The idle/static pose is not part
# of this range -- it is sampled separately via REL_STATIC_ENVS (see below).
TWIST_RANGE = ((0.109, 0.264), (0.0, 0.0), (0.0, 0.0))
# In play mode, pin a single twist so the reference ghost shows one clean gait.
PLAY_TWIST = ((0.18, 0.18), (0.0, 0.0), (0.0, 0.0))
# Resample the commanded twist on this timer (seconds), mirroring the velocity task. Because all
# gaits share one period, a resample keeps the phase clock and just snaps the clip to the new twist.
RESAMPLING_TIME_RANGE = (3.0, 8.0)
# Fraction of resamples that command a zero twist -> the static idle clip (velocity's standing envs).
REL_STATIC_ENVS = 0.05


def unitree_g1_crawling_fwd_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree G1 (29-DoF) twist-conditioned crawling config with an idle-pose stop."""
  cfg = unitree_g1_custom_contactrich_flat_tracking_env_cfg(
    has_state_estimation=False, play=play
  )

  # Contact-rich crawling puts the whole body on the ground (condim=3 -> 3 constraint rows per
  # contact), so the per-env active-constraint count (nefc) exceeds the mujoco-warp default njmax
  # and constraints get dropped ("nefc overflow"). Give a comfortable cap with headroom for pile-ups.
  cfg.sim.njmax = 512

  # --- swap the single-clip motion command for the twist-indexed library command ---
  # Copy the (already play-adjusted) MotionCommandCfg fields into a LibraryMotionCommandCfg.
  old = cfg.commands["motion"]
  cfg.commands["motion"] = LibraryMotionCommandCfg(
    entity_name=old.entity_name,
    # Timer-driven twist resampling (velocity-style); dormant in play (twist is pinned anyway).
    resampling_time_range=RESAMPLING_TIME_RANGE,
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
    twist_command_range=(PLAY_TWIST if play else TWIST_RANGE),
    rel_static_envs=(0.0 if play else REL_STATIC_ENVS),
    # motion_file is unused by the library loader, but train/play's tracking-task guard
    # (isinstance MotionCommandCfg) requires an existing path -> point it at the motion dir.
    motion_file=MOTION_DIR,
  )

  # --- observations: tell the policy the commanded twist (+ a phase clock), no noise ---
  twist_obs = ObservationTermCfg(
    func=mdp.commanded_twist, params={"command_name": "motion"}
  )
  phase_obs = ObservationTermCfg(
    func=mdp.motion_phase, params={"command_name": "motion"}
  )
  for group in ("actor", "critic"):
    cfg.observations[group].terms["commanded_twist"] = replace(twist_obs)
    cfg.observations[group].terms["motion_phase"] = replace(phase_obs)

  # --- strip the target-motion reference from the ACTOR only (keep it in the critic) ---
  # The actor now sees only proprioception + phase + commanded twist; the critic keeps the full
  # reference (asymmetric). The library still drives learning via the imitation reward.
  for ref_term in ("command", "motion_anchor_ori_b"):
    cfg.observations["actor"].terms.pop(ref_term, None)

  # Restore a reference-free orientation signal: projected gravity = the robot's own IMU tilt
  # (roll/pitch), replacing the orientation lost with motion_anchor_ori_b. Yaw is left to reward.
  cfg.observations["actor"].terms["projected_gravity"] = ObservationTermCfg(
    func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
  )

  # --- reward: light direct twist tracking to hit the (continuous) commanded twist ---
  cfg.rewards["twist"] = RewardTermCfg(
    func=mdp.twist_tracking,
    weight=0.3,
    params={"command_name": "motion", "std": 0.1},
  )

  # --- position cost: replace the base ABSOLUTE anchor-position reward with an egocentric,
  # path-following one. The clip's absolute position is a forward sawtooth (resets each loop), so
  # tracking it punishes net progress; the egocentric target advances with the reference velocity and
  # re-bases to the robot on resample, so it keeps a real position cost without fighting forward
  # motion. Same weight/std as the base term it replaces.
  cfg.rewards["motion_global_root_pos"] = RewardTermCfg(
    func=mdp.egocentric_anchor_position_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 0.3},
  )

  return cfg
