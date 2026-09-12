"""Unitree G1 STANDING differential-drive walking: tank-style twist command over the upright
walk gait library.

The upright sibling of ``G1-Crawling-DiffDrive``. Same library machinery -- the twist-indexed
``LibraryMotionCommand`` with the differential-drive sampler (translate XOR rotate, forward or
backward, never lateral), blended clip transitions, a REFERENCE-FREE actor (proprioception +
projected gravity + commanded twist + phase clock; the critic keeps the full reference), the
egocentric path/heading rewards that let looping and turning clips accumulate net motion, and an
idle-pose stop. Three things are upright-specific:

* the base is ``G1-Tracking-Custom`` (feet-only frictional contact, the shared custom DR, actuator
  delay), NOT the contact-rich crawl base (whole-body condim 3, raised constraint cap);
* the initial / zero-action pose is the STANDING idle (mj-nlp's captured
  ``trajectories/g1/poses/stand_idle_qpos.csv``), the same pose the zero-twist clip holds -- the
  crawl env did the same with its prone idle;
* the twist reward's heading is the pelvis yaw (body-x azimuth); the crawl one reads body-Z.

Library: mj-nlp's examples/g1_mimic_periodic/library -- walk_fwd (vx +0.50..+1.00), walk_bck
(vx -0.90..-0.40), walk_turn_pos/neg (in-place wz +-0.50..1.50), all T = 1.4 s -> 70 tracking
frames at 50 Hz, plus the standing idle. Selection lives in
``crawling_common.library.LIBRARY_SPECS["standing_diffdrive"]``; build/refresh it with
``uv run python -m mjlab.scripts.build_library standing_diffdrive``.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.crawling_common.library import LIBRARY_SPECS, load_idle_qpos
from mjlab.tasks.crawling_diffdrive.mdp.commands import DiffDriveMotionCommandCfg
from mjlab.tasks.standing_diffdrive import mdp
from mjlab.tasks.tracking.config.g1_custom.env_cfgs import (
  unitree_g1_custom_flat_tracking_env_cfg,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# One spec drives both the library build and this env: the converted tracking dir the command
# loads, and the standing idle csv that is both the zero-twist clip and the robot's initial pose.
SPEC = LIBRARY_SPECS["standing_diffdrive"]
MOTION_DIR = str(SPEC.tracking_dir)
IDLE_QPOS_CSV = SPEC.idle_csv

# Straight-mode speed ranges [m/s] = the walk_fwd / walk_bck grid spans (0.05 m/s steps). Turn-mode
# yaw-rate MAGNITUDE range [rad/s] = the walk_turn grids (0.1 rad/s steps; sign randomized).
VX_FWD_RANGE = (0.50, 1.00)
VX_BCK_RANGE = (-0.90, -0.40)
WZ_RANGE = (0.50, 1.50)
# Fraction of resamples that turn in place; of the straight ones, the fraction walked backward; and
# the fraction that stand (zero twist -> the idle clip). Same split as the crawl.
REL_TURN_ENVS = 0.4
REL_BACK_ENVS = 0.5
REL_STATIC_ENVS = 0.05
# Timer-driven twist resampling (s). Every clip shares the 1.4 s period, so a resample keeps the
# phase clock and blends the reference to the new clip over BLEND_TIME_S instead of teleporting.
RESAMPLING_TIME_RANGE = (3.0, 8.0)
BLEND_TIME_S = 0.4
# The turn clips have vx = vy = 0 and the straight clips wz = 0, so snapping is 1-D within a mode
# and the cross-axis weights never compete; equal weights.
TWIST_METRIC_WEIGHTS = (1.0, 1.0, 1.0)
# Twist-reward width. MEASURED, not copied from the crawl (0.1): the reference's own pelvis twist
# scatters about its stride mean by 0.22 (walk 0.75), 0.40 (walk 1.0) and 0.60 (turn 1.0) RMS,
# wz-dominated -- with 0.1 even perfect tracking would score exp(-6) ~ 0 and the term would carry
# no gradient. 0.5 gives ~0.8 on the walks and ~0.25 on the turns at zero tracking error.
TWIST_STD = 0.5
# Play: pin one mid-range forward gait so the reference ghost shows one clean clip.
PLAY_VX_FWD = (0.75, 0.75)


def unitree_g1_standing_diffdrive_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 upright differential-drive walking: translate(fwd/bck)-XOR-rotate command + blending."""
  cfg = unitree_g1_custom_flat_tracking_env_cfg(has_state_estimation=False, play=play)

  # mode_machine 11 (hip pitch on the 22.5:1 gearbox) is inherited from the custom tracking base.

  # --- start on, and centre the action space on, the standing idle pose ---
  # The joint-position action is applied as ``target = default_joint_pos + scale*action``, so the
  # initial state's joint angles are where a zero action sits. Use the idle pose (the same csv that
  # becomes the zero-twist clip) so a zero action already IS the stand, and idle <-> walk blends
  # stay near the gait's height band (base z 0.776 vs the clips' 0.734-0.769; the custom base's
  # HOME keyframe stands at 0.80 with the feet in the air). Scoped to this env; g1_custom itself
  # is untouched.
  pos, quat, joint_pos = load_idle_qpos(IDLE_QPOS_CSV)
  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.init_state = replace(
    robot_cfg.init_state, pos=pos, rot=quat, joint_pos=joint_pos
  )

  # --- swap the single-clip motion command for the differential-drive library command ---
  # Copy the (already play-adjusted) MotionCommandCfg fields; add the library + mode-split ranges.
  old = cfg.commands["motion"]
  assert isinstance(old, MotionCommandCfg)
  cfg.commands["motion"] = DiffDriveMotionCommandCfg(
    entity_name=old.entity_name,
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
    # motion_file is unused by the library loader, but train/play's tracking-task guard
    # (isinstance MotionCommandCfg) requires an existing path -> point it at the motion dir.
    motion_file=MOTION_DIR,
    # Unused by the diffdrive sampler; kept sane for any base-class reads.
    twist_command_range=(
      (VX_BCK_RANGE[0], VX_FWD_RANGE[1]),
      (0.0, 0.0),
      (-WZ_RANGE[1], WZ_RANGE[1]),
    ),
    twist_metric_weights=TWIST_METRIC_WEIGHTS,
    rel_static_envs=(0.0 if play else REL_STATIC_ENVS),
    blend_time_s=BLEND_TIME_S,
    vx_fwd_range=(PLAY_VX_FWD if play else VX_FWD_RANGE),
    vx_bck_range=VX_BCK_RANGE,
    wz_range=WZ_RANGE,
    rel_turn_envs=(0.0 if play else REL_TURN_ENVS),
    rel_back_envs=(0.0 if play else REL_BACK_ENVS),
  )

  # --- observations: the commanded twist (+ a phase clock), no noise, in both groups ---
  twist_obs = ObservationTermCfg(
    func=mdp.commanded_twist, params={"command_name": "motion"}
  )
  phase_obs = ObservationTermCfg(
    func=mdp.motion_phase, params={"command_name": "motion"}
  )
  for group in ("actor", "critic"):
    cfg.observations[group].terms["commanded_twist"] = replace(twist_obs)
    cfg.observations[group].terms["motion_phase"] = replace(phase_obs)

  # --- strip the target-motion reference from the ACTOR only (the critic keeps it, asymmetric) ---
  # has_state_estimation=False already dropped motion_anchor_pos_b and base_lin_vel; the actor now
  # sees only proprioception + phase + commanded twist. The library still drives learning via the
  # imitation rewards.
  for ref_term in ("command", "motion_anchor_ori_b"):
    cfg.observations["actor"].terms.pop(ref_term, None)

  # Restore a reference-free orientation signal: projected gravity = the robot's own IMU tilt
  # (roll/pitch), replacing the orientation lost with motion_anchor_ori_b. Yaw is left to reward.
  cfg.observations["actor"].terms["projected_gravity"] = ObservationTermCfg(
    func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
  )

  # --- reward: light direct twist tracking toward the (continuous) commanded twist, UPRIGHT
  # heading (pelvis yaw), see mdp/rewards.py ---
  cfg.rewards["twist"] = RewardTermCfg(
    func=mdp.twist_tracking,
    weight=0.3,
    params={"command_name": "motion", "std": TWIST_STD},
  )

  # --- position cost: egocentric path target instead of the clip's ABSOLUTE anchor position. A
  # looping clip's position is a sawtooth (the walk resets 1 m back every stride), so tracking it
  # punishes net progress; the egocentric target advances with the reference velocity and re-bases
  # to the robot on resample. Same weight/std as the base term it replaces.
  cfg.rewards["motion_global_root_pos"] = RewardTermCfg(
    func=mdp.egocentric_anchor_position_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 0.3},
  )

  # --- orientation cost: the same fix for HEADING. A turn clip's absolute yaw wraps back 81 deg
  # every stride; the egocentric heading target accumulates the reference yaw rate instead, so net
  # turning is rewarded while uprightness is still tracked. Same weight/std as the base term.
  cfg.rewards["motion_global_root_ori"] = RewardTermCfg(
    func=mdp.egocentric_anchor_orientation_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 0.4},
  )

  # Action-rate penalty comes from the shared limb/waist split applied by the custom tracking
  # builder (custom_rewards.add_custom_g1_action_rate_split); nothing to re-apply here.

  return cfg
