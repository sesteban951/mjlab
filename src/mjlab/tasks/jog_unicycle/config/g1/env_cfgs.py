"""Unitree G1 unicycle JOGGING: drive-and-turn twist command over the jog gait library.

The running sibling of ``G1-Walking-DiffDrive``. Same base (``G1-Tracking-Custom``: feet-only
frictional contact, the shared custom DR, actuator delay), same blended library command, same
reference-free actor, same egocentric path/heading rewards, same standing idle stop, same
pelvis-yaw twist reward. ONE difference: the command set.

  diff drive   translate XOR rotate   [vx, 0, 0]  or  [0, 0, wz]
  unicycle     translate AND rotate   [vx, 0, wz] or  [0, 0, wz]      <- this task

``wz = 0`` is the straight jog, so it is the lower edge of the arc mode rather than a mode of its
own. Lateral motion is still never commanded.

Library: ``LIBRARY_SPECS["jog_unicycle"]``; build/refresh with
``uv run python -m mjlab.scripts.build_library jog_unicycle``.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.crawling_common.library import LIBRARY_SPECS, load_idle_qpos
from mjlab.tasks.jog_unicycle import mdp
from mjlab.tasks.jog_unicycle.mdp.commands import (
  UnicycleMotionCommandCfg,
  span_normalized_weights,
)
from mjlab.tasks.tracking.config.g1_custom.env_cfgs import (
  unitree_g1_custom_flat_tracking_env_cfg,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# One spec drives both the library build and this env: the converted tracking dir the command
# loads, and the idle csv that is both the zero-twist clip and the robot's initial pose.
SPEC = LIBRARY_SPECS["jog_unicycle"]
MOTION_DIR = str(SPEC.tracking_dir)
IDLE_QPOS_CSV = SPEC.idle_csv

# ===== THE RANGES ARE THE LIBRARY'S, MEASURED ====================================================
# Every range here is a subset of what the built library spans. Commanding outside the grid does
# not fail -- it silently snaps to the nearest edge clip and the twist reward then chases a twist no
# clip demonstrates -- so these are read off the library, never guessed. Re-read it after a rebuild:
#
#   uv run python -c "import numpy as np,glob; \
#     t=np.stack([np.load(f)['twist'] for f in glob.glob('<tracking_dir>/*.npz')]); \
#     print('vx',t[:,0].min(),t[:,0].max(),'wz',t[:,2].min(),t[:,2].max())"
#
# Measured 2026-09-13 over 210 clips: vx [-1.00, +1.50], vy [0, 0], wz [-2.00, +2.00].
VX_FWD_RANGE = (0.50, 1.50)  # arc-mode forward speed [m/s]   = run_fwd's vx grid
VX_BCK_RANGE = (-1.00, -0.50)  # arc-mode backward speed [m/s]  = run_bck's vx grid
ARC_WZ_RANGE = (0.0, 0.50)  # arc-mode yaw MAGNITUDE [rad/s]; 0 = straight jog
WZ_RANGE = (1.00, 2.00)  # pivot-mode yaw MAGNITUDE [rad/s]
#
# THE TWO YAW RANGES DO NOT OVERLAP, and that is the library, not an oversight: the arc grids sweep
# wz in [-0.5, +0.5] (a jog cannot turn hard) while the pivots sweep 1.0-2.0 (a stationary turn
# can). So "turn while jogging" and "turn in place" are genuinely separate regimes here, with no
# demonstrated motion between 0.5 and 1.0 rad/s. Widening either range needs new clips, not a
# wider number.
# =================================================================================================

# Mode split: pivots, then of the arcing envs the fraction run backward, then the stand fraction.
# Fewer pivots than the walk task's 0.4 -- the arc mode here already covers every turning radius
# from straight to tight, so pure pivots are one end of a continuum rather than half the task.
REL_TURN_ENVS = 0.3
REL_BACK_ENVS = 0.4
# As in the standing task: higher than the velocity task's 0.05, because standing also means
# UN-learning the phase->flight coupling the jog clips teach.
REL_STATIC_ENVS = 0.15
# Timer-driven resampling (s). Every clip shares one period, so a resample keeps the phase clock
# and blends the reference to the new clip over BLEND_TIME_S instead of teleporting.
RESAMPLING_TIME_RANGE = (3.0, 8.0)
BLEND_TIME_S = 0.4

# Nearest-clip metric. DERIVED, not chosen: on a 2-D (vx, wz) grid every candidate clip differs on
# both axes at once, so the weights decide whether 1 m/s of speed error outranks 1 rad/s of yaw
# error -- different units. span_normalized_weights makes a full-range miss cost the same on each.
# The walk task could use equal weights because its grid is 1-D within a mode; this one cannot.
TWIST_METRIC_WEIGHTS = span_normalized_weights(
  (VX_BCK_RANGE[0], VX_FWD_RANGE[1]), (-WZ_RANGE[1], WZ_RANGE[1])
)

# Twist-reward width. MEASURED on this library's 210 clips, not inherited from the walk task: the
# reference's own pelvis twist scatters about its stride mean by 0.44 RMS on the arcs and 0.76 on
# the pivots (wz-dominated -- a 2 rad/s pivot's yaw rate swings +-0.74 within a stride). At the
# walk's 0.5 a PERFECTLY tracking pivot would score exp(-0.76^2/0.5^2) = 0.10, a tenth of the
# term's range, and carry almost no gradient; 0.7 gives 0.67 on the arcs and 0.31 on the pivots,
# which is the balance the walk task's own 0.5 was chosen for (~0.8 walks / ~0.25 turns).
# Re-measure after any library rebuild: per-clip RMS of [vx, vy, wz] about that clip's own mean.
TWIST_STD = 0.7

# Play: pin one mid-range STRAIGHT jog (wz = 0) so the reference ghost shows one clean clip.
PLAY_VX_FWD = (1.00, 1.00)
PLAY_ARC_WZ = (0.0, 0.0)


def unitree_g1_jog_unicycle_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 unicycle jogging: arc(fwd/bck + turn) or pivot command + blended clip transitions."""
  cfg = unitree_g1_custom_flat_tracking_env_cfg(has_state_estimation=False, play=play)

  # mode_machine 11 (hip pitch on the 22.5:1 gearbox) is inherited from the custom tracking base.

  # --- start on, and centre the action space on, the idle pose ---
  # The joint-position action is applied as ``target = default_joint_pos + scale*action``, so the
  # initial state's joint angles are where a zero action sits. Use the idle pose (the same csv that
  # becomes the zero-twist clip) so a zero action already IS the stand.
  pos, quat, joint_pos = load_idle_qpos(IDLE_QPOS_CSV)
  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.init_state = replace(
    robot_cfg.init_state, pos=pos, rot=quat, joint_pos=joint_pos
  )

  # --- swap the single-clip motion command for the unicycle library command ---
  old = cfg.commands["motion"]
  assert isinstance(old, MotionCommandCfg)
  cfg.commands["motion"] = UnicycleMotionCommandCfg(
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
    # Unused by the unicycle sampler; kept sane for any base-class reads. The full commanded box.
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
    arc_wz_range=(PLAY_ARC_WZ if play else ARC_WZ_RANGE),
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
  for ref_term in ("command", "motion_anchor_ori_b"):
    cfg.observations["actor"].terms.pop(ref_term, None)

  # Restore a reference-free orientation signal: projected gravity = the robot's own IMU tilt
  # (roll/pitch), replacing the orientation lost with motion_anchor_ori_b. Yaw is left to reward.
  cfg.observations["actor"].terms["projected_gravity"] = ObservationTermCfg(
    func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
  )

  # --- reward: light direct twist tracking toward the commanded twist, pelvis-yaw heading ---
  cfg.rewards["twist"] = RewardTermCfg(
    func=mdp.twist_tracking,
    weight=0.3,
    params={"command_name": "motion", "std": TWIST_STD},
  )

  # --- position cost: egocentric path target instead of the clip's ABSOLUTE anchor position. A
  # looping clip's position is a sawtooth, so tracking it punishes net progress; the egocentric
  # target advances with the reference velocity and re-bases to the robot on resample.
  cfg.rewards["motion_global_root_pos"] = RewardTermCfg(
    func=mdp.egocentric_anchor_position_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 0.3},
  )

  # --- orientation cost: the same fix for HEADING. An arcing clip's absolute yaw wraps back every
  # stride; the egocentric heading target accumulates the reference yaw rate instead, so net
  # turning is rewarded while uprightness is still tracked.
  cfg.rewards["motion_global_root_ori"] = RewardTermCfg(
    func=mdp.egocentric_anchor_orientation_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 0.4},
  )

  # Action-rate penalty comes from the shared limb/waist split applied by the custom tracking
  # builder (custom_rewards.add_custom_g1_action_rate_split); nothing to re-apply here.

  return cfg
