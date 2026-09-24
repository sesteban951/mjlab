"""Unitree G1 unicycle WALKING: drive-and-turn twist command over the walk gait library.

The walking sibling of ``G1-Jog-Unicycle``. Same base (``G1-Tracking-Custom``: feet-only
frictional contact, the shared custom DR, actuator delay), same blended library command, same
unicycle sampler, same reference-free actor, same egocentric path/heading rewards, same standing
idle stop, same pelvis-yaw twist reward. What differs from the jog is the library and its ranges;
what differs from ``G1-Walking-DiffDrive`` is the command set:

  diff drive   translate XOR rotate   [vx, 0, 0]  or  [0, 0, wz]
  unicycle     translate AND rotate   [vx, 0, wz] or  [0, 0, wz]      <- this task

``wz = 0`` is the straight walk, so it is the lower edge of the arc mode rather than a mode of its
own. Lateral motion is still never commanded.

Library: ``LIBRARY_SPECS["walk_unicycle"]``; build/refresh with
``uv run python -m mjlab.scripts.build_library walk_unicycle``.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.crawling_common.library import LIBRARY_SPECS, load_idle_qpos
from mjlab.tasks.crawling_common.observations import make_actor_reference_free
from mjlab.tasks.jog_unicycle.mdp.commands import (
  UnicycleMotionCommandCfg,
  span_normalized_weights,
)
from mjlab.tasks.tracking.config.g1_custom.env_cfgs import (
  unitree_g1_custom_flat_tracking_env_cfg,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.walk_unicycle import mdp

# One spec drives both the library build and this env: the converted tracking dir the command
# loads, and the idle csv that is both the zero-twist clip and the robot's initial pose.
SPEC = LIBRARY_SPECS["walk_unicycle"]
MOTION_DIR = str(SPEC.tracking_dir)
IDLE_QPOS_CSV = SPEC.idle_csv

# ===== THE RANGES ARE THE LIBRARY'S ==============================================================
# Every range here must be a subset of what the built library spans. Commanding outside the grid
# does not fail -- it silently snaps to the nearest edge clip and the twist reward then chases a
# twist no clip demonstrates -- so these are read off the library, never guessed. They are the
# grids of mj-nlp's examples/g1_mimic_periodic/config_library.py (walk_fwd, walk_bck,
# walk_turn_pos, walk_turn_neg); re-read the built clips after any rebuild:
#
#   uv run python -c "import numpy as np,glob; \
#     t=np.stack([np.load(f)['twist'] for f in glob.glob('<tracking_dir>/*.npz')]); \
#     print('vx',t[:,0].min(),t[:,0].max(),'wz',t[:,2].min(),t[:,2].max())"
#
VX_FWD_RANGE = (0.50, 0.90)  # arc-mode forward speed [m/s]   = walk_fwd's vx grid
VX_BCK_RANGE = (-0.80, -0.50)  # arc-mode backward speed [m/s]  = walk_bck's vx grid
ARC_WZ_RANGE = (0.0, 0.50)  # arc-mode yaw MAGNITUDE [rad/s]; 0 = straight walk
WZ_RANGE = (0.50, 1.50)  # pivot-mode yaw MAGNITUDE [rad/s]
#
# Unlike the jog, the two yaw ranges MEET at 0.5 rad/s: the arc grids sweep wz in [-0.5, +0.5]
# and the pivots start where they stop, so a turn command is demonstrated at every rate from
# straight to the fastest pivot -- as an arc below 0.5 and as a stationary turn above it.
# =================================================================================================

# Mode split, the jog's: pivots, then of the arcing envs the fraction walked backward, then the
# stand fraction. Fewer pivots than the diff-drive walk's 0.4 -- the arc mode here already covers
# every turning radius from straight to tight, so pure pivots are one end of a continuum rather
# than half the task.
REL_TURN_ENVS = 0.3
REL_BACK_ENVS = 0.4
# As in the standing task: higher than the velocity task's 0.05, because standing also means
# UN-learning the phase->swing coupling the walk clips teach.
REL_STATIC_ENVS = 0.15
# Timer-driven resampling (s). Every clip shares the 1.4 s period, so a resample keeps the phase
# clock and blends the reference to the new clip over BLEND_TIME_S instead of teleporting.
RESAMPLING_TIME_RANGE = (3.0, 8.0)
BLEND_TIME_S = 0.4

# Nearest-clip metric. DERIVED, not chosen: on a 2-D (vx, wz) grid every candidate clip differs on
# both axes at once, so the weights decide whether 1 m/s of speed error outranks 1 rad/s of yaw
# error -- different units. span_normalized_weights makes a full-range miss cost the same on each.
# The diff-drive walk could use equal weights because its grid is 1-D within a mode; this cannot.
TWIST_METRIC_WEIGHTS = span_normalized_weights(
  (VX_BCK_RANGE[0], VX_FWD_RANGE[1]), (-WZ_RANGE[1], WZ_RANGE[1])
)

# Twist-reward width. The diff-drive walk's 0.5, MEASURED on its straight and pivot clips: the
# reference's own pelvis twist scatters about its stride mean by 0.22 (walk 0.75), 0.40 (walk
# 1.0) and 0.60 (turn 1.0) RMS, wz-dominated, and 0.5 gives ~0.8 on the walks and ~0.25 on the
# turns at zero tracking error. The ARC clips are not yet measured -- the jog's arcs came out at
# 0.44 RMS and its pivots at 0.76, which is why it sits at 0.7. Re-measure after the build:
# per-clip RMS of [vx, vy, wz] about that clip's own mean, and widen if a perfectly tracking
# clip would score below ~0.25.
TWIST_STD = 0.5

# Play: pin one mid-range STRAIGHT walk (wz = 0) so the reference ghost shows one clean clip.
PLAY_VX_FWD = (0.75, 0.75)
PLAY_ARC_WZ = (0.0, 0.0)


def unitree_g1_walk_unicycle_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 unicycle walking: arc(fwd/bck + turn) or pivot command + blended clip transitions."""
  cfg = unitree_g1_custom_flat_tracking_env_cfg(has_state_estimation=False, play=play)

  # mode_machine 11 (hip pitch on the 22.5:1 gearbox) is inherited from the custom tracking base.

  # --- start on, and centre the action space on, the standing idle pose ---
  # The joint-position action is applied as ``target = default_joint_pos + scale*action``, so the
  # initial state's joint angles are where a zero action sits. Use the idle pose (the same csv that
  # becomes the zero-twist clip) so a zero action already IS the stand, and idle <-> walk blends
  # stay near the gait's height band (base z 0.776 vs the clips' 0.734-0.769; the custom base's
  # HOME keyframe stands at 0.80 with the feet in the air).
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

  # --- reference-free actor: commanded twist + phase in both groups, reference terms stripped
  # from the actor, projected gravity restored in their place (see crawling_common.observations).
  make_actor_reference_free(cfg)

  # --- reward: light direct twist tracking toward the commanded twist, pelvis-yaw heading ---
  cfg.rewards["twist"] = RewardTermCfg(
    func=mdp.twist_tracking,
    weight=0.3,
    params={"command_name": "motion", "std": TWIST_STD},
  )

  # --- position cost: egocentric path target instead of the clip's ABSOLUTE anchor position. A
  # looping clip's position is a sawtooth (the walk resets ~1 m back every stride), so tracking
  # it punishes net progress; the egocentric target advances with the reference velocity and
  # re-bases to the robot on resample.
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
