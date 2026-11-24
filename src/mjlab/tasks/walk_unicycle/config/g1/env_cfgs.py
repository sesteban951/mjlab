"""Unitree G1 unicycle WALKING: drive-and-turn twist command over the walk gait library.

The walking sibling of ``G1-Jog-Unicycle``: same base (``G1-Tracking-Custom``), library command,
unicycle sampler, reference-free actor, egocentric path/heading rewards, standing idle stop and
pelvis-yaw twist reward. Only the library and its commanded ranges differ. Commands: arcs
``[vx, 0, wz]`` (``wz = 0`` is the straight walk) or pivots ``[0, 0, wz]``, never lateral.

Library: ``LIBRARY_SPECS["walk_unicycle"]``; ``uv run python -m mjlab.scripts.build_library walk_unicycle``.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.gait_library.env_cfgs import (
  library_command_fields,
  make_actor_reference_free,
  use_egocentric_root_rewards,
)
from mjlab.tasks.gait_library.library import LIBRARY_SPECS, load_idle_qpos
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
    **library_command_fields(old, MOTION_DIR),
    resampling_time_range=RESAMPLING_TIME_RANGE,
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

  # --- reference-free actor: twist + phase in both groups, reference terms off the actor,
  # projected gravity restored in their place (see gait_library.env_cfgs).
  make_actor_reference_free(cfg)

  # --- feet-vs-ground contact sensor: what the foot-slip and soft-landing terms read. Same
  # definition as the velocity tasks' (ankle_roll subtrees against the terrain, net force,
  # air-time tracking for first-contact detection).
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg,)

  # --- reward: light direct twist tracking toward the commanded twist, pelvis-yaw heading ---
  cfg.rewards["twist"] = RewardTermCfg(
    func=mdp.twist_tracking,
    weight=0.3,
    params={"command_name": "motion", "std": TWIST_STD},
  )

  # --- root pose: track the unrolled reference, not the clip's absolute (sawtooth) pose,
  # so net progress and net turning are not punished.
  use_egocentric_root_rewards(cfg)

  # --- stomping mitigation (hardware): foot xy speed in contact and touchdown force, only while a
  # nonzero twist is commanded (the velocity task's terms, gated; see jog_unicycle.mdp.rewards).
  cfg.rewards["foot_slip"] = RewardTermCfg(
    func=mdp.feet_slip_twist,
    weight=-0.1,
    params={
      "sensor_name": feet_ground_cfg.name,
      "command_name": "motion",
      "command_threshold": 0.05,
      "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    },
  )
  cfg.rewards["soft_landing"] = RewardTermCfg(
    func=mdp.soft_landing_twist,
    weight=-0.001,
    params={
      "sensor_name": feet_ground_cfg.name,
      "command_name": "motion",
      "command_threshold": 0.05,
    },
  )

  # Action-rate penalty: the shared limb/waist split (custom_rewards.add_custom_g1_action_rate_split)
  # is already applied by the custom tracking builder; only its weights are overridden here, so every
  # G1-Walk-Unicycle-* variant inherits them and the shared defaults (-0.15 / -0.35) stay untouched.
  cfg.rewards["action_rate_l2_limbs"].weight = -0.1
  cfg.rewards["action_rate_l2_waist"].weight = -0.3

  return cfg
