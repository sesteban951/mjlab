"""Unitree G1 walk+jog unicycle: one twist command over a two-stride gait library.

Reuses ``G1-Jog-Unicycle``'s env config verbatim (G1-Tracking-Custom base, standing
idle, reference-free actor, blended clip transitions, pelvis-yaw twist reward,
twist-gated foot terms), then repoints the command at the merged walk+jog library,
widens the pivot range to its span (the other ranges already match the jog's) and sets
the twist-reward std between the two families' values. The two-stride handling itself
lives in the library command."""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.crawling_common.library import LIBRARY_SPECS
from mjlab.tasks.jog_unicycle.config.g1.env_cfgs import unitree_g1_jog_unicycle_env_cfg
from mjlab.tasks.jog_unicycle.mdp.commands import (
  UnicycleMotionCommandCfg,
  span_normalized_weights,
)

# The merged library (walk clips T = 70 frames, jog clips T = 43). Selection and the
# twist cut between the families live in
# crawling_common.library.LIBRARY_SPECS["walkjog_unicycle"]; rebuild with `uv run python
# -m mjlab.scripts.build_library walkjog_unicycle`.
SPEC = LIBRARY_SPECS["walkjog_unicycle"]
MOTION_DIR = str(SPEC.tracking_dir)

# ===== THE RANGES ARE THE MERGED LIBRARY'S, MEASURED ==================================
# Union of the two families after the cut (see the spec). Below the boundary a command
# snaps to a walk clip, above it to a jog clip, so speed (mostly) selects the gait:
#
#   forward   walk +0.50..+0.90  |  jog +1.00..+1.50      (cut at +0.95)
#   backward  walk -0.70..-0.50  |  jog -1.00..-0.80      (cut at -0.7315)
#   pivot     walk  0.50.. 1.45  |  jog  1.50.. 2.00      (cut at  1.4945, on |wz|)
#   arc       both families sweep |wz| 0..0.50 at every speed
#
# Measured 2026-09-27 over 456 clips: vx [-1.00, +1.50], vy [0, 0], wz [-2.00, +2.00].
VX_FWD_RANGE = (0.50, 1.50)  # arc-mode forward speed [m/s]: walk band then jog band
VX_BCK_RANGE = (-1.00, -0.50)  # arc-mode backward speed [m/s]
ARC_WZ_RANGE = (
  0.0,
  0.50,
)  # arc-mode yaw MAGNITUDE [rad/s]; 0 = straight, both families
WZ_RANGE = (
  0.50,
  2.00,
)  # pivot-mode yaw MAGNITUDE [rad/s], now CONTINUOUS across the families
# ======================================================================================

# The mode split is inherited from the jog task (0.3 pivot / 0.4 of the arcs backward /
# 0.15 stand).

# Nearest-clip metric, derived from the merged spans exactly as the jog task derives its
# own: the grid is 2-D in (vx, wz) so the weights must make a full-range miss cost the
# same on each axis.
TWIST_METRIC_WEIGHTS = span_normalized_weights(
  (VX_BCK_RANGE[0], VX_FWD_RANGE[1]), (-WZ_RANGE[1], WZ_RANGE[1])
)

# Twist-reward width. The walk's own measured scatter wants 0.5 and the jog's 0.7 (see
# those tasks); the merged library contains both, and one std has to serve. 0.6 is the
# midpoint, NOT a measurement -- re-derive it from the per-clip RMS of [vx, vy, wz]
# about each clip's mean if the twist term reads as pinned near 0 or 1 in training.
TWIST_STD = 0.6

# Play: pin the walk's native straight gait so the reference ghost shows one clean clip.
PLAY_VX_FWD = (0.75, 0.75)
PLAY_ARC_WZ = (0.0, 0.0)


def unitree_g1_walkjog_unicycle_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 walk+jog unicycle: the jog task's env over the merged two-stride library."""
  cfg = unitree_g1_jog_unicycle_env_cfg(play=play)

  # Repoint the unicycle command at the merged library and open the ranges to its span.
  # The command class, sampler, rewards, observations, RSI, blend window and mode split
  # are all inherited; only the library, the pivot range and the twist-reward std
  # change. (The jog and walk idle poses are byte-identical, so the inherited initial
  # state already matches the merged library's idle clip.)
  old = cfg.commands["motion"]
  assert isinstance(old, UnicycleMotionCommandCfg)
  cfg.commands["motion"] = replace(
    old,
    motion_dir=MOTION_DIR,
    allow_ragged=SPEC.ragged,  # two strides in one library (see the spec)
    motion_file=MOTION_DIR,  # unused by the loader; tracking-task guard wants a path
    twist_command_range=(
      (VX_BCK_RANGE[0], VX_FWD_RANGE[1]),
      (0.0, 0.0),
      (-WZ_RANGE[1], WZ_RANGE[1]),
    ),
    twist_metric_weights=TWIST_METRIC_WEIGHTS,
    vx_fwd_range=(PLAY_VX_FWD if play else VX_FWD_RANGE),
    vx_bck_range=VX_BCK_RANGE,
    arc_wz_range=(PLAY_ARC_WZ if play else ARC_WZ_RANGE),
    wz_range=WZ_RANGE,
  )

  cfg.rewards["twist"].params["std"] = TWIST_STD

  return cfg
