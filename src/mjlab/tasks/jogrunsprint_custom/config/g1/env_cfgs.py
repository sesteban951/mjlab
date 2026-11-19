"""G1 jog + run + sprint: the jog task's env over the merged three-stride library.

Reuses ``G1-Jog-Unicycle`` verbatim and only repoints the command at
``LIBRARY_SPECS["jogrunsprint"]``, widens the forward range and resets the twist std.
Rebuild with ``uv run python -m mjlab.scripts.build_library jogrunsprint`` (after
jog_unicycle, run_fwd, sprint_fwd). Design notes: ../../README.md.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.crawling_common.library import LIBRARY_SPECS
from mjlab.tasks.jog_unicycle.config.g1.env_cfgs import unitree_g1_jog_unicycle_env_cfg
from mjlab.tasks.jog_unicycle.mdp.commands import (
  UnicycleMotionCommandCfg,
  span_normalized_weights,
)

SPEC = LIBRARY_SPECS["jogrunsprint"]
MOTION_DIR = str(SPEC.tracking_dir)

# Measured library span: jog 0.5-1.5, run 1.5-2.5, sprint 2.5-3.5 m/s forward.
VX_FWD_RANGE = (0.50, 3.50)  # arc-mode forward speed [m/s]
VX_BCK_RANGE = (-1.00, -0.50)  # arc-mode backward speed [m/s], jog only
ARC_WZ_RANGE = (0.0, 0.50)  # arc-mode yaw magnitude [rad/s]
WZ_RANGE = (1.00, 2.00)  # pivot yaw magnitude [rad/s], the jog's fast pivots

# Backward arcs only exist at jog speed, so give the wide forward range more samples.
REL_BACK_ENVS = 0.2

TWIST_METRIC_WEIGHTS = span_normalized_weights(
  (VX_BCK_RANGE[0], VX_FWD_RANGE[1]), (-WZ_RANGE[1], WZ_RANGE[1])
)

# Per-clip twist RMS scatter: jog 0.32, run 0.59, sprint 1.03, pivots 0.90 (README).
TWIST_STD = 0.9

# Play: a straight mid-range run.
PLAY_VX_FWD = (2.00, 2.00)
PLAY_ARC_WZ = (0.0, 0.0)


def unitree_g1_jogrunsprint_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 jog + run + sprint: the jog task's env over the merged library."""
  cfg = unitree_g1_jog_unicycle_env_cfg(play=play)

  old = cfg.commands["motion"]
  assert isinstance(old, UnicycleMotionCommandCfg)
  cfg.commands["motion"] = replace(
    old,
    motion_dir=MOTION_DIR,
    allow_ragged=SPEC.ragged,
    motion_file=MOTION_DIR,  # unused by the loader; the tracking-task guard wants a path
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
    rel_back_envs=(0.0 if play else REL_BACK_ENVS),
  )

  cfg.rewards["twist"].params["std"] = TWIST_STD

  return cfg
