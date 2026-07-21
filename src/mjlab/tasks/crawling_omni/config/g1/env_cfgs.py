"""Unitree G1 OMNIDIRECTIONAL crawling: full 3-D twist gait-library tracking with blended transitions.

Reuses ``G1-Crawling-Fwd-Blending``'s env config verbatim (contact-rich physics, action re-centered
on the crawl pose, reference-free actor, blended clip transitions, idle-pose stop), then repoints
the twist command at the FULL 3-D gait library and opens the commanded-twist range on all three
axes ``[vx, vy, wz]``. Only four command fields change; everything else is inherited, so the two
envs stay in lock-step.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.crawling_common.library import LIBRARY_SPECS
from mjlab.tasks.crawling_fwd_blending.config.g1.env_cfgs import (
  unitree_g1_crawling_fwd_blending_env_cfg,
)

# Converted tracking-format 3-D crawl library (the full fwd [vx, vy, wz] grid + idle). Selection
# lives in crawling_common.library.LIBRARY_SPECS["omni"]; build/refresh it with
# `python -m mjlab.scripts.build_library omni`.
MOTION_DIR = str(LIBRARY_SPECS["omni"].tracking_dir)

# Commanded twist [(vx_lo, vx_hi), (vy..), (wz..)] -- FULL all-directions span (m/s, m/s, rad/s):
# forward+backward vx, lateral vy, and turn wz up to the in-place turn rate.
TWIST_RANGE = ((-0.25, 0.30), (-0.12, 0.12), (-0.60, 0.60))
# Play: pin a single mid-forward, no-steer gait so the reference ghost shows one clean clip.
PLAY_TWIST = ((0.20, 0.20), (0.0, 0.0), (0.0, 0.0))
# Nearest-clip metric weights for [vx, vy, wz]. The grid spans are comparable in magnitude
# (vx ~0.2, vy/wz ~0.3), so equal weights snap sensibly; retune if one axis should dominate.
TWIST_METRIC_WEIGHTS = (1.0, 1.0, 1.0)


def unitree_g1_crawling_omni_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 omnidirectional crawling config: 3-D twist library + blended transitions."""
  cfg = unitree_g1_crawling_fwd_blending_env_cfg(play=play)

  # Repoint the (blending) twist command at the 3-D library and open all three axes. The command
  # class, rewards, observations, RSI and blend window are all inherited unchanged.
  cmd = cfg.commands["motion"]
  cmd.motion_dir = MOTION_DIR
  cmd.motion_file = (
    MOTION_DIR  # unused by the library loader; the tracking-task guard wants a path
  )
  cmd.twist_command_range = PLAY_TWIST if play else TWIST_RANGE
  cmd.twist_metric_weights = TWIST_METRIC_WEIGHTS

  return cfg
