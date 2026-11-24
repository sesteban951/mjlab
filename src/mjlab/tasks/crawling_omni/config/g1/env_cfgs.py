"""Unitree G1 OMNIDIRECTIONAL crawling: full 3-D twist gait-library tracking with blended transitions.

Builds on the shared crawl base (contact-rich physics, action centred on the crawl pose,
reference-free actor, blended clip transitions, idle-pose stop), pointed at the FULL 3-D gait
library, and opens the commanded-twist range on all three axes ``[vx, vy, wz]``.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.crawling_common.env_cfgs import unitree_g1_crawling_env_cfg
from mjlab.tasks.gait_library.library import LIBRARY_SPECS
from mjlab.tasks.gait_library.mdp.blending import BlendingMotionCommandCfg
from mjlab.viewer.viewer_config import ViewerConfig

# The full forward [vx, vy, wz] crawl grid plus a zero-twist idle clip.
# Rebuild with `uv run python -m mjlab.scripts.build_library omni`.
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
  cfg = unitree_g1_crawling_env_cfg(MOTION_DIR, play=play)

  # Open all three twist axes; the base's uniform per-axis sampler does the rest.
  cmd = cfg.commands["motion"]
  assert isinstance(cmd, BlendingMotionCommandCfg)
  cmd.twist_command_range = PLAY_TWIST if play else TWIST_RANGE
  cmd.twist_metric_weights = TWIST_METRIC_WEIGHTS

  # Replay-only viewer tweaks: hide the green reference ghost, and free the camera (don't pin the
  # view to the first robot) so it can be panned/orbited manually. replace() avoids mutating a
  # possibly-shared viewer object.
  if play:
    cmd.debug_vis = False
    cfg.viewer = replace(cfg.viewer, origin_type=ViewerConfig.OriginType.WORLD)

  return cfg
