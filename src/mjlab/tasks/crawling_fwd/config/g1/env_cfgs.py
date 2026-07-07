"""Unitree G1 crawling (forward + idle stop) environment config.

Same as ``G1-Crawling-NoStopping`` but the commanded speed range includes 0. Speeds below
``STOP_SPEED_THRESHOLD`` are pinned to 0, which snaps to the static idle clip
(``crawl_vp0_000.npz``, produced by scripts/library_to_npz.py from ``qpos_idle.csv``). The robot
then holds the idle pose instead of crawling. No freeze/rest-reward machinery -- imitating the
static idle clip *is* the rest, using the same tracking rewards as every other speed.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.crawling_no_stopping.config.g1.env_cfgs import (
  unitree_g1_crawling_flat_env_cfg,
)
from mjlab.tasks.crawling_no_stopping.mdp.commands import LibraryMotionCommandCfg

# Include 0: below STOP_SPEED_THRESHOLD the command pins to 0 -> idle clip (stop).
SPEED_RANGE = (0.0, 0.40)
# Midpoint between the idle (0.0) and slowest crawl (0.08) clips: commands in [0, 0.04) stop.
STOP_SPEED_THRESHOLD = 0.04
# In play mode, pin a single speed. Set 0.0 to watch the idle-pose stop instead of a crawl.
PLAY_SPEED = 0.28


def unitree_g1_crawling_fwd_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Unitree G1 (29-DoF) forward crawling config with an idle-pose stop."""
  cfg = unitree_g1_crawling_flat_env_cfg(play=play)

  # Widen the speed range to include a stop band and pin sub-threshold commands to the idle clip.
  motion = cfg.commands["motion"]
  assert isinstance(motion, LibraryMotionCommandCfg)
  motion.speed_command_range = (PLAY_SPEED, PLAY_SPEED) if play else SPEED_RANGE
  motion.stop_speed_threshold = STOP_SPEED_THRESHOLD

  return cfg
