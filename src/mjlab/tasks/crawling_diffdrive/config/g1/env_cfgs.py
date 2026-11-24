"""Unitree G1 DIFFERENTIAL-DRIVE crawling: tank-style twist command over the crawl gait library.

Builds on the shared crawl base (contact-rich physics, action centred on the crawl pose,
reference-free actor, blended clip transitions, idle-pose stop), then swaps the twist command
for the differential-drive variant (translate XOR rotate, forward or backward) over a library
that mixes pure-forward, pure-backward and pure-turn clips plus an idle clip.
"""

from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.crawling_common.env_cfgs import unitree_g1_crawling_env_cfg
from mjlab.tasks.crawling_diffdrive.mdp.commands import DiffDriveMotionCommandCfg
from mjlab.tasks.gait_library.library import LIBRARY_SPECS

# Straight forward, straight backward and in-place turn clips plus a zero-twist idle clip.
# Rebuild with `uv run python -m mjlab.scripts.build_library diffdrive`.
MOTION_DIR = str(LIBRARY_SPECS["diffdrive"].tracking_dir)

# Straight-mode speed ranges [m/s]: forward (vx>0) and backward (vx<0). Turn-mode yaw-rate magnitude
# range [rad/s] (sign randomized).
VX_FWD_RANGE = (0.10, 0.30)
VX_BCK_RANGE = (-0.25, -0.05)
WZ_RANGE = (0.25, 0.55)
# Fraction of resamples that turn in place; of the straight ones, fraction driven backward. A
# separate rel_static (inherited, 0.15) is idle.
REL_TURN_ENVS = 0.4
REL_BACK_ENVS = 0.5
# Play: pin a single forward gait so the reference ghost shows one clean clip.
PLAY_VX_FWD = (0.20, 0.20)


def unitree_g1_crawling_diffdrive_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 differential-drive crawling config: translate(fwd/bck)-XOR-rotate command + blended transitions."""
  cfg = unitree_g1_crawling_env_cfg(MOTION_DIR, play=play)

  # Re-wrap the blending command as the differential-drive one, keeping every field.
  old = cfg.commands["motion"]
  kwargs = {f.name: getattr(old, f.name) for f in fields(old)}
  # twist_command_range is unused by the diffdrive sampler but kept sane for any base-class reads.
  kwargs["twist_command_range"] = (
    (VX_BCK_RANGE[0], VX_FWD_RANGE[1]),
    (0.0, 0.0),
    (-WZ_RANGE[1], WZ_RANGE[1]),
  )

  cfg.commands["motion"] = DiffDriveMotionCommandCfg(
    **kwargs,
    vx_fwd_range=(PLAY_VX_FWD if play else VX_FWD_RANGE),
    vx_bck_range=VX_BCK_RANGE,
    wz_range=WZ_RANGE,
    rel_turn_envs=(0.0 if play else REL_TURN_ENVS),
    rel_back_envs=(0.0 if play else REL_BACK_ENVS),
  )

  return cfg
