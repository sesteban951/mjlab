"""Unitree G1 crawling (forward + idle stop) with blended clip transitions.

Reuses ``G1-Crawling-Fwd``'s env config verbatim, then swaps its twist command for the blending
variant (see crawling_fwd_blending/mdp/commands.py). Everything else -- contact-rich physics,
twist library, reference-free actor, rewards -- is identical.
"""

from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.crawling_fwd.config.g1.env_cfgs import (
  unitree_g1_crawling_fwd_env_cfg,
)
from mjlab.tasks.crawling_fwd_blending.mdp.commands import BlendingMotionCommandCfg

# Reference-blend window on a mid-episode twist resample (seconds).
BLEND_TIME_S = 0.4


def unitree_g1_crawling_fwd_blending_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the twist-conditioned crawling config with blended clip transitions."""
  cfg = unitree_g1_crawling_fwd_env_cfg(play=play)

  # Swap the Fwd twist command for the blending variant, carrying over all its fields.
  old = cfg.commands["motion"]
  kwargs = {f.name: getattr(old, f.name) for f in fields(old)}
  cfg.commands["motion"] = BlendingMotionCommandCfg(**kwargs, blend_time_s=BLEND_TIME_S)

  return cfg
