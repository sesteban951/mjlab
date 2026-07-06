"""Crawling-specific rewards.

The BeyondMimic tracking rewards already yield speed control (imitating the speed-selected clip),
but the command is continuous while the library is discrete; this small term nudges the policy to
interpolate between clips toward the exact commanded forward speed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .commands import LibraryMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def forward_speed_tracking(
  env: ManagerBasedRlEnv, command_name: str, std: float, axis: int = 1
) -> torch.Tensor:
  """exp(-(v_anchor[axis] - commanded_speed)^2 / std^2); axis=1 -> world +Y (crawl forward)."""
  cmd = cast(LibraryMotionCommand, env.command_manager.get_term(command_name))
  v_fwd = cmd.robot_anchor_lin_vel_w[:, axis]
  return torch.exp(-torch.square(v_fwd - cmd.speed_command) / (std**2))
