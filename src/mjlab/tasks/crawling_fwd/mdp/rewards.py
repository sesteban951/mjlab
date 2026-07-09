"""Crawling-specific rewards.

The BeyondMimic tracking rewards already yield twist control (imitating the twist-selected clip),
but the command is continuous while the library is discrete; this small term nudges the policy to
interpolate between clips toward the exact commanded twist [vx, vy, wz].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .commands import LibraryMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def twist_tracking(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """exp(-||achieved_twist - commanded_twist||^2 / std^2).

  ``achieved_twist`` is the robot anchor's world-frame planar twist ``[vx_w, vy_w, wz_w]`` (x/y
  linear vel + z angular vel). This is exact while the gaits share a canonical heading (the current
  forward + idle library, and any vx/vy grid). NOTE: once turning gaits (wz != 0) rotate the
  robot's heading mid-episode, world x/y no longer equal body-frame forward/lateral -- revisit to a
  heading-frame twist then.
  """
  cmd = cast(LibraryMotionCommand, env.command_manager.get_term(command_name))
  v = cmd.robot_anchor_lin_vel_w  # (N, 3) world linear vel
  w = cmd.robot_anchor_ang_vel_w  # (N, 3) world angular vel
  achieved = torch.stack([v[:, 0], v[:, 1], w[:, 2]], dim=-1)  # [vx, vy, wz]
  return torch.exp(
    -torch.sum(torch.square(achieved - cmd.twist_command), dim=-1) / (std**2)
  )


def egocentric_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """exp(-||ref_anchor_pos_w - robot_anchor_pos_w||^2 / std^2).

  Unlike ``motion_global_anchor_position_error_exp`` (which tracks the clip's ABSOLUTE world
  position -- a forward sawtooth that resets each loop and so penalizes net progress),
  ``ref_anchor_pos_w`` is an egocentric path target: re-based to the robot at each twist resample and
  advanced by the reference anchor velocity. It penalizes drifting off the intended path (lateral
  slip, forward lag, wrong height) without punishing forward progress.
  """
  cmd = cast(LibraryMotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(torch.square(cmd.ref_anchor_pos_w - cmd.robot_anchor_pos_w), dim=-1)
  return torch.exp(-error / std**2)
