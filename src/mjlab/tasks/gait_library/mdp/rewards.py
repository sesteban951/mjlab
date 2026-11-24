"""Gait-library rewards: the root terms against the reference unrolled in the world.

A looping clip's absolute root pose is a sawtooth, so tracking it punishes net progress
and net turning. These terms track the unrolled reference anchor instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import LibraryMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def egocentric_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """exp(-||ref_anchor_pos_w - robot_anchor_pos_w||^2 / std^2).

  ``ref_anchor_pos_w`` is the anchor of the reference clip UNROLLED in the world: re-based
  onto the robot at each resample, then advanced by the clip's own planar step, with the
  clip's height. It penalizes drifting off the commanded path since the last resample
  (lateral slip, heading drift, forward lag, wrong height) without punishing progress.
  """
  cmd = cast(LibraryMotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(torch.square(cmd.ref_anchor_pos_w - cmd.robot_anchor_pos_w), dim=-1)
  return torch.exp(-error / std**2)


def egocentric_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """exp(-quat_error(ref_anchor_quat_w, robot_anchor_quat_w)^2 / std^2).

  Orientation analog of ``egocentric_anchor_position_error_exp``. ``ref_anchor_quat_w``
  is the unrolled reference anchor's orientation: its heading is re-based onto the robot
  at each resample and then turns with the clip, while roll and pitch are the clip's own
  at the current frame. It penalizes heading drift and tilt without punishing turning.
  """
  cmd = cast(LibraryMotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(cmd.ref_anchor_quat_w, cmd.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)
