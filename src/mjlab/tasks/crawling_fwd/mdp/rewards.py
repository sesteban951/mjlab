"""Crawling-specific rewards.

The BeyondMimic tracking rewards already yield twist control (imitating the twist-selected clip),
but the command is continuous while the library is discrete; this small term nudges the policy to
interpolate between clips toward the exact commanded twist [vx, vy, wz].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude

from .commands import LibraryMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def twist_tracking(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """exp(-||achieved_twist - commanded_twist||^2 / std^2), with the achieved planar twist expressed
  in the robot's HEADING frame so the BODY-frame command ``[vx=forward, vy=lateral, wz=yaw-rate]``
  (the mj-nlp gait-library convention) is tracked regardless of the robot's absolute heading.

  Measured on the PELVIS (the free-joint root -- the body mj-nlp defined the twist from), NOT the
  torso anchor (they differ through the waist joints). Heading = azimuth of the pelvis body-Z axis
  ground projection: in the prone crawl body-x points down and body-z points forward, so this is the
  crawl heading (base-x / yaw_quat would be wrong here). The world pelvis lin-vel is rotated by
  Rz(-psi) into that frame; wz is the world-z angular rate (== heading-z on level ground). Without
  this, a robot correctly executing a turn (net yaw != 0) is penalized because world x/y no longer
  equal body forward/lateral.
  """
  cmd = cast(LibraryMotionCommand, env.command_manager.get_term(command_name))
  data = cmd.robot.data
  v = data.root_link_lin_vel_w  # (N, 3) PELVIS world linear vel
  w = data.root_link_ang_vel_w  # (N, 3) PELVIS world angular vel
  ez = torch.zeros_like(v)
  ez[:, 2] = 1.0
  z_w = quat_apply(data.root_link_quat_w, ez)  # pelvis body-Z axis in world
  psi = torch.atan2(
    z_w[:, 1], z_w[:, 0]
  )  # crawl heading (ground-projected body-Z azimuth)
  c, s = torch.cos(psi), torch.sin(psi)
  vx = c * v[:, 0] + s * v[:, 1]  # world -> heading frame: Rz(-psi) . v_xy
  vy = -s * v[:, 0] + c * v[:, 1]
  achieved = torch.stack(
    [vx, vy, w[:, 2]], dim=-1
  )  # [body-forward, body-lateral, yaw-rate]
  return torch.exp(
    -torch.sum(torch.square(achieved - cmd.twist_command), dim=-1) / (std**2)
  )


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
