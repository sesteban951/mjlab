"""Upright twist-tracking reward.

The crawl ``twist_tracking`` (crawling_fwd/mdp/rewards.py) expresses the achieved planar pelvis
twist in the robot's heading frame so the BODY-frame command ``[vx, vy, wz]`` is tracked whatever
the absolute heading -- but it reads the heading off the pelvis body-Z axis, because prone the
crawl's forward axis IS body-z. Standing up, body-z is vertical (its ground projection is ~0 and
its azimuth is noise) and forward is body-x. This is the same reward with the heading taken as the
pelvis yaw, the azimuth of body-x -- the frame mj-nlp defines the walk twist in (its
``twist_to_disp`` / ``measure_native_twist`` integrate the body twist from the base yaw).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.crawling_fwd.mdp.commands import LibraryMotionCommand
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def twist_tracking(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """exp(-||achieved_twist - commanded_twist||^2 / std^2), achieved twist in the pelvis YAW frame.

  Measured on the PELVIS (the free-joint root -- the body mj-nlp defined the twist from), not the
  torso anchor. Heading psi = azimuth of the pelvis body-x axis; the world pelvis linear velocity
  is rotated by Rz(-psi) into [forward, lateral]; wz is the world-z angular rate (== the yaw rate
  on level ground). The reference's own pelvis twist scatters about its stride mean by 0.2-0.6 RMS
  (wz-dominated), so ``std`` has to be of that order for the term to carry gradient.
  """
  cmd = cast(LibraryMotionCommand, env.command_manager.get_term(command_name))
  data = cmd.robot.data
  v = data.root_link_lin_vel_w  # (N, 3) pelvis world linear velocity
  w = data.root_link_ang_vel_w  # (N, 3) pelvis world angular velocity
  ex = torch.zeros_like(v)
  ex[:, 0] = 1.0
  x_w = quat_apply(data.root_link_quat_w, ex)  # pelvis body-x axis in world
  psi = torch.atan2(
    x_w[:, 1], x_w[:, 0]
  )  # pelvis yaw (ground-projected body-x azimuth)
  c, s = torch.cos(psi), torch.sin(psi)
  vx = c * v[:, 0] + s * v[:, 1]  # world -> heading frame: Rz(-psi) . v_xy
  vy = -s * v[:, 0] + c * v[:, 1]
  achieved = torch.stack([vx, vy, w[:, 2]], dim=-1)  # [forward, lateral, yaw-rate]
  return torch.exp(
    -torch.sum(torch.square(achieved - cmd.twist_command), dim=-1) / (std**2)
  )
