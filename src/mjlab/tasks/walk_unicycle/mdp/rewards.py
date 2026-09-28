"""Swing-foot geometry penalties for the upright library gaits: anti toe-skid.

Why these exist: the walk_fwd clips swing the foot TOE-DOWN (~64 deg at the apex, toe-first
landing) with the toe tip skimming 1-2 cm above the floor for the last 40 % of every swing, and
nothing in the reward set looks at the toe -- the tracked foot site sits under the ankle, body
position tracking has std 0.3 m (a 2 cm miss is invisible) and the contact terms only fire once
a foot is planted. In MuJoCo the toe slides; on the robot it catches, the swing ends early with
the foot behind the hip, and above ~0.6 m/s the pelvis pitches forward at 1-3 deg/s until the
robot walks into the floor (hardware log 2026_09_27__21_13_40, ground-truth sim probes at 0.6 /
0.75 / 0.89 land the foot +24 / +27 / +28 cm ahead of the base; the robot lands it at -3 cm).

Both terms take the swing phase from the REFERENCE (its ankle above the stance height), so they
need no contact sensor and are exactly in phase with what the tracking terms ask for; both are
gated on a nonzero commanded twist like the other foot terms (``_twist_active``), so a standing
robot is never pushed to lift a foot; both are hinge costs that are exactly zero once the foot
is clear / flat, so they do not fight the reference where it is already fine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.crawling_fwd.mdp.commands import LibraryMotionCommand
from mjlab.tasks.walking_diffdrive.mdp.rewards import _twist_active
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
# Toe tip in the ankle_roll_link frame: the front foot collision capsules sit at x = 0.075,
# z = -0.025 with radius 0.01, so the sole's front edge is ~0.10 ahead and 0.035 below the link.
_TOE_OFFSET = (0.10, 0.0, -0.035)


def _feet(env: ManagerBasedRlEnv, command_name: str, swing_height: float):
  """(cmd, robot ankle body indexes [2], reference-swing mask [B, 2], env-origin z [B, 1])."""
  cmd = cast(LibraryMotionCommand, env.command_manager.get_term(command_name))
  k = [cmd.cfg.body_names.index(n) for n in _FEET]  # index within the tracked subset
  robot_idx = [int(cmd.body_indexes[i]) for i in k]
  origin_z = env.scene.env_origins[:, 2:3]
  ref_ankle_z = (
    cmd.body_pos_w[:, k, 2] - origin_z
  )  # reference ankle height above the plane
  in_swing = ref_ankle_z > swing_height
  return cmd, robot_idx, in_swing, origin_z


def toe_clearance(
  env: ManagerBasedRlEnv,
  command_name: str,
  min_height: float = 0.03,
  swing_height: float = 0.05,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Hinge cost on the SWING foot's toe tip dipping below ``min_height`` over the plane.

  ``relu(min_height - toe_z) / min_height`` per swing foot: 0 when clear, 1 with the toe on the
  floor; summed over the feet, so the range is [0, 2] per step. Swing = the reference ankle is
  above ``swing_height`` (its stance height is ~0.025); ``min_height`` is the margin the toe must
  keep, 3 cm being what the real floor took away.
  """
  cmd, idx, in_swing, origin_z = _feet(env, command_name, swing_height)
  pos = cmd.robot.data.body_link_pos_w[:, idx]  # [B, 2, 3]
  quat = cmd.robot.data.body_link_quat_w[:, idx]  # [B, 2, 4]
  toe = torch.tensor(_TOE_OFFSET, device=pos.device, dtype=pos.dtype).expand_as(pos)
  toe_z = (pos + quat_apply(quat, toe))[..., 2] - origin_z  # [B, 2]
  deficit = torch.relu(min_height - toe_z) / min_height
  cost = torch.sum(deficit * in_swing.float(), dim=1)
  n_swing = torch.clamp(in_swing.float().sum(), min=1.0)
  env.extras["log"]["Metrics/swing_toe_height_mean"] = (
    torch.sum(toe_z * in_swing.float()) / n_swing
  )
  return cost * _twist_active(env, command_name, command_threshold)


def swing_foot_pitch(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_toe_down: float = 0.35,
  swing_height: float = 0.05,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Hinge cost on the SWING foot pointing its toe down past ``max_toe_down`` [rad].

  Foot pitch = angle of the ankle_roll_link's x axis below the horizontal (toe-down positive);
  cost ``relu(pitch - max_toe_down)`` per swing foot, in radians, summed over the feet. This is
  the cause behind the toe skimming (the ankle apex is 18 cm; the toe hangs ~9 cm below it), so
  the two terms together let the policy either lift the toe or flatten the foot.
  """
  cmd, idx, in_swing, _ = _feet(env, command_name, swing_height)
  quat = cmd.robot.data.body_link_quat_w[:, idx]  # [B, 2, 4]
  ex = torch.zeros(quat.shape[:-1] + (3,), device=quat.device, dtype=quat.dtype)
  ex[..., 0] = 1.0
  x_w = quat_apply(quat, ex)  # foot x axis in world
  pitch = torch.atan2(-x_w[..., 2], torch.hypot(x_w[..., 0], x_w[..., 1]))  # [B, 2]
  cost = torch.sum(torch.relu(pitch - max_toe_down) * in_swing.float(), dim=1)
  n_swing = torch.clamp(in_swing.float().sum(), min=1.0)
  env.extras["log"]["Metrics/swing_foot_pitch_mean"] = (
    torch.sum(pitch * in_swing.float()) / n_swing
  )
  return cost * _twist_active(env, command_name, command_threshold)
