"""Crawling-specific rewards.

The BeyondMimic tracking rewards already yield speed control (imitating the speed-selected clip),
but the command is continuous while the library is discrete; ``forward_speed_tracking`` nudges the
policy toward the exact commanded speed.

The rest of this module implements the v2 "belly rest": when the command is below the stop
threshold the env is flagged ``is_stopped`` (see commands.py). ``gate_moving`` zeroes the imitation
rewards for those envs so the frozen crawl pose stops paying out, and the ``rest_*`` rewards take
over to drive a low-effort prone hold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

import torch

from .commands import LibraryMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _cmd(env: ManagerBasedRlEnv, command_name: str) -> LibraryMotionCommand:
  return cast(LibraryMotionCommand, env.command_manager.get_term(command_name))


def forward_speed_tracking(
  env: ManagerBasedRlEnv, command_name: str, std: float, axis: int = 1
) -> torch.Tensor:
  """exp(-(v_anchor[axis] - commanded_speed)^2 / std^2); axis=1 -> world +Y (crawl forward)."""
  cmd = _cmd(env, command_name)
  v_fwd = cmd.robot_anchor_lin_vel_w[:, axis]
  return torch.exp(-torch.square(v_fwd - cmd.speed_command) / (std**2))


# --- v2 belly-rest -----------------------------------------------------------------------------

def gate_moving(base_func: Callable) -> Callable:
  """Wrap a reward func so it pays out only for MOVING (non-stopped) envs, else 0.

  Used to switch the imitation rewards off while an env is in belly rest; the wrapped func is
  called by the reward manager as ``func(env, **params)`` and must have ``command_name`` in params.
  """

  def wrapped(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
    cmd = _cmd(env, params["command_name"])
    return base_func(env, **params) * (~cmd.is_stopped).float()

  wrapped.__name__ = f"gated_{getattr(base_func, '__name__', 'reward')}"
  return wrapped


def rest_base_height(
  env: ManagerBasedRlEnv, command_name: str, target_height: float, std: float
) -> torch.Tensor:
  """When stopped, reward the anchor (torso) height near a low target -> get on the belly."""
  cmd = _cmd(env, command_name)
  h = cmd.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return torch.exp(-torch.square(h - target_height) / (std**2)) * cmd.is_stopped.float()


def rest_effort(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """When stopped, penalize actuator force (L2) -> minimize effort, i.e. relax onto the ground."""
  cmd = _cmd(env, command_name)
  force = cmd.robot.data.actuator_force
  return torch.sum(torch.square(force), dim=1) * cmd.is_stopped.float()


def rest_stillness(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """When stopped, penalize base linear + angular velocity -> hold still (velocities -> 0)."""
  cmd = _cmd(env, command_name)
  v = cmd.robot.data.root_link_lin_vel_w
  w = cmd.robot.data.root_link_ang_vel_w
  pen = torch.sum(torch.square(v), dim=1) + torch.sum(torch.square(w), dim=1)
  return pen * cmd.is_stopped.float()
