"""Crawling-specific observation terms: the commanded speed and a phase clock."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from .commands import LibraryMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def commanded_speed(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Per-env commanded forward speed, shape (num_envs, 1)."""
  cmd = cast(LibraryMotionCommand, env.command_manager.get_term(command_name))
  return cmd.speed_command.view(env.num_envs, 1)


def motion_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Phase clock as (sin, cos) of 2*pi*time_step/T, shape (num_envs, 2)."""
  cmd = cast(LibraryMotionCommand, env.command_manager.get_term(command_name))
  phase = cmd.time_steps.float() / max(int(cmd.motion.time_step_total), 1)
  ang = 2.0 * math.pi * phase
  return torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)
