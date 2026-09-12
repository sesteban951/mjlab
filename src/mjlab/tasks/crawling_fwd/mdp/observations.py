"""Crawling-specific observation terms: the commanded twist and a phase clock."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from .commands import LibraryMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def commanded_twist(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Per-env commanded twist [vx, vy, wz], shape (num_envs, 3)."""
  cmd = cast(LibraryMotionCommand, env.command_manager.get_term(command_name))
  return cmd.twist_command.view(env.num_envs, 3)


def motion_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Phase clock as (sin, cos) of 2*pi*time_step/T, blanked to (0, 0) when idle.

  An idle clip is one pose repeated for the whole period, so the reference holds perfectly still --
  but the clock keeps turning, and a policy trained on periodic gaits reads a rotating phase as
  "swing your limbs". That is what makes a stopped robot shuffle in place instead of standing
  statically, and it is the ONLY drive left once the commanded twist is exactly zero.

  Blanking to (0, 0) is what the velocity task does with its own clock
  (``velocity_custom.mdp.observations.phase``). (0, 0) is off the unit circle, so it is unreachable
  during normal gait and reads as an unambiguous "no gait" flag rather than "gait paused at some
  angle" -- which a frozen point ON the circle would be indistinguishable from.

  Blanks the OBSERVATION only; ``time_steps`` keeps advancing. Every frame of an idle clip is
  identical, so the counter is already a no-op for the reference, and freezing it would perturb the
  transition blend and the adaptive-sampling bins for no gain.
  """
  cmd = cast(LibraryMotionCommand, env.command_manager.get_term(command_name))
  phase = cmd.time_steps.float() / max(int(cmd.motion.time_step_total), 1)
  ang = 2.0 * math.pi * phase
  obs = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)
  return torch.where(cmd.is_idle.unsqueeze(1), torch.zeros_like(obs), obs)
