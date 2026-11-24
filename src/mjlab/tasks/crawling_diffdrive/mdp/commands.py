"""Differential-drive twist command for the crawl gait library.

Subclasses the gait library's :class:`BlendingMotionCommand` and overrides only the twist
sampler: on every resample each env is idle ``[0, 0, 0]`` (prob ``rel_static_envs``),
turns in place ``[0, 0, wz]`` (prob ``rel_turn_envs``) or drives straight ``[vx, 0, 0]``,
forward or backward per ``rel_back_envs``. Never a mixed arc, never lateral. The
nearest-clip snap is unchanged: straight clips have wz = 0 and turn clips vx = 0, so a
command lands on a clip of its own mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.gait_library.mdp.blending import (
  BlendingMotionCommand,
  BlendingMotionCommandCfg,
)
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class DiffDriveMotionCommand(BlendingMotionCommand):
  """Tank-style crawl command: translate ``[±vx,0,0]`` XOR rotate ``[0,0,wz]`` (see module docstring)."""

  cfg: DiffDriveMotionCommandCfg

  def _resample_twist(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    dev = self.device
    twist = torch.zeros(n, 3, device=dev)

    # One mode per env: [0, p_static) idle | [p_static, p_static+p_turn) turn | else straight.
    p_static = float(self.cfg.rel_static_envs)
    p_turn = float(self.cfg.rel_turn_envs)
    mode = torch.rand(n, device=dev)
    is_turn = (mode >= p_static) & (mode < p_static + p_turn)
    is_straight = mode >= (p_static + p_turn)

    # Straight mode: forward (vx>0) or backward (vx<0); rel_back_envs picks the direction. The two
    # ranges are independent because the forward and backward libraries are not symmetric.
    vx_fwd = sample_uniform(
      self.cfg.vx_fwd_range[0], self.cfg.vx_fwd_range[1], (n,), device=dev
    )
    vx_bck = sample_uniform(
      self.cfg.vx_bck_range[0], self.cfg.vx_bck_range[1], (n,), device=dev
    )
    go_back = torch.rand(n, device=dev) < float(self.cfg.rel_back_envs)
    vx = torch.where(go_back, vx_bck, vx_fwd)
    twist[:, 0] = torch.where(is_straight, vx, twist[:, 0])

    # Turn mode: wz magnitude in wz_range, random sign, everything else 0.
    wz = sample_uniform(self.cfg.wz_range[0], self.cfg.wz_range[1], (n,), device=dev)
    sign = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    twist[:, 2] = torch.where(is_turn, wz * sign, twist[:, 2])

    self.twist_command[env_ids] = twist

    # Snap through the base so the ragged-library phase remap runs: the clip must never
    # change without time_steps being rescaled onto its length (see _snap_to_library).
    self._snap_to_library(env_ids)


@dataclass(kw_only=True)
class DiffDriveMotionCommandCfg(BlendingMotionCommandCfg):
  """Config for :class:`DiffDriveMotionCommand`: replaces the per-axis twist range with a
  differential-drive mode split (straight fwd/bck vx-ranges + in-place wz-range + idle fraction)."""

  # Straight-mode speed ranges [m/s]: forward (vx>0) and backward (vx<0), sampled independently.
  vx_fwd_range: tuple[float, float] = (0.10, 0.30)
  vx_bck_range: tuple[float, float] = (-0.25, -0.05)
  # Turn-mode commanded yaw-rate MAGNITUDE range [rad/s]; the sign is randomized each resample.
  wz_range: tuple[float, float] = (0.25, 0.55)
  # Fraction of resamples assigned the turn mode (rel_static_envs, inherited, is the idle fraction;
  # the remainder go straight). Of the straight envs, rel_back_envs is the fraction driven backward.
  rel_turn_envs: float = 0.4
  rel_back_envs: float = 0.5

  def build(self, env: ManagerBasedRlEnv) -> DiffDriveMotionCommand:
    return DiffDriveMotionCommand(self, env)
