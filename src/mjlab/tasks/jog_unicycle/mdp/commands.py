"""Unicycle twist command for a jogging gait library.

Subclasses ``crawling_fwd_blending``'s :class:`BlendingMotionCommand` (so mid-episode clip switches
still blend the reference) and overrides ONLY the twist sampler. Where the differential-drive
sampler is translate XOR rotate, a unicycle drives and turns AT THE SAME TIME -- each env is
assigned one of three mutually-exclusive modes on every resample:

  * idle  (prob ``rel_static_envs``):  twist = [0,  0, 0 ]  -> snaps to the zero-twist stop clip
  * turn  (prob ``rel_turn_envs``):    twist = [0,  0, wz]  -> pure in-place pivot (wz signed)
  * arc   (remaining probability):     twist = [vx, 0, wz]  -> jog forward (vx>0) or backward
                                                               (vx<0) WHILE turning

``arc_wz_range`` reaches down to 0, so a straight jog is the ``wz -> 0`` end of the arc mode rather
than a fourth mode; there is no separate "straight" case to keep in sync. Lateral motion is still
never commanded (``vy`` is always 0) -- that is what makes this a unicycle and not a free twist.

WHY THE METRIC WEIGHTS MATTER HERE AND NOT IN THE DIFF-DRIVE. Snapping picks the library clip
minimising ``sum_i w_i (twist_i - clip_i)^2``. The diff-drive library is 1-D within a mode (straight
clips have wz=0, turn clips have vx=0), so the axes never compete and equal weights are fine. A
unicycle library is a genuine 2-D (vx, wz) grid: every candidate differs on both axes at once, and
the weights decide whether 1 m/s of speed error is worth more or less than 1 rad/s of yaw error --
two different units. :func:`span_normalized_weights` makes that choice explicit and range-derived
instead of leaving it to an accident of units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.crawling_fwd_blending.mdp.commands import (
  BlendingMotionCommand,
  BlendingMotionCommandCfg,
)
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def span_normalized_weights(
  vx_range: tuple[float, float], wz_range: tuple[float, float]
) -> tuple[float, float, float]:
  """Twist-metric weights for which a full-range miss costs the same on ``vx`` as on ``wz``.

  The metric is squared, so a weight of ``1/span^2`` normalises each axis to its own commanded
  span; the triple is then rescaled so the largest weight is 1 (the metric is scale-free, this
  only keeps the numbers readable). ``vy`` takes ``vx``'s weight: every clip and every command has
  ``vy = 0``, so the value is inert, but matching ``vx`` keeps it harmless if a lateral clip ever
  enters the library rather than silently free.
  """
  sx = max(abs(vx_range[1] - vx_range[0]), 1e-6)
  sz = max(abs(wz_range[1] - wz_range[0]), 1e-6)
  wx, wz = 1.0 / sx**2, 1.0 / sz**2
  m = max(wx, wz)
  return (wx / m, wx / m, wz / m)


class UnicycleMotionCommand(BlendingMotionCommand):
  """Unicycle jog command: arc ``[±vx, 0, wz]``, pivot ``[0, 0, wz]``, or stop (module docstring)."""

  cfg: UnicycleMotionCommandCfg

  def _resample_twist(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    dev = self.device
    twist = torch.zeros(n, 3, device=dev)

    # One mode per env: [0, p_static) idle | [p_static, p_static+p_turn) turn | else arc.
    p_static = float(self.cfg.rel_static_envs)
    p_turn = float(self.cfg.rel_turn_envs)
    mode = torch.rand(n, device=dev)
    is_turn = (mode >= p_static) & (mode < p_static + p_turn)
    is_arc = mode >= (p_static + p_turn)

    # ARC: speed and yaw rate drawn INDEPENDENTLY, which is the whole difference from the
    # differential drive. Forward and backward ranges are separate because a jog library is not
    # symmetric in vx; the arc yaw magnitude is its own range (usually tighter than a pivot's,
    # since a fast jog cannot turn as hard as a stationary pivot).
    vx_fwd = sample_uniform(
      self.cfg.vx_fwd_range[0], self.cfg.vx_fwd_range[1], (n,), device=dev
    )
    vx_bck = sample_uniform(
      self.cfg.vx_bck_range[0], self.cfg.vx_bck_range[1], (n,), device=dev
    )
    go_back = torch.rand(n, device=dev) < float(self.cfg.rel_back_envs)
    vx = torch.where(go_back, vx_bck, vx_fwd)
    arc_wz = sample_uniform(
      self.cfg.arc_wz_range[0], self.cfg.arc_wz_range[1], (n,), device=dev
    )
    arc_sign = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    twist[:, 0] = torch.where(is_arc, vx, twist[:, 0])
    twist[:, 2] = torch.where(is_arc, arc_wz * arc_sign, twist[:, 2])

    # TURN: pivot in place -- wz magnitude in wz_range, random sign, vx left at 0.
    wz = sample_uniform(self.cfg.wz_range[0], self.cfg.wz_range[1], (n,), device=dev)
    sign = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    twist[:, 2] = torch.where(is_turn, wz * sign, twist[:, 2])

    self.twist_command[env_ids] = twist

    # Snap to the nearest library clip under the weighted-L2 twist metric (same as the base).
    dist = (
      (self.motion.lib_twists[None] - twist[:, None]) ** 2 * self.twist_metric_weights
    ).sum(dim=-1)  # (n_envs, n_clips)
    self.clip_idx[env_ids] = torch.argmin(dist, dim=-1)


@dataclass(kw_only=True)
class UnicycleMotionCommandCfg(BlendingMotionCommandCfg):
  """Config for :class:`UnicycleMotionCommand`: a unicycle mode split (arc fwd/bck vx-ranges with
  their own yaw range + in-place pivot wz-range + idle fraction)."""

  # Arc-mode speed ranges [m/s]: forward (vx>0) and backward (vx<0), sampled independently.
  vx_fwd_range: tuple[float, float] = (1.00, 1.60)
  vx_bck_range: tuple[float, float] = (-0.90, -0.50)
  # Arc-mode yaw-rate MAGNITUDE [rad/s], sign randomized. REACHES 0 on purpose: wz=0 is a straight
  # jog, so the straight case is this mode's lower edge and needs no branch of its own.
  arc_wz_range: tuple[float, float] = (0.0, 1.00)
  # Pivot-mode yaw-rate MAGNITUDE [rad/s]; sign randomized, vx held at 0.
  wz_range: tuple[float, float] = (0.50, 1.50)
  # Fraction of resamples assigned the pivot mode (rel_static_envs, inherited, is the idle
  # fraction; the remainder arc). Of the arcing envs, rel_back_envs is the fraction run backward.
  rel_turn_envs: float = 0.3
  rel_back_envs: float = 0.4

  def build(self, env: ManagerBasedRlEnv) -> UnicycleMotionCommand:
    return UnicycleMotionCommand(self, env)
