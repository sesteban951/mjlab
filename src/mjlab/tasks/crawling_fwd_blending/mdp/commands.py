"""Twist-conditioned gait-library command with blended clip transitions.

On a mid-episode twist resample the reference is not hard-switched: the old and new
clip are sampled at the shared phase IN THEIR HEADING FRAMES and interpolated by
``blend_alpha``, which ramps 0 -> 1 over ``blend_time_s``. Positions and joints lerp,
orientations nlerp, and velocities are the lerp plus ``alpha_dot * (x_b - x_a)`` so they
stay the derivative of the served pose. The reference frame advances by the lerped
planar step, so path and heading ramp smoothly. The commanded twist still steps at the
resample: only the imitation reference blends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.crawling_fwd.mdp.commands import (
  LibraryMotionCommand as FwdMotionCommand,
)
from mjlab.tasks.crawling_fwd.mdp.commands import (
  LibraryMotionCommandCfg as FwdMotionCommandCfg,
)
from mjlab.tasks.crawling_fwd.mdp.commands import _phase_remap
from mjlab.utils.lab_api.math import quat_box_minus

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class BlendingMotionCommand(FwdMotionCommand):
  """Twist-indexed gait-library command that blends across clip transitions (see module docstring)."""

  cfg: BlendingMotionCommandCfg

  def __init__(self, cfg: BlendingMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    # Blend state: clip we are fading FROM, and the fade weight (0 = prev clip, 1 = current clip).
    self.clip_idx_prev = self.clip_idx.clone()
    self.blend_alpha = torch.ones(self.num_envs, device=self.device)
    # Transition length in control steps (>=1). blend_alpha reaches 1 after this many steps.
    self.blend_steps = max(1, round(cfg.blend_time_s / env.step_dt))

  # --- transition bookkeeping -------------------------------------------------------------------

  def _resample_reference(self, env_ids: torch.Tensor) -> None:
    if not self._rsi_on_resample:
      # Timer resample: start a blend FROM the currently-active clip.
      self.clip_idx_prev[env_ids] = self.clip_idx[env_ids]
      self.blend_alpha[env_ids] = 0.0
    else:
      # Episode reset: RSI teleports through the BLENDED accessors, so close any blend
      # still open from the ending episode FIRST.
      self.blend_alpha[env_ids] = 1.0
    super()._resample_reference(env_ids)  # re-rolls twist -> clip_idx (+ RSI)
    if self._rsi_on_resample:
      # Episode reset: RSI teleported the robot onto the target clip -> no blend.
      self.clip_idx_prev[env_ids] = self.clip_idx[env_ids]
      self.blend_alpha[env_ids] = 1.0

  def _advance_clock(self, ids: torch.Tensor | slice) -> None:
    super()._advance_clock(ids)
    self.blend_alpha[ids] = (self.blend_alpha[ids] + 1.0 / self.blend_steps).clamp_max(
      1.0
    )

  # --- blended gather helpers -------------------------------------------------------------------

  def _alpha_dot(self) -> torch.Tensor:
    """d(blend_alpha)/dt while a blend is running, 0 once it has reached 1. (N,)"""
    rate = 1.0 / (self.blend_steps * self._env.step_dt)
    return torch.where(
      self.blend_alpha < 1.0,
      torch.full_like(self.blend_alpha, rate),
      torch.zeros_like(self.blend_alpha),
    )

  @property
  def time_steps_prev(self) -> torch.Tensor:
    """``time_steps`` mapped onto the PREVIOUS clip's length. (N,)

    Identical to ``time_steps`` when both clips are the same length, which is every
    single-period library. In a ragged one the outgoing and incoming clips have
    different strides, so blending them at the same frame NUMBER would compare different
    points of the cycle -- and would index past the shorter clip's real frames into the
    loader's padding. The blend is defined at a shared PHASE. This is the NEAREST frame
    holding that phase (for bounds and diagnostics); the blend itself reads the clip at
    the fractional frame, see ``_prev_frame``."""
    if not self.motion.ragged:
      return self.time_steps
    return _phase_remap(
      self.time_steps,
      self.motion.n_frames[self.clip_idx],
      self.motion.n_frames[self.clip_idx_prev],
    )

  def _prev_frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The OUTGOING clip's frame at the shared phase as ``(lo, hi, frac)``, for linear
    interpolation between its two bracketing frames.

    ``time_steps_prev`` rounds this to ONE frame; read there, an outgoing clip of another
    stride advances 1-or-2 frames per step, which jitters against the smooth scaled
    velocity ``_lerp_vel`` serves. Interpolating makes the outgoing pose move at exactly
    ``_prev_rate`` frames per step. In a single-period library ``frac`` is 0 and ``lo``
    is ``time_steps``. Pose and velocity are both interpolated linearly, so they agree
    to within half a frame's velocity change."""
    n_prev = self.motion.n_frames[self.clip_idx_prev]
    if not self.motion.ragged:
      return self.time_steps, self.time_steps, torch.zeros_like(self.blend_alpha)
    n_cur = self.motion.n_frames[self.clip_idx].float()
    x = self.time_steps.float() / n_cur * n_prev.float()
    lo = x.floor().long()
    return lo % n_prev, (lo + 1) % n_prev, x - lo.float()

  def _gather_prev(self, arr: torch.Tensor) -> torch.Tensor:
    """The outgoing clip's ``arr`` sample at the shared phase (see ``_prev_frame``)."""
    if not self.motion.ragged:
      return arr[self.clip_idx_prev, self.time_steps]
    lo, hi, frac = self._prev_frame()
    a0, a1 = arr[self.clip_idx_prev, lo], arr[self.clip_idx_prev, hi]
    return torch.lerp(a0, a1, frac.view(-1, *([1] * (a0.ndim - 1))).to(a0.dtype))

  def _gather_prev_quat(self, arr: torch.Tensor) -> torch.Tensor:
    """``_gather_prev`` for quaternions: hemisphere-corrected nlerp between the
    bracketing frames (adjacent frames of one clip are close, so nlerp tracks the
    arc)."""
    if not self.motion.ragged:
      return arr[self.clip_idx_prev, self.time_steps]
    lo, hi, frac = self._prev_frame()
    a0, a1 = arr[self.clip_idx_prev, lo], arr[self.clip_idx_prev, hi]
    a1 = torch.where((a0 * a1).sum(-1, keepdim=True) < 0, -a1, a1)
    q = torch.lerp(a0, a1, frac.view(-1, *([1] * (a0.ndim - 1))))
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

  def _cur(self, arr: torch.Tensor) -> torch.Tensor:
    """The incoming clip's ``arr`` sample."""
    return arr[self.clip_idx, self.time_steps]

  def _w(self, ndim: int) -> torch.Tensor:
    """blend_alpha broadcast to a gathered tensor of rank ``ndim`` ((N,) -> (N, 1, ...))."""
    return self.blend_alpha.view(-1, *([1] * (ndim - 1)))

  def _lerp(self, arr: torch.Tensor) -> torch.Tensor:
    a = self._gather_prev(arr)
    return torch.lerp(a, self._cur(arr), self._w(a.ndim).to(a.dtype))

  def _nlerp(self, arr: torch.Tensor) -> torch.Tensor:
    a, b = self._gather_prev_quat(arr), self._cur(arr)
    b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)  # shortest-arc hemisphere
    q = torch.lerp(a, b, self._w(a.ndim))
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

  def _ramp(self, delta: torch.Tensor) -> torch.Tensor:
    """``alpha_dot * delta``: the velocity the blend ramp itself adds to a served pose."""
    return self._alpha_dot().view(-1, *([1] * (delta.ndim - 1))) * delta

  def _prev_rate(self) -> torch.Tensor:
    """Frames per env step at which the OUTGOING clip is played during a blend:
    ``n_prev / n_cur`` (see ``time_steps_prev``). Exactly 1 in a single-period library.
    (N,)"""
    if not self.motion.ragged:
      return torch.ones_like(self.blend_alpha)
    n_prev = self.motion.n_frames[self.clip_idx_prev].float()
    return n_prev / self.motion.n_frames[self.clip_idx].float()

  def _lerp_vel(self, arr: torch.Tensor) -> torch.Tensor:
    """Blend a per-body velocity array. The outgoing clip is played at ``_prev_rate``
    frames per step, so its ANCHOR-RELATIVE velocity is scaled by that rate. Its
    anchor's own velocity stays native, like the planar step the path is built from."""
    i = self.motion_anchor_body_index
    a = self._gather_prev(arr)
    a = a[:, i : i + 1] + self._prev_rate().view(-1, 1, 1) * (a - a[:, i : i + 1])
    return torch.lerp(a, self._cur(arr), self._w(a.ndim))

  # --- the served sample: blended [clip_idx_prev -> clip_idx] at the shared phase ---

  def _joint_pos(self) -> torch.Tensor:
    return self._lerp(self.motion.joint_pos)

  def _joint_vel(self) -> torch.Tensor:
    jp = self.motion.joint_pos
    a = self._gather_prev(self.motion.joint_vel) * self._prev_rate()[:, None]
    vel = torch.lerp(a, self._cur(self.motion.joint_vel), self._w(a.ndim))
    return vel + self._ramp(self._cur(jp) - self._gather_prev(jp))

  def _local_pos(self) -> torch.Tensor:
    return self._lerp(self.motion.body_pos_l)

  def _local_quat(self) -> torch.Tensor:
    return self._nlerp(self.motion.body_quat_l)

  def _local_lin_vel(self) -> torch.Tensor:
    pos = self.motion.body_pos_l
    delta = self._cur(pos) - self._gather_prev(pos)
    return self._lerp_vel(self.motion.body_lin_vel_l) + self._ramp(delta)

  def _local_ang_vel(self) -> torch.Tensor:
    quat = self.motion.body_quat_l
    a, b = self._gather_prev_quat(quat), self._cur(quat)
    b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)  # as in _nlerp
    delta = quat_box_minus(b.reshape(-1, 4), a.reshape(-1, 4)).reshape(*a.shape[:-1], 3)
    return self._lerp_vel(self.motion.body_ang_vel_l) + self._ramp(delta)

  def _planar_step(self) -> tuple[torch.Tensor, torch.Tensor]:
    return (
      self._lerp(self.motion.anchor_step_xy),
      self._lerp(self.motion.anchor_step_yaw),
    )


@dataclass(kw_only=True)
class BlendingMotionCommandCfg(FwdMotionCommandCfg):
  """Config for :class:`BlendingMotionCommand`: adds the transition-blend window."""

  # Length of the reference blend on a mid-episode twist resample (seconds). Rounded to control
  # steps at build time; the reference fades from the old clip to the new one over this window.
  blend_time_s: float = 0.4

  def build(self, env: ManagerBasedRlEnv) -> BlendingMotionCommand:
    return BlendingMotionCommand(self, env)
