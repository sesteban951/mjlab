"""Twist-conditioned gait-library command with blended clip transitions.

Subclasses ``crawling_fwd``'s :class:`LibraryMotionCommand`. The only change: on a mid-episode
(timer) twist resample, the tracked reference is not hard-switched to the new clip -- both the old
and new clip are gathered at the shared phase and interpolated by ``blend_alpha``, which ramps 0->1
over ``blend_time_s``. Positions/joints use linear interpolation; orientations use a normalized-lerp
(nlerp, a cheap slerp approximation that is accurate for the small orientation gaps within a short
window). Velocities are the lerp PLUS the time derivative of the ramp, ``alpha_dot * (x_b - x_a)``,
so the served velocity is the derivative of the served position; the body terms take that
difference relative to the anchor, whose own velocity stays the plain lerp (the egocentric path and
heading targets integrate it). Episode-reset RSI is unaffected (blend_alpha starts at 1 -> no blend).

The commanded-twist observation (``twist_command``, read by ``commanded_twist`` / ``twist_tracking``)
deliberately still steps at the resample -- only the imitation *reference* blends -- so the policy
learns an intrinsic graceful transition instead of merely tracking a ramped command.
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

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if not self._rsi_on_resample:
      # Timer resample: start a blend FROM the currently-active clip.
      self.clip_idx_prev[env_ids] = self.clip_idx[env_ids]
      self.blend_alpha[env_ids] = 0.0
    else:
      # Episode reset: RSI teleports through the BLENDED accessors, so close any blend
      # still open from the ending episode FIRST, or the written pose is a lerp with a
      # stale clip.
      self.blend_alpha[env_ids] = 1.0
    super()._resample_command(env_ids)  # re-rolls twist -> clip_idx (+ RSI/rebase)
    if self._rsi_on_resample:
      # Episode reset: RSI teleported the robot onto the target clip -> no blend.
      self.clip_idx_prev[env_ids] = self.clip_idx[env_ids]
      self.blend_alpha[env_ids] = 1.0

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    super()._update_command(
      env_ids
    )  # advance phase + egocentric target (blended anchor vel below)
    # Scoped like the base's phase clock: a partial reset must not advance other envs' blends.
    ids = slice(None) if env_ids is None else env_ids
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

    ``time_steps_prev`` rounds this to ONE frame. Read there, an outgoing clip whose
    stride differs advances 1-or-2 frames per step (70/43 = 1.63) in a stair-step, whose
    finite difference jitters by about a frame per step against the smooth scaled
    velocity ``_lerp_vel`` serves. Interpolating removes that: the served outgoing
    position then moves at exactly ``_prev_rate`` frames per step. In a single-period
    library ``frac`` is 0 and ``lo`` is ``time_steps``, so this is the plain gather.

    Positions and velocities are both interpolated LINEARLY, so the served velocity
    matches the served position's derivative to within half a frame's velocity change
    (measured 0.1-0.2 rad/s in joints on the walk/jog clips, vs 0.01-0.05 at an integer
    rate). Cubic Hermite with the stored velocities as tangents would close that gap if
    it ever matters. Separately, these clips are not exactly periodic: a finite
    difference across a clip's wrap frame spikes (up to 1.65 rad/s on the jog), which is
    a property of the data and hits single-period libraries the same way."""
    n_prev = self.motion.n_frames[self.clip_idx_prev]
    if not self.motion.ragged:
      return self.time_steps, self.time_steps, torch.zeros_like(self.blend_alpha)
    n_cur = self.motion.n_frames[self.clip_idx].float()
    x = self.time_steps.float() / n_cur * n_prev.float()
    lo = x.floor().long()
    return lo % n_prev, (lo + 1) % n_prev, x - lo.float()

  def _gather_prev(self, arr: torch.Tensor, i: int | None = None) -> torch.Tensor:
    """The outgoing clip's ``arr`` sample at the shared phase (see ``_prev_frame``)."""
    lo, hi, frac = self._prev_frame()
    if i is None:
      a0, a1 = arr[self.clip_idx_prev, lo], arr[self.clip_idx_prev, hi]
    else:
      a0, a1 = arr[self.clip_idx_prev, lo, i], arr[self.clip_idx_prev, hi, i]
    return torch.lerp(a0, a1, frac.view(-1, *([1] * (a0.ndim - 1))))

  def _gather_prev_quat(self, arr: torch.Tensor, i: int | None = None) -> torch.Tensor:
    """``_gather_prev`` for quaternions: hemisphere-corrected nlerp between the
    bracketing frames (adjacent frames of one clip are close, so nlerp tracks the
    arc)."""
    lo, hi, frac = self._prev_frame()
    if i is None:
      a0, a1 = arr[self.clip_idx_prev, lo], arr[self.clip_idx_prev, hi]
    else:
      a0, a1 = arr[self.clip_idx_prev, lo, i], arr[self.clip_idx_prev, hi, i]
    a1 = torch.where((a0 * a1).sum(-1, keepdim=True) < 0, -a1, a1)
    q = torch.lerp(a0, a1, frac.view(-1, *([1] * (a0.ndim - 1))))
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

  def _dpos_rel(self) -> torch.Tensor:
    """(x_b - x_a) of every body position, minus the anchor's own difference. (N, B, 3)"""
    a = self._gather_prev(self.motion.body_pos_w)
    b = self.motion.body_pos_w[self.clip_idx, self.time_steps]
    d = b - a
    return d - d[:, self.motion_anchor_body_index : self.motion_anchor_body_index + 1]

  def _drot_rel(self) -> torch.Tensor:
    """log(q_b q_a^-1) of every body, world frame, minus the anchor's. (N, B, 3)"""
    a = self._gather_prev_quat(self.motion.body_quat_w)
    b = self.motion.body_quat_w[self.clip_idx, self.time_steps]
    b = torch.where(
      (a * b).sum(-1, keepdim=True) < 0, -b, b
    )  # same hemisphere as _nlerp
    n, nb = a.shape[:2]
    d = quat_box_minus(b.reshape(-1, 4), a.reshape(-1, 4)).reshape(n, nb, 3)
    return d - d[:, self.motion_anchor_body_index : self.motion_anchor_body_index + 1]

  def _w(self, ndim: int) -> torch.Tensor:
    """blend_alpha broadcast to a gathered tensor of rank ``ndim`` ((N,) -> (N, 1, ...))."""
    return self.blend_alpha.view(-1, *([1] * (ndim - 1)))

  def _lerp(self, arr: torch.Tensor) -> torch.Tensor:
    a = self._gather_prev(arr)
    b = arr[self.clip_idx, self.time_steps]
    return torch.lerp(a, b, self._w(a.ndim))

  def _nlerp(self, arr: torch.Tensor) -> torch.Tensor:
    a = self._gather_prev_quat(arr)
    b = arr[self.clip_idx, self.time_steps]
    b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)  # shortest-arc hemisphere
    q = torch.lerp(a, b, self._w(a.ndim))
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

  def _lerp_anchor(self, arr: torch.Tensor) -> torch.Tensor:
    i = self.motion_anchor_body_index
    a = self._gather_prev(arr, i)
    b = arr[self.clip_idx, self.time_steps, i]
    return torch.lerp(a, b, self._w(a.ndim))

  def _nlerp_anchor(self, arr: torch.Tensor) -> torch.Tensor:
    i = self.motion_anchor_body_index
    a = self._gather_prev_quat(arr, i)
    b = arr[self.clip_idx, self.time_steps, i]
    b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)
    q = torch.lerp(a, b, self._w(a.ndim))
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

  def _prev_rate(self) -> torch.Tensor:
    """Frames per env step at which the OUTGOING clip is played during a blend:
    ``n_prev / n_cur`` (see ``time_steps_prev``). Exactly 1 in a single-period library.
    (N,)"""
    if not self.motion.ragged:
      return torch.ones_like(self.blend_alpha)
    n_prev = self.motion.n_frames[self.clip_idx_prev].float()
    return n_prev / self.motion.n_frames[self.clip_idx].float()

  def _lerp_vel(self, arr: torch.Tensor, anchor_relative: bool) -> torch.Tensor:
    """Blend a velocity array. The outgoing clip's POSITIONS are served at
    ``_prev_rate`` frames per step, so its velocity is scaled by the same factor to stay
    the derivative of the served position. For per-body world velocities only the
    ANCHOR-RELATIVE part is scaled: the anchor's own velocity stays native, because the
    egocentric path target integrates it and needs the smooth native ramp (see
    ``_update_command``)."""
    a = self._gather_prev(arr)
    b = arr[self.clip_idx, self.time_steps]
    r = self._prev_rate().view(-1, *([1] * (a.ndim - 1)))
    if anchor_relative:
      i = self.motion_anchor_body_index
      anchor = a[:, i : i + 1]
      a = anchor + r * (a - anchor)
    else:
      a = r * a
    return torch.lerp(a, b, self._w(a.ndim))

  # --- reference accessors: blended [clip_idx_prev -> clip_idx] at the shared phase ---

  @property
  def joint_pos(self) -> torch.Tensor:
    return self._lerp(self.motion.joint_pos)

  @property
  def joint_vel(self) -> torch.Tensor:
    dq = self.motion.joint_pos[self.clip_idx, self.time_steps] - self._gather_prev(
      self.motion.joint_pos
    )
    vel = self._lerp_vel(self.motion.joint_vel, anchor_relative=False)
    return vel + self._alpha_dot()[:, None] * dq

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._lerp(self.motion.body_pos_w) + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._nlerp(self.motion.body_quat_w)

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self._lerp_anchor(self.motion.body_pos_w) + self._env.scene.env_origins

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self._nlerp_anchor(self.motion.body_quat_w)

  # Velocities: override only the clip-frame gathers; the base's accessors rotate them into the
  # robot's heading frame.
  def _clip_body_lin_vel_w(self) -> torch.Tensor:
    return (
      self._lerp_vel(self.motion.body_lin_vel_w, anchor_relative=True)
      + self._alpha_dot()[:, None, None] * self._dpos_rel()
    )

  def _clip_body_ang_vel_w(self) -> torch.Tensor:
    return (
      self._lerp_vel(self.motion.body_ang_vel_w, anchor_relative=True)
      + self._alpha_dot()[:, None, None] * self._drot_rel()
    )

  def _clip_anchor_lin_vel_w(self) -> torch.Tensor:
    return self._lerp_anchor(self.motion.body_lin_vel_w)

  def _clip_anchor_ang_vel_w(self) -> torch.Tensor:
    return self._lerp_anchor(self.motion.body_ang_vel_w)


@dataclass(kw_only=True)
class BlendingMotionCommandCfg(FwdMotionCommandCfg):
  """Config for :class:`BlendingMotionCommand`: adds the transition-blend window."""

  # Length of the reference blend on a mid-episode twist resample (seconds). Rounded to control
  # steps at build time; the reference fades from the old clip to the new one over this window.
  blend_time_s: float = 0.4

  def build(self, env: ManagerBasedRlEnv) -> BlendingMotionCommand:
    return BlendingMotionCommand(self, env)
