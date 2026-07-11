"""Twist-conditioned gait-library command with blended clip transitions.

Subclasses ``crawling_fwd``'s :class:`LibraryMotionCommand`. The only change: on a mid-episode
(timer) twist resample, the tracked reference is not hard-switched to the new clip -- both the old
and new clip are gathered at the shared phase and interpolated by ``blend_alpha``, which ramps 0->1
over ``blend_time_s``. Positions/velocities/joints use linear interpolation; orientations use a
normalized-lerp (nlerp, a cheap slerp approximation that is accurate for the small orientation
gaps within a short window). Episode-reset RSI is unaffected (blend_alpha starts at 1 -> no blend).

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
    super()._resample_command(env_ids)  # re-rolls twist -> clip_idx (+ RSI/rebase)
    if self._rsi_on_resample:
      # Episode reset: RSI teleported the robot onto the target clip -> no blend.
      self.clip_idx_prev[env_ids] = self.clip_idx[env_ids]
      self.blend_alpha[env_ids] = 1.0

  def _update_command(self) -> None:
    super()._update_command()  # advance phase + egocentric target (uses blended anchor vel below)
    self.blend_alpha = (self.blend_alpha + 1.0 / self.blend_steps).clamp_max(1.0)

  # --- blended gather helpers -------------------------------------------------------------------

  def _w(self, ndim: int) -> torch.Tensor:
    """blend_alpha broadcast to a gathered tensor of rank ``ndim`` ((N,) -> (N, 1, ...))."""
    return self.blend_alpha.view(-1, *([1] * (ndim - 1)))

  def _lerp(self, arr: torch.Tensor) -> torch.Tensor:
    a = arr[self.clip_idx_prev, self.time_steps]
    b = arr[self.clip_idx, self.time_steps]
    return torch.lerp(a, b, self._w(a.ndim))

  def _nlerp(self, arr: torch.Tensor) -> torch.Tensor:
    a = arr[self.clip_idx_prev, self.time_steps]
    b = arr[self.clip_idx, self.time_steps]
    b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)  # shortest-arc hemisphere
    q = torch.lerp(a, b, self._w(a.ndim))
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

  def _lerp_anchor(self, arr: torch.Tensor) -> torch.Tensor:
    i = self.motion_anchor_body_index
    a = arr[self.clip_idx_prev, self.time_steps, i]
    b = arr[self.clip_idx, self.time_steps, i]
    return torch.lerp(a, b, self._w(a.ndim))

  def _nlerp_anchor(self, arr: torch.Tensor) -> torch.Tensor:
    i = self.motion_anchor_body_index
    a = arr[self.clip_idx_prev, self.time_steps, i]
    b = arr[self.clip_idx, self.time_steps, i]
    b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)
    q = torch.lerp(a, b, self._w(a.ndim))
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

  # --- reference accessors: blended [clip_idx_prev -> clip_idx] at the shared phase --------------

  @property
  def joint_pos(self) -> torch.Tensor:
    return self._lerp(self.motion.joint_pos)

  @property
  def joint_vel(self) -> torch.Tensor:
    return self._lerp(self.motion.joint_vel)

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._lerp(self.motion.body_pos_w) + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._nlerp(self.motion.body_quat_w)

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._lerp(self.motion.body_lin_vel_w)

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._lerp(self.motion.body_ang_vel_w)

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self._lerp_anchor(self.motion.body_pos_w) + self._env.scene.env_origins

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self._nlerp_anchor(self.motion.body_quat_w)

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self._lerp_anchor(self.motion.body_lin_vel_w)

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self._lerp_anchor(self.motion.body_ang_vel_w)


@dataclass(kw_only=True)
class BlendingMotionCommandCfg(FwdMotionCommandCfg):
  """Config for :class:`BlendingMotionCommand`: adds the transition-blend window."""

  # Length of the reference blend on a mid-episode twist resample (seconds). Rounded to control
  # steps at build time; the reference fades from the old clip to the new one over this window.
  blend_time_s: float = 0.4

  def build(self, env: ManagerBasedRlEnv) -> BlendingMotionCommand:
    return BlendingMotionCommand(self, env)
