"""Twist-conditioned gait-library motion command for crawling.

Extends the BeyondMimic ``MotionCommand`` (per-frame reference tracking of a single clip) to a
LIBRARY of phase-aligned periodic gaits, each labelled with a planar twist ``[vx, vy, wz]``
(m/s, m/s, rad/s) stored in the clip's npz. All clips share the same period/frame count ``T``, so
they stack into a leading ``(S, T, ...)`` clip axis. Each env samples a target twist (on a timer,
velocity-style, plus a ``rel_static_envs`` fraction that commands a zero twist), snaps it to the
nearest library clip (``clip_idx``) under a weighted-L2 metric, and gathers the reference with a
per-env ``[clip_idx, time_steps]`` pair. All gaits share one period, so a mid-episode resample keeps
the phase clock and just snaps ``clip_idx`` to the new twist (no teleport); the phase wraps mod ``T``
so a clip loops to fill an episode. RSI (reference-state init) is applied only at episode reset.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.managers import CommandTerm
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class LibraryMotionLoader:
  """Loads all tracking-format clips in a directory, stacked along a leading clip axis.

  Same per-clip schema as ``tracking.mdp.commands.MotionLoader`` (joint_pos/joint_vel +
  body_{pos,quat,lin_vel,ang_vel}_w), but every array gains a leading ``(S,)`` dimension. Each clip
  also carries a ``twist`` label ``[vx, vy, wz]`` (m/s, m/s, rad/s), stacked into ``lib_twists
  (S, 3)``; clips are ordered by filename (deterministic)."""

  def __init__(
    self, motion_dir: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    files = sorted(glob.glob(os.path.join(motion_dir, "*.npz")))
    if not files:
      raise FileNotFoundError(f"no .npz clips found in motion_dir: {motion_dir}")

    def stack(key: str) -> torch.Tensor:
      arrs = [np.asarray(np.load(f)[key], dtype=np.float32) for f in files]
      counts = {a.shape[0] for a in arrs}
      if len(counts) != 1:
        raise ValueError(f"clips have differing frame counts for '{key}': {counts}")
      return torch.tensor(np.stack(arrs, axis=0), dtype=torch.float32, device=device)

    self.joint_pos = stack("joint_pos")  # (S, T, nj)
    self.joint_vel = stack("joint_vel")  # (S, T, nj)
    self._body_pos_w = stack("body_pos_w")  # (S, T, nbody, 3)
    self._body_quat_w = stack("body_quat_w")
    self._body_lin_vel_w = stack("body_lin_vel_w")
    self._body_ang_vel_w = stack("body_ang_vel_w")

    self._body_indexes = body_indexes
    self.body_pos_w = self._body_pos_w[:, :, self._body_indexes]  # (S, T, K, 3)
    self.body_quat_w = self._body_quat_w[:, :, self._body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, :, self._body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, :, self._body_indexes]

    twists = [
      np.asarray(np.load(f)["twist"], dtype=np.float32).reshape(3) for f in files
    ]
    self.lib_twists = torch.tensor(
      np.stack(twists, axis=0), dtype=torch.float32, device=device
    )  # (S, 3) = [vx, vy, wz]
    self.num_clips = len(files)
    self.time_step_total = int(self.joint_pos.shape[1])  # T


class LibraryMotionCommand(MotionCommand):
  """Per-frame reference tracking over a twist-indexed gait library (see module docstring)."""

  cfg: LibraryMotionCommandCfg

  def __init__(self, cfg: LibraryMotionCommandCfg, env: ManagerBasedRlEnv):
    # Skip MotionCommand.__init__ (it builds a single-clip MotionLoader from cfg.motion_file);
    # replicate its body with a LibraryMotionLoader + per-env twist state instead.
    CommandTerm.__init__(self, cfg, env)

    self.robot = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = LibraryMotionLoader(
      cfg.motion_dir, self.body_indexes, device=self.device
    )

    self.twist_metric_weights = torch.tensor(
      cfg.twist_metric_weights, dtype=torch.float32, device=self.device
    )  # (3,) weights the [vx, vy, wz] nearest-clip metric (units differ)

    # A positive static fraction commands a zero twist for some envs, so the library MUST contain a
    # zero-twist (idle) clip for them to snap to (see scripts/library_to_npz.py).
    if cfg.rel_static_envs > 0.0:
      lib_twist_norm = torch.linalg.norm(
        self.motion.lib_twists * self.twist_metric_weights, dim=-1
      )
      assert bool(torch.any(lib_twist_norm < 1e-4)), (
        "rel_static_envs > 0 requires a zero-twist idle clip in motion_dir "
        f"(e.g. crawl_fwd_vx_000.npz); found twists {self.motion.lib_twists.tolist()}"
      )

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.clip_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.twist_command = torch.zeros(self.num_envs, 3, device=self.device)
    # RSI (reference-state init) is applied only when resampling at an episode reset, not on the
    # mid-episode timer resample -- so the robot physically transitions between twists.
    self._rsi_on_resample = False

    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
    self.bin_failed_count = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    for key in (
      "error_anchor_pos",
      "error_anchor_rot",
      "error_anchor_lin_vel",
      "error_anchor_ang_vel",
      "error_body_pos",
      "error_body_rot",
      "error_joint_pos",
      "error_joint_vel",
      "sampling_entropy",
      "sampling_top1_prob",
      "sampling_top1_bin",
    ):
      self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  # --- twist sampling ---------------------------------------------------------------------------

  def _resample_twist(self, env_ids: torch.Tensor) -> None:
    """Sample a target twist ``[vx, vy, wz]`` per env and snap it to the nearest library clip.

    Each component is drawn uniformly from its own range in ``twist_command_range``. A
    ``rel_static_envs`` fraction of envs are instead assigned a zero twist (mirroring the velocity
    task's standing envs) so they snap to the zero-twist idle clip and hold the static pose.
    """
    ranges = torch.tensor(
      self.cfg.twist_command_range, dtype=torch.float32, device=self.device
    )  # (3, 2)
    twist = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=self.device
    )
    if self.cfg.rel_static_envs > 0.0:
      is_static = (
        torch.rand(len(env_ids), device=self.device) < self.cfg.rel_static_envs
      )
      twist[is_static] = 0.0
    self.twist_command[env_ids] = twist
    # Nearest clip under the weighted-L2 twist metric.
    dist = (
      (self.motion.lib_twists[None] - twist[:, None]) ** 2 * self.twist_metric_weights
    ).sum(dim=-1)  # (n_envs, n_clips)
    self.clip_idx[env_ids] = torch.argmin(dist, dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Always re-roll the twist -> clip selection. Pick clip_idx BEFORE any RSI so the base reads
    # body_pos_w[env_ids, 0] from the newly selected clip.
    self._resample_twist(env_ids)
    # RSI (start-frame sampling + teleport to the reference pose) only at episode reset. On the
    # mid-episode timer resample we keep the phase clock and the robot's state, so it physically
    # transitions to the new twist's gait (all gaits share one period, so clip_idx just swaps).
    if self._rsi_on_resample:
      super()._resample_command(env_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
    # Episode reset: resample the twist AND RSI the robot into the selected clip's start pose.
    self._rsi_on_resample = True
    try:
      return super().reset(env_ids)
    finally:
      self._rsi_on_resample = False

  def _update_command(self) -> None:
    # Advance the shared phase clock and LOOP (wrap mod T) -- periodic gaits, no resample/RSI at the
    # clip boundary (twist resampling is timer-driven; see _resample_command).
    self.time_steps += 1
    self.time_steps %= self.motion.time_step_total
    self.update_relative_body_poses()
    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  # --- reference accessors: index [clip_idx, time_steps] instead of [time_steps] ----------------

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.clip_idx, self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.clip_idx, self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.clip_idx, self.time_steps]
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.clip_idx, self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.clip_idx, self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.clip_idx, self.time_steps]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[
        self.clip_idx, self.time_steps, self.motion_anchor_body_index
      ]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[
      self.clip_idx, self.time_steps, self.motion_anchor_body_index
    ]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[
      self.clip_idx, self.time_steps, self.motion_anchor_body_index
    ]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[
      self.clip_idx, self.time_steps, self.motion_anchor_body_index
    ]


@dataclass(kw_only=True)
class LibraryMotionCommandCfg(MotionCommandCfg):
  """Config for :class:`LibraryMotionCommand`.

  Replaces the single ``motion_file`` with a ``motion_dir`` of twist-labelled clips and adds the
  twist command ranges. ``motion_file`` is inherited but unused (kept defaulted)."""

  motion_dir: str = ""
  # Per-component sample range for the commanded twist [(vx_lo, vx_hi), (vy_..), (wz_..)]. Each env
  # draws each component uniformly, then snaps to the nearest library clip.
  twist_command_range: tuple[
    tuple[float, float], tuple[float, float], tuple[float, float]
  ] = ((0.08, 0.40), (0.0, 0.0), (0.0, 0.0))
  # Relative weights for the [vx, vy, wz] nearest-clip metric (units differ: m/s vs rad/s), so a
  # dense twist grid snaps sensibly across components.
  twist_metric_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
  # Fraction of resamples assigned a zero twist -> the static idle clip (mirrors the velocity task's
  # rel_standing_envs). 0 disables the static pose; > 0 requires a zero-twist clip in motion_dir.
  rel_static_envs: float = 0.0
  motion_file: str = (
    ""  # unused (LibraryMotionLoader reads motion_dir); keep to satisfy the base
  )

  def build(self, env: ManagerBasedRlEnv) -> LibraryMotionCommand:
    return LibraryMotionCommand(self, env)
