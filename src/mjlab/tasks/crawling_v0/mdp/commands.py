"""Velocity-conditioned gait-library motion command for crawling.

Extends the BeyondMimic ``MotionCommand`` (per-frame reference tracking of a single clip) to a
LIBRARY of phase-aligned clips at different forward speeds. All clips share the same period/frame
count ``T``, so they stack into a leading ``(S, T, ...)`` speed axis. Each env samples a target
forward speed on reset, snaps it to the nearest library clip (``speed_idx``), and gathers the
reference with a per-env ``[speed_idx, time_steps]`` pair. Everything else (RSI, adaptive time
sampling, metrics, ghost viz, reference-state reset) is inherited unchanged.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.managers import CommandTerm
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Parse a signed speed from a library filename, e.g. "crawl_vp0_280" -> +0.280, "crawl_vm0_120" -> -0.120.
_SPEED_RE = re.compile(r"_v([pm])(\d+)_(\d+)")


def _parse_speed(filename: str) -> float:
  m = _SPEED_RE.search(os.path.basename(filename))
  if m is None:
    raise ValueError(f"cannot parse speed from filename: {filename!r} (expected '..._v[p|m]I_FFF.npz')")
  sign = 1.0 if m.group(1) == "p" else -1.0
  return sign * (int(m.group(2)) + int(m.group(3)) / 1000.0)


class LibraryMotionLoader:
  """Loads all tracking-format clips in a directory, stacked along a leading speed axis.

  Same per-clip schema as ``tracking.mdp.commands.MotionLoader`` (joint_pos/joint_vel +
  body_{pos,quat,lin_vel,ang_vel}_w), but every array gains a leading ``(S,)`` dimension and the
  clips are sorted by ascending speed (parsed from the filename)."""

  def __init__(self, motion_dir: str, body_indexes: torch.Tensor, device: str = "cpu") -> None:
    files = sorted(glob.glob(os.path.join(motion_dir, "*.npz")))
    if not files:
      raise FileNotFoundError(f"no .npz clips found in motion_dir: {motion_dir}")
    files = sorted(files, key=_parse_speed)  # ascending speed
    speeds = [_parse_speed(f) for f in files]

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

    self.lib_speeds = torch.tensor(speeds, dtype=torch.float32, device=device)  # (S,)
    self.num_speeds = len(speeds)
    self.time_step_total = int(self.joint_pos.shape[1])  # T


class LibraryMotionCommand(MotionCommand):
  """Per-frame reference tracking over a speed-indexed gait library (see module docstring)."""

  cfg: LibraryMotionCommandCfg

  def __init__(self, cfg: LibraryMotionCommandCfg, env: ManagerBasedRlEnv):
    # Skip MotionCommand.__init__ (it builds a single-clip MotionLoader from cfg.motion_file);
    # replicate its body with a LibraryMotionLoader + per-env speed state instead.
    CommandTerm.__init__(self, cfg, env)

    self.robot = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = LibraryMotionLoader(cfg.motion_dir, self.body_indexes, device=self.device)

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.speed_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.speed_command = torch.zeros(self.num_envs, device=self.device)

    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
    self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
    self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    for key in (
      "error_anchor_pos", "error_anchor_rot", "error_anchor_lin_vel", "error_anchor_ang_vel",
      "error_body_pos", "error_body_rot", "error_joint_pos", "error_joint_vel",
      "sampling_entropy", "sampling_top1_prob", "sampling_top1_bin",
    ):
      self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  # --- speed sampling ---------------------------------------------------------------------------

  def _resample_speed(self, env_ids: torch.Tensor) -> None:
    """Sample a target forward speed per env and snap it to the nearest library clip."""
    lo, hi = self.cfg.speed_command_range
    v = sample_uniform(lo, hi, (len(env_ids),), device=self.device)
    self.speed_command[env_ids] = v
    self.speed_idx[env_ids] = torch.argmin(
      torch.abs(self.motion.lib_speeds[None, :] - v[:, None]), dim=1
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Pick the clip (speed_idx) BEFORE the base reads body_pos_w[env_ids, 0] for RSI.
    self._resample_speed(env_ids)
    super()._resample_command(env_ids)

  # --- reference accessors: index [speed_idx, time_steps] instead of [time_steps] ---------------

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.speed_idx, self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.speed_idx, self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.speed_idx, self.time_steps]
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.speed_idx, self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.speed_idx, self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.speed_idx, self.time_steps]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.speed_idx, self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.speed_idx, self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[
      self.speed_idx, self.time_steps, self.motion_anchor_body_index
    ]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[
      self.speed_idx, self.time_steps, self.motion_anchor_body_index
    ]


@dataclass(kw_only=True)
class LibraryMotionCommandCfg(MotionCommandCfg):
  """Config for :class:`LibraryMotionCommand`.

  Replaces the single ``motion_file`` with a ``motion_dir`` of per-speed clips and adds the
  forward-speed command range. ``motion_file`` is inherited but unused (kept defaulted)."""

  motion_dir: str = ""
  speed_command_range: tuple[float, float] = (0.08, 0.40)
  motion_file: str = ""  # unused (LibraryMotionLoader reads motion_dir); keep to satisfy the base

  def build(self, env: ManagerBasedRlEnv) -> LibraryMotionCommand:
    return LibraryMotionCommand(self, env)
