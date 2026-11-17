"""Twist-conditioned gait-library motion command.

Extends the BeyondMimic ``MotionCommand`` to a LIBRARY of periodic one-stride clips, each
labelled with a planar twist ``[vx, vy, wz]``. Each env snaps its commanded twist to the
nearest clip and plays it on a per-env ``[clip_idx, time_steps]`` clock.
The served reference is the clip UNROLLED in the world: clips are stored in their own
heading frame, and a per-env SE(2) reference frame is advanced by the clip's exact planar
step every frame and re-based onto the robot at every resample. Height, roll and pitch
are read straight from the clip, so they cannot drift.
"""

from __future__ import annotations

import glob
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.managers import CommandTerm
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import (
  quat_inv,
  quat_mul,
  sample_uniform,
  wrap_to_pi,
)

if TYPE_CHECKING:
  from collections.abc import Callable, Iterator
  from typing import Any

  import viser

  from mjlab.envs import ManagerBasedRlEnv


def _yaw_angle(q: torch.Tensor) -> torch.Tensor:
  """Yaw [rad] of wxyz quaternions, ``(..., 4) -> (...)`` (the ``yaw_quat`` angle)."""
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rot_z(yaw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate ``(..., 2)`` or ``(..., 3)`` vectors about world z by ``yaw`` (broadcast)."""
  c, s = torch.cos(yaw), torch.sin(yaw)
  xy = torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)
  return xy if v.shape[-1] == 2 else torch.cat([xy, v[..., 2:]], dim=-1)


def _yaw_mul(yaw: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
  """``qz(yaw) * q``: rotate wxyz quaternions about world z by ``yaw`` (broadcast)."""
  c, s = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
  w, x, y, z = q.unbind(-1)
  return torch.stack([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w], -1)


def _phase_remap(
  t: torch.Tensor, n_old: torch.Tensor, n_new: torch.Tensor
) -> torch.Tensor:
  """Frame ``t`` of an ``n_old``-frame cycle -> the frame of an ``n_new``-frame cycle at
  the same PHASE. Exact identity where the lengths match; otherwise the nearest frame,
  with a phase of ~1.0 wrapping to frame 0 (a cycle has no frame ``n``). ``(N,)`` long
  tensors; callers guarantee ``t < n_old``."""
  remapped = torch.round(t.float() / n_old.float() * n_new.float()).long() % n_new
  return torch.where(n_old == n_new, t, remapped)


def _fit_frame_to_clip(t: torch.Tensor, t_max: int, n: torch.Tensor) -> torch.Tensor:
  """A frame drawn on the padded ``[0, t_max)`` grid -> the same fraction of an
  ``n``-frame clip, in ``[0, n)``. Identity for a full-length clip."""
  return (t * n) // t_max


class LibraryMotionLoader:
  """Loads every tracking-format clip in a directory, stacked on a leading clip axis.

  Per-clip schema is ``tracking.mdp.commands.MotionLoader``'s plus ``twist`` ``[vx, vy,
  wz]`` and ``stride`` ``[tx, ty, dpsi]``, the SE(2) map carrying frame 0 to the closing
  frame (frame 0 one stride on). Clips are ordered by filename. The ``*_l`` arrays are
  the clips in their heading frame (see ``_build_local``)."""

  def __init__(
    self,
    motion_dir: str,
    body_indexes: torch.Tensor,
    device: str = "cpu",
    allow_ragged: bool = False,
    *,
    anchor_index: int,
  ) -> None:
    files = sorted(glob.glob(os.path.join(motion_dir, "*.npz")))
    if not files:
      raise FileNotFoundError(f"no .npz clips found in motion_dir: {motion_dir}")

    # Per-clip frame counts. A single-period library has one value and the padding below
    # is a no-op. A RAGGED library (walk + jog in one) mixes strides: shorter clips are
    # padded to the longest so the arrays still stack, and ``n_frames`` records each
    # clip's real length. It is opt-in (``allow_ragged``) so a stray wrong-length clip
    # in a single-period library fails loudly here instead of quietly changing that
    # clip's period.
    counts = [int(np.load(f)["joint_pos"].shape[0]) for f in files]
    if min(counts) < 1:
      raise ValueError(f"empty clip: {files[counts.index(min(counts))]}")
    if len(set(counts)) > 1 and not allow_ragged:
      example = {
        n: os.path.basename(files[counts.index(n)]) for n in sorted(set(counts))
      }
      raise ValueError(
        f"clips in {motion_dir} have differing frame counts {sorted(set(counts))} "
        f"(e.g. {example}); set allow_ragged=True on the command cfg to mix strides"
      )
    t_max = max(counts)

    def stack(key: str) -> torch.Tensor:
      arrs = []
      for f, n in zip(files, counts, strict=True):
        a = np.asarray(np.load(f)[key], dtype=np.float32)
        if a.shape[0] != n:
          raise ValueError(
            f"{os.path.basename(f)}: '{key}' has {a.shape[0]} frames, joint_pos has {n}"
          )
        if n < t_max:  # pad by holding the last frame; never read (see n_frames)
          a = np.concatenate([a, np.repeat(a[-1:], t_max - n, axis=0)], axis=0)
        arrs.append(a)
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
    missing = [os.path.basename(f) for f in files if "stride" not in np.load(f).files]
    if missing:
      raise KeyError(
        f"{len(missing)} clip(s) in {motion_dir} have no 'stride' (e.g. {missing[0]}); "
        "rebuild the library with `uv run python -m mjlab.scripts.build_library`"
      )
    strides = [np.asarray(np.load(f)["stride"], dtype=np.float32) for f in files]
    self.lib_strides = torch.tensor(
      np.stack(strides, axis=0).reshape(-1, 3), dtype=torch.float32, device=device
    )  # (S, 3) = [tx, ty, dpsi]
    self.num_clips = len(files)
    self.time_step_total = int(self.joint_pos.shape[1])  # T_max (padded width)
    self.n_frames = torch.tensor(
      counts, dtype=torch.long, device=device
    )  # (S,) each clip's own period in frames
    self.ragged = len(set(counts)) > 1
    hist = {n: counts.count(n) for n in sorted(set(counts))}
    print(f"[INFO] LibraryMotionLoader: {self.num_clips} clips, frames -> count {hist}")
    self._build_local(anchor_index)

  def _build_local(self, anchor_index: int) -> None:
    """Re-express every clip in its HEADING frame and store that frame's motion.

    The heading frame sits at the anchor's xy and turns uniformly through the stride's
    yaw ``dpsi``, i.e. it is the clip's nominal unicycle frame. ``body_*_l`` are the body
    arrays in it (z stays absolute), so clips of different twist can be blended at a
    shared phase. ``anchor_step_xy`` / ``_yaw`` are the frame's exact move from each
    frame to the next, expressed in the frame itself, the last one reaching the closing
    frame through ``stride``. Composing them replays the clip exactly: nothing drifts."""
    n = self.n_frames.view(-1, 1)  # (S, 1)
    frame = torch.arange(self.time_step_total, device=n.device)[None]  # (1, T)
    # float64: the steps are accumulated over a whole episode by the command.
    stride = self.lib_strides.double()
    txy, dpsi = stride[:, :2], stride[:, 2:]  # (S, 2), (S, 1)
    heading = dpsi * frame / n  # (S, T)
    axy = self.body_pos_w[:, :, anchor_index, :2].double()  # (S, T, 2)
    close_xy = _rot_z(dpsi[:, 0], axy[:, 0]) + txy  # anchor at the closing frame
    next_xy = torch.roll(axy, -1, dims=1).scatter(
      1, (n - 1)[:, :, None].expand(-1, -1, 2), close_xy[:, None]
    )
    real = frame < n  # (S, T) False on a ragged clip's padding
    self.anchor_step_xy = _rot_z(-heading, next_xy - axy) * real[..., None]
    self.anchor_step_yaw = (dpsi / n).expand_as(heading) * real
    yaw = -heading[:, :, None]  # broadcast over bodies
    centred = self.body_pos_w.double()
    centred[..., :2] -= axy[:, :, None]
    self.body_pos_l = _rot_z(yaw, centred).float()
    self.body_quat_l = _yaw_mul(yaw, self.body_quat_w.double()).float()
    self.body_lin_vel_l = _rot_z(yaw, self.body_lin_vel_w.double()).float()
    self.body_ang_vel_l = _rot_z(yaw, self.body_ang_vel_w.double()).float()


class LibraryMotionCommand(MotionCommand):
  """Per-frame reference tracking over a twist-indexed gait library (see module docstring)."""

  cfg: LibraryMotionCommandCfg
  # Narrow the loader type: the base builds a single-clip MotionLoader, this class
  # replaces it with the stacked library loader, and every accessor below reads
  # library-only attributes.
  motion: LibraryMotionLoader

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
      cfg.motion_dir,
      self.body_indexes,
      device=self.device,
      allow_ragged=cfg.allow_ragged,
      anchor_index=self.motion_anchor_body_index,
    )

    self.twist_metric_weights = torch.tensor(
      cfg.twist_metric_weights, dtype=torch.float32, device=self.device
    )  # (3,) weights the [vx, vy, wz] nearest-clip metric (units differ)

    # Index of the library's zero-twist (idle) clip, or None if it has none. Located
    # UNCONDITIONALLY, not just when rel_static_envs > 0: play mode zeroes the static fraction but
    # teleop can still command a zero twist, and the idle stop keys off this index.
    lib_twist_norm = torch.linalg.norm(
      self.motion.lib_twists * self.twist_metric_weights, dim=-1
    )
    idle_clips = torch.nonzero(lib_twist_norm < 1e-4).flatten()
    self._idle_clip_idx: int | None = (
      int(idle_clips[0]) if idle_clips.numel() > 0 else None
    )
    # A positive static fraction commands a zero twist for some envs, so the library MUST contain a
    # zero-twist (idle) clip for them to snap to (see scripts/library_to_npz.py).
    if cfg.rel_static_envs > 0.0:
      assert self._idle_clip_idx is not None, (
        "rel_static_envs > 0 requires a zero-twist idle clip in motion_dir "
        f"(e.g. crawl_fwd_vx_000.npz); found twists {self.motion.lib_twists.tolist()}"
      )

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.clip_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.twist_command = torch.zeros(self.num_envs, 3, device=self.device)
    # True where the selected reference IS the idle clip. Set by _apply_idle_stop; read by
    # mdp.motion_phase to blank the phase clock so a stopped robot stands STATICALLY.
    self.is_idle = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    # RSI (reference-state init) only at an episode reset; a timer resample keeps the
    # robot's state so it physically transitions between twists.
    self._rsi_on_resample = False
    # Serve reference velocities in the world frame instead of the robot's heading frame.
    # Only RSI and the viewer scrubber need that, to write them into the sim.
    self._world_velocities = False
    # Pinned twist for EVERY env (None -> the task's own sampler), e.g. play's --twist or the
    # viewer teleop. Changes arrive through set_fixed_twist() as a one-slot mailbox and are
    # applied in _update_command, on the sim thread.
    self._fixed_twist: tuple[float, float, float] | None = cfg.fixed_twist
    self._twist_request: tuple[bool, tuple[float, float, float] | None] = (False, None)
    # Reference frame: world SE(2) pose of the clip's heading frame (the reference
    # anchor's xy and its yaw). float64: both accumulate a step per frame all episode.
    self.ref_xy = torch.zeros(self.num_envs, 2, dtype=torch.float64, device=self.device)
    self.ref_yaw = torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)
    # Served reference tensors, memoized between changes of the command state, and the
    # depth of open _mutation() scopes (nothing is memoized inside one).
    self._cache: dict[str, torch.Tensor] = {}
    self._mutating = 0

    # Envs whose reference frame must be moved onto the robot at the next update, once
    # the sim has been forwarded past the resample.
    self._rebase_pending = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

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
    self._snap_to_library(env_ids)

  def _snap_to_library(self, env_ids: torch.Tensor) -> None:
    """clip_idx[env_ids] <- the library clip nearest to twist_command[env_ids] under the
    weighted-L2 twist metric."""
    twist = self.twist_command[env_ids]
    dist = (
      (self.motion.lib_twists[None] - twist[:, None]) ** 2 * self.twist_metric_weights
    ).sum(dim=-1)  # (n_envs, n_clips)
    old = self.clip_idx[env_ids]
    self.clip_idx[env_ids] = torch.argmin(dist, dim=-1)
    if self.motion.ragged:
      # Swapping between clips of different length must preserve PHASE, not the frame
      # number: a walk env at frame 60 of 70 is 86% through its stride, and the matching
      # jog frame is 37 of 43, not 60 (which does not exist). Remap so a mid-episode
      # twist change lands at the same point in the new gait's cycle, which is what
      # keeps the transition from teleporting.
      n_old = self.motion.n_frames[old]
      t = self.time_steps[env_ids]
      assert bool((t < n_old).all()), "time_steps ran past its clip before the swap"
      self.time_steps[env_ids] = _phase_remap(
        t, n_old, self.motion.n_frames[self.clip_idx[env_ids]]
      )

  # --- start-frame sampling on a ragged library ---------------------------------------
  # The base draws the RSI start frame on ``[0, time_step_total)``, which is the PADDED
  # width ``T_max``; on a shorter clip that lands in padding (a held last frame), the
  # robot would be teleported there and the clock would then wrap somewhere else. Fit
  # every draw onto the selected clip's own length, keeping the base's bin statistics on
  # a common PHASE scale.

  def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
    super()._uniform_sampling(env_ids)
    self._fit_start_frames(env_ids)

  def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
    # The base attributes the ending episode's failure bin as ``t / time_step_total``;
    # lift the clip-scale frame onto the padded scale first so a bin means the same
    # phase for every stride. These envs are being reset, so the lifted value is
    # overwritten by the draw.
    n = self.motion.n_frames[self.clip_idx[env_ids]]
    self.time_steps[env_ids] = (
      self.time_steps[env_ids] * self.motion.time_step_total
    ) // n
    super()._adaptive_sampling(env_ids)
    self._fit_start_frames(env_ids)

  def _fit_start_frames(self, env_ids: torch.Tensor) -> None:
    n = self.motion.n_frames[self.clip_idx[env_ids]]
    self.time_steps[env_ids] = _fit_frame_to_clip(
      self.time_steps[env_ids], self.motion.time_step_total, n
    )

  @contextmanager
  def _mutation(self) -> Iterator[None]:
    """Scope of a change to the command state (clip, phase, blend, reference frame).

    The served reference is memoized between such changes; inside one it is recomputed
    on every read, so a half-updated state is never cached."""
    self._mutating += 1
    self._cache.clear()
    try:
      yield
    finally:
      self._mutating -= 1
      self._cache.clear()

  def invalidate_reference(self) -> None:
    """Drop the memoized reference. Call after writing the command state from outside
    (e.g. ``time_steps`` or ``clip_idx`` from a script or a viewer callback)."""
    self._cache.clear()

  def _cached(self, key: str, compute: Callable[[], torch.Tensor]) -> torch.Tensor:
    """``compute()``, memoized until the command state next changes. Read-only."""
    if self._mutating:
      return compute()
    if key not in self._cache:
      self._cache[key] = compute()
    return self._cache[key]

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    # The viewer scrubber runs over the padded ``[0, T_max)``: read its frame as a
    # phase.
    n = self.motion.n_frames[self.clip_idx[env_ids]]
    with self._mutation():
      self.time_steps[env_ids] = _fit_frame_to_clip(
        torch.full_like(n, int(frame)), self.motion.time_step_total, n
      )
      self._world_velocities = True
      try:
        self._write_reference_state_to_sim(
          env_ids,
          self.body_pos_w[env_ids, 0],
          self.body_quat_w[env_ids, 0],
          self.body_lin_vel_w[env_ids, 0],
          self.body_ang_vel_w[env_ids, 0],
          self.joint_pos[env_ids],
          self.joint_vel[env_ids],
        )
      finally:
        self._world_velocities = False

  def _apply_idle_stop(self) -> None:
    """Force an exactly-zero twist wherever the reference snapped to the idle clip.

    The nearest-clip snap already has an implicit deadband -- a command closer to the zero-twist
    clip than to the slowest gait lands on the idle clip -- but the RAW commanded twist is what the
    actor observes (``mdp.commanded_twist``) and what ``twist_tracking`` rewards. So a small nonzero
    command used to ask the policy to creep while the reference held still, and the two fought.
    Collapsing the twist onto the snap makes them agree by construction:

        idle clip  <=>  zero twist  <=>  blanked phase clock

    The deadband is therefore DERIVED from the library rather than hand-tuned: it is exactly the
    Voronoi cell of the idle clip under the twist metric, so it can never disagree with the
    reference actually being tracked.

    Mirrors the velocity task, which zeroes sub-threshold commands at resample
    (``velocity_custom.mdp.velocity_command``, the ``norm(...) > 0.1`` mask) AND re-zeroes its
    standing envs on every update. Running every step is what makes this hold for subclasses that
    override the twist sampler (``DiffDriveMotionCommand``) and for live teleop between resamples.
    """
    if self._idle_clip_idx is None:
      return
    self.is_idle = self.clip_idx == self._idle_clip_idx
    self.twist_command[self.is_idle] = 0.0

  def nearest_clip_twist(self, twist: tuple[float, float, float]) -> tuple[float, ...]:
    """The library twist a command would snap to (for display; no state change)."""
    t = torch.tensor(twist, dtype=torch.float32, device=self.device)
    dist = ((self.motion.lib_twists - t) ** 2 * self.twist_metric_weights).sum(dim=-1)
    return tuple(float(v) for v in self.motion.lib_twists[int(torch.argmin(dist))])

  def set_fixed_twist(self, twist: tuple[float, float, float] | None) -> None:
    """Pin the commanded twist of EVERY env, or release the pin with None.

    Takes effect on the next control step through the timer-resample path (nearest clip, blend,
    egocentric re-base -- no RSI), and stays across later timer resamples until released. Only a
    request is stored here, so this is safe to call from a viewer or GUI thread."""
    pinned = (
      None if twist is None else (float(twist[0]), float(twist[1]), float(twist[2]))
    )
    self._twist_request = (True, pinned)

  @property
  def fixed_twist(self) -> tuple[float, float, float] | None:
    return self._fixed_twist

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    with self._mutation():
      self._resample_reference(env_ids)

  def _resample_reference(self, env_ids: torch.Tensor) -> None:
    # Pick the clip BEFORE any RSI so the teleport reads the new clip. A pin REPLACES
    # the sampler, so the clip is snapped (and its phase remapped) exactly once.
    if self._fixed_twist is None:
      self._resample_twist(env_ids)
    else:
      self.twist_command[env_ids] = torch.tensor(
        self._fixed_twist, dtype=torch.float32, device=self.device
      )
      self._snap_to_library(env_ids)
    if self._rsi_on_resample:
      # Episode reset: start at the env origin facing +x and teleport the robot onto
      # the reference there, with the reference's own world-frame velocity.
      self.ref_xy[env_ids] = self._env.scene.env_origins[env_ids, :2].double()
      self.ref_yaw[env_ids] = 0.0
      self._world_velocities = True
      try:
        super()._resample_command(env_ids)
      finally:
        self._world_velocities = False
    # Every resample re-bases the reference frame onto the robot (see _rebase). After
    # RSI that absorbs the xy / yaw pose noise, which a reference-free actor cannot see.
    self._rebase_pending[env_ids] = True

    # Whatever picked the twist above (base sampler, subclass sampler, or a teleop pin), collapse
    # near-idle commands onto the idle clip's exact zero.
    self._apply_idle_stop()

    # Every writer of time_steps above (phase remap, start-frame fit) keeps it inside
    # its clip, so padding is never read. Fail loudly here rather than ever serve a held
    # frame.
    n = self.motion.n_frames[self.clip_idx[env_ids]]
    assert bool((self.time_steps[env_ids] < n).all()), (
      "time_steps outside its clip after resample"
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
    # Episode reset: resample the twist AND RSI the robot into the selected clip's start pose.
    self._rsi_on_resample = True
    try:
      return super().reset(env_ids)
    finally:
      self._rsi_on_resample = False

  def _rebase(self, env_ids: torch.Tensor) -> None:
    """Move the reference frame onto the robot: its anchor's xy and heading.

    Forgives the path and heading error accumulated so far, so the root terms measure
    drift since the last resample. Height, roll and pitch stay the clip's. Needs the
    robot's kinematics forwarded past any teleport."""
    if env_ids.numel() == 0:
      return
    robot_q = self.robot_anchor_quat_w[env_ids]
    dyaw = _yaw_angle(quat_mul(robot_q, quat_inv(self.anchor_quat_w[env_ids])))
    self.ref_yaw[env_ids] = wrap_to_pi(self.ref_yaw[env_ids] + dyaw.double())
    self.ref_xy[env_ids] = self.robot_anchor_pos_w[env_ids, :2].double()
    self._rebase_pending[env_ids] = False

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    with self._mutation():
      # A pin request from set_fixed_twist(): apply it to every env via the timer path.
      if self._twist_request[0]:
        _, twist = self._twist_request
        self._twist_request = (False, None)
        self._fixed_twist = twist
        self._resample_command(torch.arange(self.num_envs, device=self.device))
      # Every step, not only at resample: keeps a live teleop twist and any subclass
      # sampler consistent with the snapped clip.
      self._apply_idle_stop()
      # On a reset call (env_ids given) only the reset envs advance, so a partial reset
      # does not tick other envs' clocks or reference frames.
      ids = slice(None) if env_ids is None else env_ids
      pending = self._rebase_pending.clone()
      if env_ids is not None:
        pending[:] = False
        pending[env_ids] = self._rebase_pending[env_ids]
      self._rebase(pending.nonzero().flatten())
      # Advance the reference frame by the served clip's exact planar step, then the
      # phase clock.
      step_xy, step_yaw = self._planar_step()
      self.ref_xy[ids] += _rot_z(self.ref_yaw[ids], step_xy[ids])
      self.ref_yaw[ids] = wrap_to_pi(self.ref_yaw[ids] + step_yaw[ids])
      self._advance_clock(ids)
    self.update_relative_body_poses()
    # Fold failure counts into the EMA only on the per-step update so manual resets do not decay
    # it faster (same rule as the base MotionCommand).
    if env_ids is None and self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def _advance_clock(self, ids: torch.Tensor | slice) -> None:
    """One frame on; each clip loops at its OWN length (a ragged library mixes strides)."""
    self.time_steps[ids] += 1
    self.time_steps[ids] %= self.motion.n_frames[self.clip_idx[ids]]

  # --- the served clip sample, in the clip's heading frame (subclasses blend here) ---

  def _local_pos(self) -> torch.Tensor:
    return self.motion.body_pos_l[self.clip_idx, self.time_steps]

  def _local_quat(self) -> torch.Tensor:
    return self.motion.body_quat_l[self.clip_idx, self.time_steps]

  def _local_lin_vel(self) -> torch.Tensor:
    return self.motion.body_lin_vel_l[self.clip_idx, self.time_steps]

  def _local_ang_vel(self) -> torch.Tensor:
    return self.motion.body_ang_vel_l[self.clip_idx, self.time_steps]

  def _joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.clip_idx, self.time_steps]

  def _joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.clip_idx, self.time_steps]

  def _planar_step(self) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference frame's move over the next step, in its own heading frame: xy
    ``(N, 2)`` and yaw ``(N,)``."""
    return (
      self.motion.anchor_step_xy[self.clip_idx, self.time_steps],
      self.motion.anchor_step_yaw[self.clip_idx, self.time_steps],
    )

  # --- reference accessors: the heading-frame sample placed by the reference frame ---

  @property
  def joint_pos(self) -> torch.Tensor:
    return self._cached("joint_pos", self._joint_pos)

  @property
  def joint_vel(self) -> torch.Tensor:
    return self._cached("joint_vel", self._joint_vel)

  def _world_pos(self) -> torch.Tensor:
    origin = torch.cat(
      [self.ref_xy.float(), self._env.scene.env_origins[:, 2:]], dim=-1
    )
    return _rot_z(self.ref_yaw.float()[:, None], self._local_pos()) + origin[:, None]

  def _world_quat(self) -> torch.Tensor:
    return _yaw_mul(self.ref_yaw.float()[:, None], self._local_quat())

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._cached("body_pos_w", self._world_pos)

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._cached("body_quat_w", self._world_quat)

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self.body_pos_w[:, self.motion_anchor_body_index]

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.body_quat_w[:, self.motion_anchor_body_index]

  # The names the egocentric root rewards read: the unrolled reference anchor.
  @property
  def ref_anchor_pos_w(self) -> torch.Tensor:
    return self.anchor_pos_w

  @property
  def ref_anchor_quat_w(self) -> torch.Tensor:
    return self.anchor_quat_w

  # --- velocities: served in the robot's heading frame -------------------------------------------

  def _velocity_yaw(self) -> torch.Tensor:
    """Yaw ``(N,)`` taking heading-frame velocities to the served frame: the ROBOT'S
    heading (the world reference turned by the yaw delta ``update_relative_body_poses``
    applies to the relative body targets), or the world while ``_world_velocities``."""
    if self._world_velocities:
      return self.ref_yaw.float()
    delta = quat_mul(self.robot_anchor_quat_w, quat_inv(self.anchor_quat_w))
    return self.ref_yaw.float() + _yaw_angle(delta)

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    local = self._cached("lin_vel_l", self._local_lin_vel)
    return _rot_z(self._velocity_yaw()[:, None], local)

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    local = self._cached("ang_vel_l", self._local_ang_vel)
    return _rot_z(self._velocity_yaw()[:, None], local)

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    local = self._cached("lin_vel_l", self._local_lin_vel)
    return _rot_z(self._velocity_yaw(), local[:, self.motion_anchor_body_index])

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    local = self._cached("ang_vel_l", self._local_ang_vel)
    return _rot_z(self._velocity_yaw(), local[:, self.motion_anchor_body_index])

  # --- viser GUI: the base's motion scrubber + a twist pin ---------------------------------------

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Adds a "Twist command" folder: one slider per axis spanning the library's twist extent and
    a "Pin twist" checkbox. While pinned, every env is commanded the slider twist (through
    set_fixed_twist, so applied on the sim thread); unpinning returns to the task's sampler. An
    axis the library never varies is shown as a fixed value."""

    def scrubbed() -> None:  # the base scrubber writes time_steps directly
      self.invalidate_reference()
      if on_change is not None:
        on_change()

    super().create_gui(
      name, server, get_env_idx, on_change=scrubbed, request_action=request_action
    )
    lo = self.motion.lib_twists.min(dim=0).values.tolist()
    hi = self.motion.lib_twists.max(dim=0).values.tolist()
    init = self._fixed_twist or (0.0, 0.0, 0.0)
    labels = (("vx [m/s]", 0.05), ("vy [m/s]", 0.05), ("wz [rad/s]", 0.1))
    values: list[float] = [float(v) for v in init]
    sliders = []
    with server.gui.add_folder("Twist command"):
      pin = server.gui.add_checkbox(
        "Pin twist", initial_value=self._fixed_twist is not None
      )
      for i, (label, step) in enumerate(labels):
        s_lo, s_hi = min(lo[i], 0.0), max(hi[i], 0.0)
        if s_hi - s_lo < 1e-9:  # axis not in the library: fixed, no slider
          values[i] = s_lo
          server.gui.add_number(label, initial_value=s_lo, disabled=True)
          continue
        values[i] = min(max(values[i], s_lo), s_hi)
        slider = server.gui.add_slider(
          label, min=s_lo, max=s_hi, step=step, initial_value=values[i]
        )
        slider.disabled = not pin.value
        sliders.append((i, slider))

      def _push() -> None:
        for i, s in sliders:
          values[i] = float(s.value)
        if pin.value:
          self.set_fixed_twist((values[0], values[1], values[2]))

      for _, s in sliders:
        s.on_update(lambda _: _push())

      @pin.on_update
      def _(_) -> None:
        for _, s in sliders:
          s.disabled = not pin.value
        if pin.value:
          _push()
        else:
          self.set_fixed_twist(None)


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
  allow_ragged: bool = False
  """Let the library mix clips of DIFFERENT frame counts (strides). Off by default so a
  stray wrong-length clip fails at load instead of silently changing one gait's period.
  Set it from ``LibrarySpec.ragged`` for a merged two-stride library (see
  ``walkjog_unicycle``)."""
  # Relative weights for the [vx, vy, wz] nearest-clip metric (units differ: m/s vs
  # rad/s), so a dense twist grid snaps sensibly across components.
  twist_metric_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
  # Fraction of resamples assigned a zero twist -> the static idle clip (mirrors the velocity task's
  # rel_standing_envs). 0 disables the static pose; > 0 requires a zero-twist clip in motion_dir.
  rel_static_envs: float = 0.0
  # Pin every env to ONE commanded twist instead of sampling (play's --twist, or a teleop start
  # value). None -> the task's own sampler. Can be changed live with set_fixed_twist().
  fixed_twist: tuple[float, float, float] | None = None
  motion_file: str = (
    ""  # unused (LibraryMotionLoader reads motion_dir); keep to satisfy the base
  )

  def build(self, env: ManagerBasedRlEnv) -> LibraryMotionCommand:
    return LibraryMotionCommand(self, env)
