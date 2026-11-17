"""Gait-library reference: seamless conversion and exact, drift-free unrolling."""

from pathlib import Path

import numpy as np
import pytest
import torch

import mjlab.tasks  # noqa: F401  Populate the task registry.
from mjlab.scripts.library_to_npz import (
  LibraryMotionLoader as SourceLoader,
)
from mjlab.scripts.library_to_npz import (
  _rebased_stride,
  _rot_z,
  _stride_map,
)
from mjlab.tasks.crawling_fwd.mdp.commands import LibraryMotionLoader
from mjlab.tasks.crawling_fwd_blending.mdp.commands import BlendingMotionCommand
from mjlab.tasks.registry import load_env_cfg

VX, WZ = 1.0, 0.8  # twist of the synthetic arc clips [m/s, rad/s]


def _arc(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Base xy and yaw of a unicycle driving ``[VX, 0, WZ]`` from the origin."""
  yaw = WZ * t
  xy = VX / WZ * np.stack([np.sin(yaw), 1.0 - np.cos(yaw)], axis=-1)
  return xy, yaw


def _yaw_quat(yaw: np.ndarray) -> np.ndarray:
  zero = np.zeros_like(yaw)
  return np.stack([np.cos(yaw / 2), zero, zero, np.sin(yaw / 2)], axis=-1)


def test_converter_resamples_one_period_with_seamless_velocities(
  tmp_path: Path,
) -> None:
  # A 0.85 s stride at 100 Hz: 42.5 output frames, so it is stretched onto 43.
  period, frames = 0.85, 86
  t = np.arange(frames) * 0.01
  state = np.zeros((frames, 71))
  state[:, 0:2], yaw = _arc(t)
  state[:, 2] = 0.8
  state[:, 3:7] = _yaw_quat(yaw)
  state[:, 7] = np.sin(2 * np.pi * t / period)  # one periodic joint
  np.savez(tmp_path / "clip.npz", state=state)
  loader = SourceLoader(str(tmp_path / "clip.npz"), 100, 50, "cpu")

  assert loader.output_frames == 43
  slow = period / (43 * 0.02)  # played 1.2 % slower than the source
  speed = loader.motion_base_lin_vels.norm(dim=-1)
  # The seam frames (first, last) carry the same velocity as every other frame.
  torch.testing.assert_close(
    speed, torch.full_like(speed, VX * slow), atol=2e-3, rtol=0.0
  )
  wz = loader.motion_base_ang_vels[:, 2]
  torch.testing.assert_close(wz, torch.full_like(wz, WZ * slow), atol=2e-3, rtol=0.0)
  phase = torch.arange(43) / 43
  joint_vel = 2 * torch.pi / period * slow * torch.cos(2 * torch.pi * phase)
  torch.testing.assert_close(
    loader.motion_dof_vels[:, 0], joint_vel.float(), atol=5e-2, rtol=0.0
  )
  assert loader.stride_dpsi == pytest.approx(WZ * period)


def test_stride_map_survives_a_rebase() -> None:
  rng = np.random.default_rng(0)
  b0, dxy, dpsi = rng.normal(size=2), rng.normal(size=2), 0.7
  g = _stride_map(b0, dxy, dpsi)
  np.testing.assert_allclose(_rot_z(dpsi)[:2, :2] @ b0 + g[:2], b0 + dxy)
  yaw, t = -0.4, rng.normal(size=2)
  moved = _rebased_stride(g, yaw, t)
  p = rng.normal(size=2)  # a point of the original clip, then of the moved one
  RA, Rg = _rot_z(yaw)[:2, :2], _rot_z(dpsi)[:2, :2]
  np.testing.assert_allclose(
    Rg @ (RA @ p + t) + moved[:2], RA @ (Rg @ p + g[:2]) + t, atol=1e-12
  )


def _write_arc_clip(d: Path, frames: int = 20) -> np.ndarray:
  """A tracking clip of two bodies riding the arc; returns its ``stride``."""
  dt = 0.02
  xy, yaw = _arc(np.arange(frames) * dt)
  offsets = np.array([[0.0, 0.0, 0.9], [0.3, 0.1, 0.5]])  # anchor, a limb
  pos = np.zeros((frames, 2, 3))
  for b, off in enumerate(offsets):
    for k in range(frames):
      pos[k, b] = _rot_z(yaw[k]) @ off + np.array([*xy[k], 0.0])
  quat = np.repeat(_yaw_quat(yaw)[:, None], 2, axis=1)
  close_xy, close_yaw = _arc(np.array([frames * dt]))
  stride = np.array([*close_xy[0], close_yaw[0]])  # frame 0 sits at the origin
  np.savez(
    d / "arc.npz",
    fps=np.array([50.0]),
    twist=np.array([VX, 0.0, WZ], dtype=np.float32),
    stride=stride,
    joint_pos=np.zeros((frames, 1), dtype=np.float32),
    joint_vel=np.zeros((frames, 1), dtype=np.float32),
    body_pos_w=pos.astype(np.float32),
    body_quat_w=quat.astype(np.float32),
    body_lin_vel_w=np.zeros((frames, 2, 3), dtype=np.float32),
    body_ang_vel_w=np.zeros((frames, 2, 3), dtype=np.float32),
  )
  return stride


def test_composed_steps_replay_the_clip_across_strides(tmp_path: Path) -> None:
  frames = 20
  _write_arc_clip(tmp_path, frames)
  lib = LibraryMotionLoader(str(tmp_path), torch.arange(2), "cpu", anchor_index=0)
  xy, yaw = torch.zeros(2, dtype=torch.float64), torch.zeros((), dtype=torch.float64)
  for step in range(5 * frames):  # five strides, 4.6 rad of turning
    k = step % frames
    c, s = torch.cos(yaw), torch.sin(yaw)
    limb = lib.body_pos_l[0, k, 1].double()
    served = xy + torch.stack([c * limb[0] - s * limb[1], s * limb[0] + c * limb[1]])
    arc_xy, arc_yaw = _arc(np.array([step * 0.02]))
    truth = _rot_z(arc_yaw[0])[:2, :2] @ np.array([0.3, 0.1]) + arc_xy[0]
    np.testing.assert_allclose(served.numpy(), truth, atol=2e-6)
    dxy = lib.anchor_step_xy[0, k]
    xy = xy + torch.stack([c * dxy[0] - s * dxy[1], s * dxy[0] + c * dxy[1]])
    yaw = yaw + lib.anchor_step_yaw[0, k]


def test_loader_requires_the_stride(tmp_path: Path) -> None:
  _write_arc_clip(tmp_path)
  clip = dict(np.load(tmp_path / "arc.npz"))
  del clip["stride"]
  np.savez(tmp_path / "arc.npz", **clip)
  with pytest.raises(KeyError, match="rebuild the library"):
    LibraryMotionLoader(str(tmp_path), torch.arange(2), "cpu", anchor_index=0)


@pytest.mark.slow
def test_walk_reference_keeps_clip_tilt_and_rebases_onto_robot() -> None:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.utils.lab_api.math import quat_apply_inverse

  cfg = load_env_cfg("G1-Walk-Unicycle")
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  cmd = env.command_manager.get_term("motion")
  assert isinstance(cmd, BlendingMotionCommand)
  env.reset()
  # RSI adds xy / yaw noise; the reset re-base puts the reference back on the robot,
  # one reference step ahead.
  gap = (cmd.anchor_pos_w - cmd.robot_anchor_pos_w)[:, :2].norm(dim=-1)
  assert float(gap.max()) < 0.04
  anchor = cmd.motion_anchor_body_index
  up = torch.tensor([0.0, 0.0, 1.0]).expand(4, 3)
  for _ in range(150):  # two strides without physics: the reference alone
    cmd._update_command(None)
    steady = cmd.blend_alpha >= 1.0
    clip_quat = cmd.motion.body_quat_w[cmd.clip_idx, cmd.time_steps, anchor]
    clip_z = cmd.motion.body_pos_w[cmd.clip_idx, cmd.time_steps, anchor, 2]
    tilt = quat_apply_inverse(cmd.anchor_quat_w, up) - quat_apply_inverse(clip_quat, up)
    assert float(tilt[steady].abs().max()) < 1e-5
    assert float((cmd.anchor_pos_w[:, 2] - clip_z)[steady].abs().max()) < 1e-5
  # Blending two opposite pivots late in the stride keeps the body layout rigid.
  twists = cmd.motion.lib_twists
  left = int((twists - torch.tensor([0.0, 0.0, 1.5])).norm(dim=-1).argmin())
  right = int((twists - torch.tensor([0.0, 0.0, -1.5])).norm(dim=-1).argmin())
  cmd.clip_idx_prev[:], cmd.clip_idx[:], cmd.time_steps[:] = right, left, 60
  names = cmd.cfg.body_names
  wrists = [names.index(f"{side}_wrist_yaw_link") for side in ("left", "right")]
  span = []
  for alpha in (0.0, 0.5, 1.0):
    cmd.blend_alpha[:] = alpha
    cmd.invalidate_reference()  # the state was written from outside the command
    pos = cmd.body_pos_w[0]
    span.append(float((pos[wrists[0]] - pos[wrists[1]]).norm()))
  assert min(span[0], span[2]) - 0.01 < span[1] < max(span[0], span[2]) + 0.01
  env.close()
