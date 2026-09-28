"""Ragged (mixed-stride) gait libraries: loader, phase remap, RSI fit, merge build."""

from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.scripts.build_library import _merge_keep, _verify
from mjlab.tasks.crawling_common.library import LibrarySpec, MergeSource, Source
from mjlab.tasks.crawling_fwd.mdp.commands import (
  LibraryMotionLoader,
  _fit_frame_to_clip,
  _phase_remap,
)

NJ, NB = 3, 2  # joints, bodies in the synthetic clips


def _write_clip(d: Path, name: str, n: int, twist, body_frames: int | None = None):
  """A tracking-format npz with ``n`` frames; ``body_pos_w`` may disagree, for tests."""
  rng = np.random.default_rng(hash(name) % 2**32)
  nb = body_frames if body_frames is not None else n
  quat = rng.standard_normal((nb, NB, 4)).astype(np.float32)
  quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
  np.savez(
    d / name,
    fps=np.array([50.0]),
    twist=np.asarray(twist, dtype=np.float32),
    joint_pos=rng.standard_normal((n, NJ)).astype(np.float32),
    joint_vel=rng.standard_normal((n, NJ)).astype(np.float32),
    body_pos_w=rng.standard_normal((nb, NB, 3)).astype(np.float32),
    body_quat_w=quat,
    body_lin_vel_w=rng.standard_normal((nb, NB, 3)).astype(np.float32),
    body_ang_vel_w=rng.standard_normal((nb, NB, 3)).astype(np.float32),
  )


@pytest.fixture
def ragged_dir(tmp_path: Path) -> Path:
  _write_clip(tmp_path, "walk.npz", 70, (0.75, 0.0, 0.0))
  _write_clip(tmp_path, "jog.npz", 43, (1.3, 0.0, 0.0))
  return tmp_path


def _load(d: Path, allow_ragged: bool) -> LibraryMotionLoader:
  return LibraryMotionLoader(str(d), torch.arange(NB), "cpu", allow_ragged=allow_ragged)


def test_loader_rejects_mixed_lengths_by_default(ragged_dir: Path) -> None:
  with pytest.raises(ValueError, match="differing frame counts"):
    _load(ragged_dir, allow_ragged=False)


def test_loader_pads_ragged_when_allowed(ragged_dir: Path) -> None:
  lib = _load(ragged_dir, allow_ragged=True)
  assert lib.ragged and lib.time_step_total == 70
  assert lib.joint_pos.shape == (2, 70, NJ)
  assert lib.body_pos_w.shape == (2, 70, NB, 3)
  # Files load in name order: jog (43) then walk (70).
  assert lib.n_frames.tolist() == [43, 70]
  jog = np.load(ragged_dir / "jog.npz")
  # Padding holds the clip's last real frame.
  last = torch.tensor(jog["joint_pos"][42:43]).expand(28, -1)
  torch.testing.assert_close(lib.joint_pos[0, 42:], last)
  torch.testing.assert_close(lib.joint_pos[0, :43], torch.tensor(jog["joint_pos"]))


def test_loader_uniform_library_is_not_ragged(tmp_path: Path) -> None:
  _write_clip(tmp_path, "a.npz", 70, (0.5, 0.0, 0.0))
  _write_clip(tmp_path, "b.npz", 70, (0.6, 0.0, 0.0))
  lib = _load(tmp_path, allow_ragged=False)
  assert not lib.ragged and lib.n_frames.tolist() == [70, 70]


def test_loader_rejects_per_key_length_mismatch(tmp_path: Path) -> None:
  _write_clip(tmp_path, "bad.npz", 43, (1.0, 0.0, 0.0), body_frames=42)
  with pytest.raises(ValueError, match="'body_pos_w' has 42 frames, joint_pos has 43"):
    _load(tmp_path, allow_ragged=True)


def test_phase_remap_identity_when_lengths_match() -> None:
  t = torch.arange(70)
  n = torch.full_like(t, 70)
  assert torch.equal(_phase_remap(t, n, n), t)


def test_phase_remap_edges_and_wrap() -> None:
  def remap(t: int, n_old: int, n_new: int) -> int:
    return int(
      _phase_remap(torch.tensor([t]), torch.tensor([n_old]), torch.tensor([n_new]))
    )

  assert remap(0, 70, 43) == 0
  assert remap(69, 70, 43) == 42  # round(69 / 70 * 43) = 42.4 -> 42, the last frame
  assert remap(42, 43, 70) == 68  # round(42 / 43 * 70) = 68.4
  assert remap(69, 70, 34) == 0  # round(33.5) = 34 == n_new: phase ~1 wraps to frame 0


def test_phase_remap_round_trip_within_one_frame() -> None:
  t = torch.arange(70)
  n70, n43 = torch.full_like(t, 70), torch.full_like(t, 43)
  back = _phase_remap(_phase_remap(t, n70, n43), n43, n70)
  err = torch.minimum((back - t).abs(), 70 - (back - t).abs())  # circular distance
  assert int(err.max()) <= 1


def test_fit_frame_to_clip_stays_in_range() -> None:
  t = torch.arange(70)
  fitted = _fit_frame_to_clip(t, 70, torch.full_like(t, 43))
  assert int(fitted.min()) == 0 and int(fitted.max()) == 42
  assert torch.equal(_fit_frame_to_clip(t, 70, torch.full_like(t, 70)), t)


def test_merge_keep_half_open_boundary_is_exclusive() -> None:
  walk = MergeSource("w", fwd_vx=(0.5, 0.95), pivot_wz=(0.0, 1.4945), idle=True)
  jog = MergeSource("j", fwd_vx=(0.95, 1.6), pivot_wz=(1.4945, 2.1))
  for twist in ([0.95, 0, 0], [0.9, 0, 0], [1.0, 0, 0], [0, 0, 1.4945], [0, 0, -1.45]):
    kept = [_merge_keep(np.array(twist), m) for m in (walk, jog)]
    assert sum(kept) == 1, f"{twist} kept by {sum(kept)} sources"
  assert _merge_keep(
    np.array([0.95, 0, 0]), jog
  )  # the shared edge goes to the upper band
  assert _merge_keep(np.array([0, 0, -1.45]), walk)  # pivots band on |wz|


def test_merge_keep_idle_tolerance_and_lateral_rejected() -> None:
  src = MergeSource("w", fwd_vx=(0.5, 1.0), idle=True)
  assert _merge_keep(
    np.array([1e-12, 0.0, 0.0]), src
  )  # an idle stored as ~0, not exactly 0
  assert not _merge_keep(np.array([0.0, 0.0, 0.0]), MergeSource("j", fwd_vx=(0.5, 1.0)))
  with pytest.raises(ValueError, match="lateral"):
    _merge_keep(np.array([0.0, 0.1, 0.0]), src)


def _merge_spec(name: str, ragged: bool) -> LibrarySpec:
  return LibrarySpec(
    name=name, merges=(MergeSource("x", fwd_vx=(0.0, 9.0), idle=True),), ragged=ragged
  )


def test_verify_reports_failures_and_passes_a_good_ragged_dir(tmp_path: Path) -> None:
  _write_clip(tmp_path, "idle.npz", 70, (0.0, 0.0, 0.0))
  _write_clip(tmp_path, "walk.npz", 70, (0.75, 0.0, 0.0))
  _write_clip(tmp_path, "jog.npz", 43, (1.3, 0.0, 0.0))
  assert _verify(_merge_spec("ok", ragged=True), tmp_path) == []
  reasons = _verify(_merge_spec("strict", ragged=False), tmp_path)
  assert any("frame counts differ" in r for r in reasons)
  _write_clip(tmp_path, "dupe.npz", 43, (1.3, 0.0, 0.0))
  assert any(
    "duplicate" in r for r in _verify(_merge_spec("ok", ragged=True), tmp_path)
  )


def test_library_spec_rejects_bad_combinations() -> None:
  with pytest.raises(ValueError, match="exactly one of sources or merges"):
    LibrarySpec(name="neither")
  with pytest.raises(ValueError, match="exactly one of sources or merges"):
    LibrarySpec(
      name="both", sources=(Source("f"),), merges=(MergeSource("x", idle=True),)
    )
  with pytest.raises(ValueError, match="staging fields it ignores"):
    LibrarySpec(name="inert", merges=(MergeSource("x", idle=True),), idle_name="x.npz")
  with pytest.raises(ValueError, match="exactly one MergeSource must set idle"):
    LibrarySpec(
      name="two_idle", merges=(MergeSource("x", idle=True), MergeSource("y", idle=True))
    )
  _merge_spec("fine", ragged=True)  # a valid merge spec constructs
