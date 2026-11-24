"""Tests for the G1 jog + run + sprint task and its cross-library phase alignment."""

import glob

import numpy as np
import pytest

from mjlab.scripts.library_to_npz import _circular_lag, _resample_periodic
from mjlab.tasks.gait_library.library import LIBRARY_SPECS
from mjlab.tasks.jog_unicycle.mdp.commands import UnicycleMotionCommandCfg
from mjlab.tasks.registry import load_env_cfg

_LEFT_ANKLE, _RIGHT_ANKLE = 6, 12  # body indices in the G1 tracking npz


def _profile(n: int, shift: float) -> np.ndarray:
  phase = 2 * np.pi * (np.arange(n) / n + shift)
  return np.stack([np.sin(phase), np.sin(phase + np.pi)])


def test_external_reference_aligns_clips_of_another_length():
  ref = _profile(43, 0.0)
  clip = _profile(32, 0.25)  # a quarter stride ahead, on a shorter clip
  s = _circular_lag(clip, _resample_periodic(ref, 32))
  assert s == 8  # delaying it a quarter of 32 frames lines it up


def test_task_uses_merged_library():
  cmd = load_env_cfg("G1-JogRunSprint-Unicycle").commands["motion"]
  assert isinstance(cmd, UnicycleMotionCommandCfg)
  assert cmd.motion_dir == str(LIBRARY_SPECS["jogrunsprint"].tracking_dir)
  assert cmd.vx_fwd_range == (0.5, 3.5)


@pytest.mark.parametrize(
  "library, families",
  [
    ("jogrunsprint", ("jog_fwd", "run_fwd", "sprint_fwd")),
    ("walkjog_unicycle", ("jog_fwd", "walk_fwd")),
  ],
)
def test_families_share_one_foot_phase(library, families):
  d = LIBRARY_SPECS[library].tracking_dir
  peaks = {}
  for fam in families:
    f = glob.glob(str(d / f"{fam}_vx*_wz+000.npz"))[0]
    z = np.load(f)["body_pos_w"][:, [_LEFT_ANKLE, _RIGHT_ANKLE], 2]
    peaks[fam] = np.argmax(z, axis=0) / len(z)
  for fam, p in peaks.items():
    gap = np.abs((p - peaks["jog_fwd"] + 0.5) % 1.0 - 0.5)
    assert np.all(gap < 0.08), (fam, p, peaks["jog_fwd"])
