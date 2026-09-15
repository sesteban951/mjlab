"""Plot the mimic-style tracking errors of every arm of a sweep over training.

Five curves per arm, in native units: base pose position and rotation, base twist
linear and angular, and joint position -- the motion command's anchor and joint
metrics, which mean the same thing in every arm. Smoothed with a running mean.

  uv run python scripts/tools/plot_ablation.py --prefix rs1 --out rs1.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# (tag, label, unit)
CURVES = [
  ("Metrics/motion/error_anchor_pos", "pose position", "m"),
  ("Metrics/motion/error_anchor_rot", "pose rotation", "rad"),
  ("Metrics/motion/error_anchor_lin_vel", "twist linear", "m/s"),
  ("Metrics/motion/error_anchor_ang_vel", "twist angular", "rad/s"),
  ("Metrics/motion/error_joint_pos", "joint position", "rad"),
]
# Train/r_* groups the env cfg logs; plotted as a second row when --rewards is set.
REWARD_CURVES = [
  ("Train/r_mimic_pos", "r_mimic_pos"),
  ("Train/r_mimic_vel", "r_mimic_vel"),
  ("Train/r_clf", "r_clf"),
  ("Train/r_uff", "r_uff"),
  ("Train/r_ufb", "r_ufb"),
  ("Train/r_u", "r_u"),
  ("Train/r_regularization", "r_regularization"),
]
ARM_ORDER = ("control", "clf_decrease", "qdes_imitation", "lqr_clf", "action_prior_exp")


def _pattern(prefix: str) -> str:
  return (
    rf"^(?:\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}}_)?{re.escape(prefix)}-"
    r"(?P<arm>[a-z_]+?)(?P<tag>-[a-z0-9]+?)?(?:-\d{8}-\d{6})?$"
  )


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
  if window <= 1 or y.size < window:
    return y
  kernel = np.ones(window) / window
  return np.convolve(y, kernel, mode="valid")


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--log-dir", default="logs/rsl_rl/g1_tracking_prior")
  ap.add_argument("--prefix", default="ablate")
  ap.add_argument("--window", type=int, default=50, help="running-mean window")
  ap.add_argument("--out", default=None, help="PNG path; default <prefix>_tracking.png")
  ap.add_argument(
    "--rewards", action="store_true", help="add a row of Train/r_* curves"
  )
  a = ap.parse_args()

  runs: dict[str, Path] = {}
  for d in sorted((REPO / a.log_dir).iterdir()):
    m = re.search(_pattern(a.prefix), d.name)
    if d.is_dir() and m:
      runs[m.group("arm") + (m.group("tag") or "")] = d
  if not runs:
    ap.error(f"no runs with prefix {a.prefix!r} under {a.log_dir}")
  order = [x for x in ARM_ORDER if x in runs] + [k for k in runs if k not in ARM_ORDER]

  rows = [CURVES] + (
    [[(tag, name, "") for tag, name in REWARD_CURVES]] if a.rewards else []
  )
  ncol = max(len(r) for r in rows)
  fig, grid = plt.subplots(
    len(rows), ncol, figsize=(4.2 * ncol, 3.6 * len(rows)), squeeze=False
  )
  for arm in order:
    ea = EventAccumulator(str(runs[arm]), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    for row, curves in zip(grid, rows, strict=True):
      for ax, (tag, label, unit) in zip(row, curves, strict=False):
        if tag not in tags:
          continue
        ev = ea.Scalars(tag)
        x = np.array([e.step for e in ev], dtype=float)
        y = _smooth(np.array([e.value for e in ev], dtype=float), a.window)
        ax.plot(x[-y.size :], y, label=arm, linewidth=1.2)
        ax.set_title(f"{label} error [{unit}]" if unit else label)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
  for row, curves in zip(grid, rows, strict=True):
    for ax in row[len(curves) :]:
      ax.axis("off")
  grid[0][0].legend(fontsize=8)
  fig.suptitle(f"{a.prefix}: tracking error over training (window {a.window})")
  fig.tight_layout()
  out = a.out or f"{a.prefix}_tracking.png"
  fig.savefig(out, dpi=130)
  print(f"wrote {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
