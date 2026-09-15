"""Plot the G1-CLF ablation training curves (one PNG + PDF per Train/ scalar)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO / "logs" / "rsl_rl" / "g1_clf_tracking"
OUT_DIR = Path(__file__).resolve().parent

# IEEE single-column width in inches.
COLUMN_WIDTH = 3.5

# Arm -> (run dir suffix, legend label, categorical color in fixed slot order).
ARMS = {
  "Traj": ("Traj", "Traj only", "#2a78d6"),
  "CLF": ("CLF", "Traj + CLF", "#eb6834"),
  "Qdes": ("Qdes", "Traj + Qdes", "#1baf7a"),
  "All": ("All", "Traj + CLF + Qdes", "#eda100"),
}

# Tag -> (file stem, y label, log scale, end labels, legend location).
METRICS = {
  "Train/mean_episode_length": (
    "mean_episode_length",
    "Mean episode length [steps]",
    False,
    False,
    "lower right",
  ),
  "Train/r_mimic_pos": (
    "r_mimic_pos",
    "Mimic position reward",
    False,
    True,
    "lower right",
  ),
  "Train/clf_V": ("clf_V", r"CLF value $V = e^\top P e$", True, True, "upper right"),
}

IEEE_RC = {
  "text.usetex": True,
  "text.latex.preamble": r"\usepackage{mathptmx}",
  "font.family": "serif",
  "font.size": 8,
  "axes.labelsize": 8,
  "axes.linewidth": 0.5,
  "xtick.labelsize": 7,
  "ytick.labelsize": 7,
  "xtick.direction": "in",
  "ytick.direction": "in",
  "xtick.top": True,
  "ytick.right": True,
  "xtick.major.width": 0.5,
  "ytick.major.width": 0.5,
  "xtick.minor.width": 0.4,
  "ytick.minor.width": 0.4,
  "xtick.major.size": 3,
  "ytick.major.size": 3,
  "xtick.minor.size": 1.5,
  "ytick.minor.size": 1.5,
  "legend.fontsize": 6.5,
  "legend.frameon": True,
  "legend.fancybox": False,
  "legend.edgecolor": "black",
  "legend.framealpha": 1.0,
  "pdf.fonttype": 42,
  "savefig.bbox": "tight",
  "savefig.pad_inches": 0.02,
}


def load(run: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
  files = sorted((LOG_ROOT / f"clf_ablation_{run}_0913").glob("events.out.tfevents.*"))
  if not files:
    raise FileNotFoundError(f"no event file for arm {run} under {LOG_ROOT}")
  acc = EventAccumulator(str(files[-1]), size_guidance={"scalars": 0})
  acc.Reload()
  events = acc.Scalars(tag)
  return np.array([e.step for e in events]), np.array([e.value for e in events])


def ema(y: np.ndarray, alpha: float) -> np.ndarray:
  out = np.empty_like(y, dtype=float)
  acc = y[0]
  for i, v in enumerate(y):
    acc = alpha * v + (1 - alpha) * acc
    out[i] = acc
  return out


def spread_labels(ys: list[float], min_gap: float) -> list[float]:
  """Nudge end-label positions apart so they never overlap, preserving order."""
  order = np.argsort(ys)
  placed = np.array(ys, dtype=float)[order]
  for i in range(1, len(placed)):
    placed[i] = max(placed[i], placed[i - 1] + min_gap)
  shift = (placed - np.array(ys)[order]).mean()
  placed -= shift
  out = np.empty_like(placed)
  out[order] = placed
  return list(out)


def plot_metric(tag: str, alpha: float) -> None:
  stem, ylabel, log, end_labels, legend_loc = METRICS[tag]
  plt.rcParams.update(IEEE_RC)
  fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.3))

  ends = []
  for run, label, color in ARMS.values():
    x, y = load(run, tag)
    smooth = ema(y, alpha)
    ax.plot(x, y, color=color, lw=0.4, alpha=0.15, zorder=1, rasterized=True)
    ax.plot(x, smooth, color=color, lw=1.0, label=label, zorder=3)
    ends.append((x[-1], smooth[-1], label, color))

  if log:
    ax.set_yscale("log")
  x_max = round(max(e[0] for e in ends), -3)
  ax.set_xlim(0, x_max * (1.45 if end_labels else 1.0))
  ax.set_xticks(np.arange(0, x_max + 1, 1000))
  ax.minorticks_on()
  ax.set_xlabel("Training iteration")
  ax.set_ylabel(ylabel)
  ax.grid(True, which="major", color="0.85", lw=0.3, ls="--", zorder=0)
  leg = ax.legend(loc=legend_loc, handlelength=1.5, borderpad=0.4, labelspacing=0.3)
  leg.get_frame().set_linewidth(0.5)
  fig.tight_layout(pad=0.2)

  if end_labels:
    fig.canvas.draw()
    # Direct end labels, spread in points after layout so close curves never overlap.
    px_per_pt = fig.dpi / 72.0
    disp_y = [ax.transData.transform((e[0], e[1]))[1] / px_per_pt for e in ends]
    lab_y = spread_labels(disp_y, min_gap=7.5)
    for (x_end, y_end, label, color), y0, y1 in zip(ends, disp_y, lab_y, strict=True):
      ax.plot(
        [x_end], [y_end], "o", ms=2.5, color=color, mec="white", mew=0.5, zorder=4
      )
      value = f"{y_end:,.0f}" if log else f"{y_end:.2f}"
      ax.annotate(
        f"{label}  {value}",
        xy=(x_end, y_end),
        xytext=(4, y1 - y0),
        textcoords="offset points",
        va="center",
        fontsize=6,
      )

  for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"clf_ablation_{stem}.{ext}", dpi=300)
  plt.close(fig)
  print(f"wrote {OUT_DIR}/clf_ablation_{stem}.{{png,pdf}}")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--ema", type=float, default=0.02, help="EMA weight on new points")
  a = ap.parse_args()
  for tag in METRICS:
    plot_metric(tag, a.ema)


if __name__ == "__main__":
  main()
