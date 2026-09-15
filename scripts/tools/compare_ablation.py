"""Compare ablation arms on the metrics that mean the same thing in every arm.

``Train/mean_reward`` is NOT one of them: each arm optimizes a different reward, so a
higher total says only that its own objective is easier to score, not that it tracks the
clip better. What is comparable is the task's own tracking error, the episode length and
the termination mix -- none of which change definition between arms.

Values are averaged over the last ``--window`` iterations up to ``--at`` to damp the
per-iteration noise.

  uv run python scripts/tools/compare_ablation.py --prefix rs1 --at 3500
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = Path(__file__).resolve().parents[2]

# (tag, label, lower_is_better)
COMPARABLE = [
  ("Train/mean_episode_length", "episode length", False),
  ("Metrics/motion/error_anchor_pos", "pose pos err", True),
  ("Metrics/motion/error_anchor_rot", "pose rot err", True),
  ("Metrics/motion/error_anchor_lin_vel", "twist lin err", True),
  ("Metrics/motion/error_anchor_ang_vel", "twist ang err", True),
  ("Metrics/motion/error_joint_pos", "joint pos err", True),
  ("Metrics/motion/error_body_pos", "body pos err", True),
  ("Metrics/motion/error_body_rot", "body rot err", True),
  ("Metrics/motion/error_body_lin_vel", "body linvel err", True),
  ("Metrics/motion/error_body_ang_vel", "body angvel err", True),
  ("Episode_Termination/anchor_pos", "term anchor_pos", True),
  ("Episode_Termination/anchor_ori", "term anchor_ori", True),
  ("Episode_Termination/ee_body_pos", "term ee_body_pos", True),
]


def _last_step(run_dir: Path) -> int:
  """Highest iteration this run logged, 0 if it logged nothing."""
  ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
  ea.Reload()
  if "Train/mean_reward" not in set(ea.Tags()["scalars"]):
    return 0
  return max((e.step for e in ea.Scalars("Train/mean_reward")), default=0)


def load(run_dir: Path, at: int, window: int) -> dict[str, float]:
  ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
  ea.Reload()
  tags = set(ea.Tags()["scalars"])
  out: dict[str, float] = {}
  for tag, _, _ in COMPARABLE:
    if tag not in tags:
      continue
    events = ea.Scalars(tag)
    vals = [e.value for e in events if at - window < e.step <= at]
    if vals:
      out[tag] = float(np.mean(vals))
  steps = [e.step for e in ea.Scalars("Train/mean_reward")] if tags else []
  out["_last_step"] = float(max(steps)) if steps else 0.0
  return out


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--log-dir", default="logs/rsl_rl/g1_tracking_prior")
  ap.add_argument(
    "--prefix",
    default="ablate",
    help="run-name prefix of the sweep, e.g. rs1; matches old timestamped names too",
  )
  ap.add_argument(
    "--pattern",
    default=None,
    help="explicit regex with an <arm> group, overriding --prefix",
  )
  ap.add_argument("--at", type=int, default=3500)
  ap.add_argument("--window", type=int, default=100)
  a = ap.parse_args()
  if a.pattern is None:
    a.pattern = (
      rf"^(?:\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}}_)?{re.escape(a.prefix)}-"
      r"(?P<arm>[a-z_]+?)(?P<tag>-[a-z0-9]+?)?(?:-\d{8}-\d{6})?$"
    )

  # An arm can have several run dirs (a relaunch, an aborted attempt). Keep the one
  # that actually trained longest, not the newest -- a stub would silently win.
  runs: dict[str, Path] = {}
  for d in sorted((REPO / a.log_dir).iterdir()):
    m = re.search(a.pattern, d.name)
    if not (d.is_dir() and m):
      continue
    # A tag (e.g. -k60) makes a second setting of an arm its own column, not a rival
    # for the same slot, so two kappas can be compared side by side.
    arm = m.group("arm") + (m.groupdict().get("tag") or "")
    if arm not in runs or _last_step(d) > _last_step(runs[arm]):
      runs[arm] = d
  if not runs:
    ap.error(f"no runs matching {a.pattern!r} under {a.log_dir}")

  order = [
    x
    for x in (
      "control",
      "action_prior_exp",
      "clf_decrease",
      "qdes_imitation",
    )
    if x in runs
  ]
  order += [k for k in runs if k not in order]
  data = {arm: load(runs[arm], a.at, a.window) for arm in order}

  print(f"\nmean over iterations ({a.at - a.window}, {a.at}]\n")
  for arm in order:
    print(f"  {arm:24s} {runs[arm].name}  (logged to {int(data[arm]['_last_step'])})")

  width = 22
  header = f"\n{'metric':<18}" + "".join(f"{arm[:width]:>{width}}" for arm in order)
  print(header)
  print("-" * len(header))
  base = order[0]
  for tag, label, lower_better in COMPARABLE:
    if not any(tag in data[arm] for arm in order):
      continue
    row = f"{label:<18}"
    for arm in order:
      v = data[arm].get(tag)
      if v is None:
        row += f"{'-':>{width}}"
        continue
      cell = f"{v:.4g}"
      if arm != base and base in data and tag in data[base]:
        b = data[base][tag]
        if abs(b) > 1e-12:
          delta = 100.0 * (v - b) / abs(b)
          mark = "+" if (delta < 0) == lower_better else "-"
          cell += f" ({delta:+.0f}% {mark})"
      row += f"{cell:>{width}}"
    print(row)
  print(f"\nvs {base}; (+) is better on that metric, (-) worse.")
  print("mean_reward is deliberately absent: the arms do not share a reward function.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
