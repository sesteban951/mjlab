"""Run the prior-term ablation: one arm per training run, one term on at a time.

Arms are ``control`` (every prior term off) plus one arm per term in
``ABLATION_TERMS``. Every arm is otherwise identical -- same clip, same tape, same
seed, same iteration count -- so a difference between two runs is attributable to the
single reward term that separates them.

The Lyapunov constants are motion-specific (V is a property of the policy, not of P),
so the intended order is: run ``control`` first, measure V off its checkpoint with
``mjlab.scripts.measure_lyapunov``, put the numbers in ``--kappa`` / ``--v-half``, then
run the remaining arms.

  # 1. control arm
  uv run python scripts/tools/ablate_prior.py --arms control

  # 2. retune from its checkpoint
  uv run python -m mjlab.scripts.measure_lyapunov \
    --checkpoint logs/rsl_rl/g1_tracking_prior/<run>/model_3999.pt \
    --motion-file <motion.npz> --tape-file <prior.npz> --out tuning.json

  # 3. the rest, with the measured constants
  uv run python scripts/tools/ablate_prior.py --kappa K --v-half V \
    --arms action_prior_exp clf_decrease qdes_imitation
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARMS = ("control", "action_prior_exp", "clf_decrease", "qdes_imitation", "lqr_clf")


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
  ap.add_argument("--task", required=True)
  ap.add_argument("--motion-file", required=True)
  ap.add_argument("--tape-file", required=True)
  ap.add_argument("--iterations", type=int, default=4000)
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--seed", type=int, default=1)
  ap.add_argument("--kappa", type=float, default=None, help="override MJLAB_LYAP_KAPPA")
  ap.add_argument(
    "--v-half", type=float, default=None, help="override MJLAB_LYAP_V_HALF"
  )
  ap.add_argument("--feedback", action="store_true", help="close the tape's LQR loop")
  ap.add_argument("--log-root", default="logs/rsl_rl")
  ap.add_argument("--run-name-prefix", default="ablate")
  ap.add_argument(
    "--tag",
    default="",
    help="run-name suffix, e.g. k60, to keep "
    "two settings of the same arm as separate runs",
  )
  ap.add_argument("--dry-run", action="store_true")
  a = ap.parse_args()

  for path in (a.motion_file, a.tape_file):
    if not Path(path).exists():
      ap.error(f"missing {path}; run scripts/tools/lqr_export_to_tape.py first")

  stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  manifest_dir = REPO / a.log_root / "ablations"
  manifest_dir.mkdir(parents=True, exist_ok=True)
  manifest = manifest_dir / f"{a.run_name_prefix}-{stamp}.json"
  records: list[dict] = []

  for arm in a.arms:
    env = dict(os.environ)
    env["MJLAB_ABLATION"] = arm
    env["MJLAB_PRIOR_TAPE"] = a.tape_file
    env["MJLAB_PRIOR_FEEDBACK"] = "1" if a.feedback else "0"
    if a.kappa is not None:
      env["MJLAB_LYAP_KAPPA"] = repr(a.kappa)
    if a.v_half is not None:
      env["MJLAB_LYAP_V_HALF"] = repr(a.v_half)

    tag = f"-{a.tag}" if a.tag else ""
    run_name = f"{a.run_name_prefix}-{arm}{tag}"
    cmd = [
      "uv", "run", "train", a.task,
      "--env.scene.num-envs", str(a.num_envs),
      "--env.commands.motion.motion-file", a.motion_file,
      "--agent.max-iterations", str(a.iterations),
      "--agent.seed", str(a.seed),
      "--agent.run-name", run_name,
      "--log-root", a.log_root,
    ]  # fmt: skip

    print(f"\n{'=' * 78}\n[{arm}] {run_name}\n{'=' * 78}")
    print(" ".join(cmd))
    overrides = {k: env[k] for k in env if k.startswith("MJLAB_")}
    print(f"env: {overrides}")
    if a.dry_run:
      continue

    started = time.time()
    proc = subprocess.run(cmd, cwd=REPO, env=env)
    records.append(
      {
        "arm": arm,
        "run_name": run_name,
        "returncode": proc.returncode,
        "minutes": round((time.time() - started) / 60, 1),
        "env": overrides,
        "cmd": cmd,
      }
    )
    manifest.write_text(json.dumps(records, indent=2))
    if proc.returncode != 0:
      print(f"[{arm}] FAILED with {proc.returncode}; stopping the sweep.")
      return proc.returncode

  if not a.dry_run:
    print(f"\nwrote {manifest}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
