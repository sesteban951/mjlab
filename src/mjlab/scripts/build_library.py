"""Build a per-controller crawl tracking library from its declarative LibrarySpec.

Selects a filtered subset of the mj-nlp master gait grids (by twist label), stages it (+ optional
idle csv) into ``crawl_<name>_library/``, then FK-converts it into ``crawl_<name>_tracking/`` via
``library_to_npz``. The selection lives in ``crawling_common.library.LIBRARY_SPECS``; the env configs
read their ``MOTION_DIR`` from the same specs, so the two never drift. Rebuild after the master grids
change (e.g. the fwd grid finishing) to refresh the tracking libraries.

  uv run python -m mjlab.scripts.build_library diffdrive            # build one controller
  uv run python -m mjlab.scripts.build_library all                  # build every registered controller
  uv run python -m mjlab.scripts.build_library omni --no-convert True  # dry-run: selection only

(Once ``uv sync`` has registered the console script, ``uv run build-library diffdrive`` also works.)
"""

import glob
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import tyro

import mjlab
from mjlab.tasks.crawling_common.library import (
  DEFAULT_GAIT_ROOT,
  DEFAULT_IDLE_CSV,
  LIBRARY_SPECS,
  LibrarySpec,
  Source,
)

_AXIS = {"vx": 0, "vy": 1, "wz": 2}


def _matches(twist: np.ndarray, src: Source) -> bool:
  """True if the clip's twist satisfies every axis-equality constraint in the source filter."""
  return all(abs(float(twist[_AXIS[k]]) - v) <= src.tol for k, v in src.keep.items())


def _select(src: Source, gait_root: Path) -> list[str]:
  """The npz paths in the source family whose twist label passes the filter."""
  family_dir = gait_root / f"gait_library_{src.family}"
  files = sorted(glob.glob(str(family_dir / "*.npz")))
  if not files:
    raise FileNotFoundError(f"no clips for family '{src.family}' under {family_dir}")
  keep = []
  for f in files:
    twist = np.asarray(np.load(f)["twist"], dtype=float).reshape(3)
    if _matches(twist, src):
      keep.append(f)
  return keep


def _verify(spec: LibrarySpec) -> None:
  """Post-build check: uniform frame count across the tracking clips + idle present if requested."""
  files = sorted(glob.glob(str(spec.tracking_dir / "*.npz")))
  frames = {int(np.load(f)["joint_pos"].shape[0]) for f in files}
  twists = np.array([np.asarray(np.load(f)["twist"]).reshape(3) for f in files])
  has_idle = bool((np.abs(twists).sum(1) < 1e-9).any())
  ok = len(frames) == 1 and (has_idle or not spec.idle)
  print(
    f"  verify: {len(files)} tracking clips | T={sorted(frames)} | "
    f"idle={'yes' if has_idle else 'no'} -> {'OK' if ok else 'CHECK'}"
  )


def build(
  spec: LibrarySpec, gait_root: Path, idle_csv: Path, convert: bool = True
) -> None:
  """Stage the spec's filtered clip selection (+ idle) and (optionally) FK-convert to tracking npz."""
  print(f"\n== library '{spec.name}' -> {spec.tracking_dir.name} ==")
  lib = spec.library_dir
  if lib.exists():
    shutil.rmtree(lib)
  lib.mkdir(parents=True)

  total = 0
  for src in spec.sources:
    selected = _select(src, gait_root)
    for f in selected:
      shutil.copy(f, lib / os.path.basename(f))
    filt = src.keep if src.keep else "all"
    print(f"  {src.family:16s} keep={filt}  -> {len(selected):>3d} clips")
    total += len(selected)

  if spec.idle:
    if not idle_csv.exists():
      raise FileNotFoundError(f"idle=True but qpos_idle.csv not found: {idle_csv}")
    shutil.copy(idle_csv, lib / "qpos_idle.csv")
    print(f"  {'idle':16s}          -> {idle_csv.name}")

  print(f"  staged {total} clips (+idle={spec.idle}) into {lib}")
  if not convert:
    print("  --no-convert: selection only, tracking library NOT built.")
    return

  # Lazy import: only the actual conversion needs torch / the mjlab sim scene.
  from mjlab.scripts.library_to_npz import main as convert_to_tracking

  convert_to_tracking(input_dir=str(lib), output_dir=str(spec.tracking_dir))
  _verify(spec)


def main(
  controller: tyro.conf.Positional[str],
  gait_root: Optional[str] = None,
  idle_csv: Optional[str] = None,
  no_convert: bool = False,
) -> None:
  """Build one controller's tracking library, or ``all`` of them.

  Args:
    controller: a key in ``LIBRARY_SPECS`` (e.g. ``diffdrive``, ``omni``) or ``all``.
    gait_root: master-grid root (defaults to the sibling mj-nlp ``examples/g1_gait``).
    idle_csv: qpos_idle.csv for the zero-twist stop clip (defaults to the crawl idle pose).
    no_convert: stage + print the selection only; skip the FK conversion (dry run).
  """
  gr = Path(gait_root) if gait_root else DEFAULT_GAIT_ROOT
  ic = Path(idle_csv) if idle_csv else DEFAULT_IDLE_CSV
  if controller == "all":
    names = list(LIBRARY_SPECS)
  elif controller in LIBRARY_SPECS:
    names = [controller]
  else:
    raise SystemExit(
      f"unknown controller '{controller}'. known: {sorted(LIBRARY_SPECS)} (or 'all')"
    )
  for nm in names:
    build(LIBRARY_SPECS[nm], gr, ic, convert=not no_convert)
  print(f"\ndone: built {len(names)} librar{'y' if len(names) == 1 else 'ies'}.")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
