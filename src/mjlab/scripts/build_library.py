"""Build a per-controller tracking library from its declarative LibrarySpec.

Selects a filtered subset of an mj-nlp master gait grid (by twist label), stages it (+ optional
idle csv) into ``<name>_library/``, then FK-converts it into ``<name>_tracking/`` via
``library_to_npz`` at the spec's own sample rate. The selection lives in
``crawling_common.library.LIBRARY_SPECS``; the env configs read their ``MOTION_DIR`` from the same
specs, so the two never drift. Rebuild after the master grids change (e.g. the fwd grid finishing)
to refresh the tracking libraries.

  uv run python -m mjlab.scripts.build_library diffdrive            # build one controller
  uv run python -m mjlab.scripts.build_library walking_diffdrive   # the upright walk library
  uv run python -m mjlab.scripts.build_library all                  # build every registered controller
  uv run python -m mjlab.scripts.build_library omni --no-convert True  # dry-run: selection only

A spec with ``merges`` instead of ``sources`` is assembled by copying banded selections
out of already-built tracking libraries (no staging, no conversion). ``--gait-root`` and
``--idle-csv`` do not apply to it and are rejected; ``--no-convert`` only prints the
selection.

(Once ``uv sync`` has registered the console script, ``uv run build-library diffdrive`` also works.)
"""

import glob
import itertools
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import tyro

import mjlab
from mjlab.tasks.crawling_common.library import (
  LIBRARY_SPECS,
  LibrarySpec,
  MergeSource,
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


def _stage_decimated(src_path, dst_path, step: int) -> None:
  """Copy one clip taking every ``step``-th frame, so a family solved at a finer sim_dt can be
  staged at the spec's common ``input_fps``.

  Only the PER-FRAME arrays are sliced -- ``state`` and ``time`` (what the converter reads) and
  ``input`` (one shorter). Everything else (the twist label, the node-grid defects, the solver
  metadata) is carried through untouched, because it is not indexed by frame.
  """
  import numpy as np

  d = np.load(src_path, allow_pickle=True)
  n = int(np.shape(d["state"])[0])
  out = {}
  for k in d.files:
    v = d[k]
    if k in ("state", "time") and np.ndim(v) >= 1 and np.shape(v)[0] == n:
      out[k] = v[::step]
    elif k == "input" and np.ndim(v) >= 1 and np.shape(v)[0] == n - 1:
      out[k] = v[::step]
    else:
      out[k] = v
  np.savez(dst_path, **out)


def _verify(spec: LibrarySpec, tracking_dir: Optional[Path] = None) -> list[str]:
  """Post-build checks on a tracking dir; returns failure reasons (empty means OK)."""
  d = tracking_dir or spec.tracking_dir
  files = sorted(glob.glob(str(d / "*.npz")))
  if not files:
    return [f"no tracking clips in {d}"]
  frames = {int(np.load(f)["joint_pos"].shape[0]) for f in files}
  twists = np.array([np.asarray(np.load(f)["twist"]).reshape(3) for f in files])
  has_idle = bool((np.abs(twists).sum(1) < 1e-9).any())
  dupes = len(twists) - len({tuple(np.round(t, 4)) for t in twists})
  reasons = []
  if len(frames) > 1 and not spec.ragged:
    reasons.append(f"frame counts differ {sorted(frames)} but the spec is not ragged")
  if spec.idle and not has_idle:
    reasons.append("spec.idle=True but there is no zero-twist clip")
  if dupes:
    reasons.append(f"{dupes} duplicate twist label(s)")
  print(
    f"  verify: {len(files)} clips | T={sorted(frames)} | "
    f"idle={'yes' if has_idle else 'no'} | duplicate twists={dupes} -> "
    f"{'OK' if not reasons else 'FAIL'}"
  )
  return reasons


def _require_ok(spec: LibrarySpec, tracking_dir: Optional[Path] = None) -> None:
  reasons = _verify(spec, tracking_dir)
  if reasons:
    raise RuntimeError(
      f"library '{spec.name}' failed verification:\n  " + "\n  ".join(reasons)
    )


_EPS = 1e-9


def _merge_keep(twist: np.ndarray, m: MergeSource) -> bool:
  """True if a converted clip's twist falls in the merge source's band for its own mode.

  Bands are half-open ``[lo, hi)``, so two sources meeting at a shared boundary value
  can never both keep the same node."""
  vx, vy, wz = (float(v) for v in twist)
  if abs(vy) > _EPS:
    raise ValueError(
      f"merge bands are defined on vx/wz; lateral twist {twist} unsupported"
    )
  if abs(vx) < _EPS and abs(wz) < _EPS:
    return m.idle
  if abs(vx) < _EPS:  # in-place pivot: banded on |wz|
    return m.pivot_wz is not None and m.pivot_wz[0] <= abs(wz) < m.pivot_wz[1]
  band = m.fwd_vx if vx > 0 else m.bck_vx  # moving: banded on vx, any wz
  return band is not None and band[0] <= vx < band[1]


def build_merged(spec: LibrarySpec, convert: bool = True) -> None:
  """Assemble a tracking library by copying banded selections out of already-built ones.

  No staging and no FK conversion: the sources are already in tracking format, so clips
  are copied verbatim and keep their own frame counts (see ``LibrarySpec.ragged``).
  Everything is validated BEFORE the output dir is touched, and the copy lands in a temp
  dir that replaces the old library only after it verifies, so a failed build never
  leaves a half-library behind. ``convert=False`` prints the selection and stops."""
  out = spec.tracking_dir
  print(f"\n== library '{spec.name}' -> {out.name}  (merge) ==")
  for a, b in itertools.combinations(spec.merges, 2):
    for band in ("fwd_vx", "bck_vx", "pivot_wz"):
      x, y = getattr(a, band), getattr(b, band)
      if x and y and x[0] < y[1] and y[0] < x[1]:
        raise ValueError(
          f"merge sources {a.spec}/{b.spec} overlap on {band}: {x} vs {y}"
        )
  plan: dict[str, str] = {}  # destination basename -> source path
  for m in spec.merges:
    if m.spec not in LIBRARY_SPECS:
      raise KeyError(f"merge source '{m.spec}' is not a registered LibrarySpec")
    src_dir = LIBRARY_SPECS[m.spec].tracking_dir
    files = sorted(glob.glob(str(src_dir / "*.npz")))
    if not files:
      raise FileNotFoundError(
        f"merge source '{m.spec}' has no clips at {src_dir}; build it first with "
        f"`uv run python -m mjlab.scripts.build_library {m.spec}`"
      )
    kept = [f for f in files if _merge_keep(np.load(f)["twist"].reshape(3), m)]
    if not kept:
      raise ValueError(f"merge source '{m.spec}' keeps no clips under its bands")
    for f in kept:
      name = os.path.basename(f)
      if name in plan:
        raise FileExistsError(f"merge sources collide on filename: {name}")
      plan[name] = f
    frames = sorted({int(np.load(f)["joint_pos"].shape[0]) for f in kept})
    print(f"  {m.spec:16s} -> {len(kept):>3d} clips  T={frames}")
  print(f"  planned {len(plan)} clips for {out}")
  if not convert:
    print("  --no-convert: selection only, tracking library NOT written.")
    return
  tmp = out.with_name(out.name + ".tmp")
  if tmp.exists():
    shutil.rmtree(tmp)
  tmp.mkdir(parents=True)
  for name, src in plan.items():
    shutil.copy(src, tmp / name)
  _require_ok(spec, tmp)
  if out.exists():
    shutil.rmtree(out)
  tmp.rename(out)
  print(f"  merged {len(plan)} clips into {out}")


def build(
  spec: LibrarySpec,
  gait_root: Optional[Path] = None,
  idle_csv: Optional[Path] = None,
  convert: bool = True,
) -> None:
  """Stage the spec's filtered clip selection (+ idle) and (optionally) FK-convert to tracking npz.

  ``gait_root`` / ``idle_csv`` override the spec's own (CLI use); None takes the spec's."""
  if spec.merges:
    if gait_root is not None or idle_csv is not None:
      raise ValueError("--gait-root / --idle-csv do not apply to a merge spec")
    build_merged(spec, convert=convert)
    return
  gait_root = gait_root or spec.gait_root
  idle_csv = idle_csv or spec.idle_csv
  print(f"\n== library '{spec.name}' -> {spec.tracking_dir.name} ==")
  print(f"  grids: {gait_root} @ {spec.input_fps:g} Hz")
  lib = spec.library_dir
  # Clear only the staged npz copies. The staging folder may also be the HOME of the spec's idle
  # csv (the walk library keeps its standing pose there, like the original crawl library), and
  # that must survive a rebuild.
  lib.mkdir(parents=True, exist_ok=True)
  for stale in lib.glob("*.npz"):
    stale.unlink()

  total = 0
  for src in spec.sources:
    selected = _select(src, gait_root)
    for f in selected:
      if src.decimate > 1:
        _stage_decimated(f, lib / os.path.basename(f), src.decimate)
      else:
        shutil.copy(f, lib / os.path.basename(f))
    filt = src.keep if src.keep else "all"
    dec = f"  decimate={src.decimate}" if src.decimate > 1 else ""
    print(f"  {src.family:16s} keep={filt}  -> {len(selected):>3d} clips{dec}")
    total += len(selected)

  if spec.idle:
    if not idle_csv.exists():
      raise FileNotFoundError(f"idle=True but qpos_idle.csv not found: {idle_csv}")
    staged = lib / "qpos_idle.csv"
    if idle_csv.resolve() != staged.resolve():  # already in place when it lives here
      shutil.copy(idle_csv, staged)
    print(f"  {'idle':16s}          -> {idle_csv}")

  print(f"  staged {total} clips (+idle={spec.idle}) into {lib}")
  if not convert:
    print("  --no-convert: selection only, tracking library NOT built.")
    return

  # Lazy import: only the actual conversion needs torch / the mjlab sim scene.
  from mjlab.scripts.library_to_npz import main as convert_to_tracking

  convert_to_tracking(
    input_dir=str(lib),
    output_dir=str(spec.tracking_dir),
    input_fps=spec.input_fps,
    idle_out=spec.idle_name,
  )
  _require_ok(spec)


def main(
  controller: tyro.conf.Positional[str],
  gait_root: Optional[str] = None,
  idle_csv: Optional[str] = None,
  no_convert: bool = False,
) -> None:
  """Build one controller's tracking library, or ``all`` of them.

  Args:
    controller: a key in ``LIBRARY_SPECS`` (e.g. ``diffdrive``, ``walking_diffdrive``) or
      ``all``.
    gait_root: master-grid root, overriding every built spec's own (the crawl specs default to
      the sibling mj-nlp ``examples/g1_gait``, the walk spec to
      ``examples/g1_mimic_periodic/library``).
    idle_csv: qpos_idle.csv for the zero-twist stop clip, overriding the spec's own.
    no_convert: stage + print the selection only; skip the FK conversion (dry run).
  """
  gr = Path(gait_root) if gait_root else None
  ic = Path(idle_csv) if idle_csv else None
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
