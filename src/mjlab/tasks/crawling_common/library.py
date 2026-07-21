"""Declarative gait-library specs for the crawl controllers (single source of truth).

Each crawl controller (env) tracks a curated subset of the mj-nlp master gait grids
(``gait_library_crawl_{fwd,bck,turn_pos,turn_neg}``), converted to mjlab tracking npz. A
:class:`LibrarySpec` names that subset -- per source family, an axis-equality filter on the twist
label ``[vx, vy, wz]`` -- plus whether to append a zero-twist idle clip. Because every gait was
generated at a common period, any mix stacks in one ``LibraryMotionLoader``.

``scripts/build_library.py`` reads these specs to (re)build ``crawl_<name>_tracking/``; the env
configs read their ``MOTION_DIR`` from the SAME spec (``LIBRARY_SPECS[name].tracking_dir``), so the
selection and the env can never drift. Central registry: add a controller by adding one entry here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mjlab

# mjlab repo root and its trajectories dir (where the built libraries live).
_MJLAB_ROOT = Path(mjlab.MJLAB_SRC_PATH).parent.parent
_TRAJ_DIR = _MJLAB_ROOT / "trajectories"
_LIB_DIR = _TRAJ_DIR / "library"  # crawl gait libraries (staged + tracking) live here
# Master gait grids live in the sibling mj-nlp repo (override with build_library --gait-root).
DEFAULT_GAIT_ROOT = _MJLAB_ROOT.parent / "mj-nlp" / "examples" / "g1_gait"
# Idle pose -> zero-twist stop clip (same csv used to re-center the crawl action space).
DEFAULT_IDLE_CSV = _LIB_DIR / "crawl_ff_loop_180_R_001__A229_library" / "qpos_idle.csv"


@dataclass(frozen=True)
class Source:
  """One master-grid family + an axis-equality filter on its per-clip twist label.

  ``keep`` maps twist axis name (``"vx"``/``"vy"``/``"wz"``) to the value a clip must have (within
  ``tol``) to be selected; an empty ``keep`` takes the whole family. Filtering reads the ``twist``
  array stored in each npz (robust -- not filename parsing)."""

  family: str
  keep: dict[str, float] = field(default_factory=dict)
  tol: float = 1e-6


@dataclass(frozen=True)
class LibrarySpec:
  """A controller's tracking library: a filtered union of master-grid sources + an optional idle clip.

  ``name`` sets the on-disk convention ``crawl_<name>_library`` (staged inputs) ->
  ``crawl_<name>_tracking`` (converted output the env loads)."""

  name: str
  sources: tuple[Source, ...]
  idle: bool = True

  @property
  def library_dir(self) -> Path:
    """Staged-inputs dir (mj-nlp [qpos|qvel] schema) the converter reads."""
    return _LIB_DIR / f"crawl_{self.name}_library"

  @property
  def tracking_dir(self) -> Path:
    """Converted tracking-npz dir the env's ``MOTION_DIR`` points at."""
    return _LIB_DIR / f"crawl_{self.name}_tracking"


# ---- central registry: one entry per crawl controller ------------------------------------------

LIBRARY_SPECS: dict[str, LibrarySpec] = {
  # Omnidirectional: ALL directions -- full forward + backward grids + in-place turns.
  "omni": LibrarySpec(
    name="omni",
    sources=(
      Source("crawl_fwd"),       # full forward grid (vx>0, + crab + curve)
      Source("crawl_bck"),       # full backward grid (vx<0)
      Source("crawl_turn_pos"),  # in-place left turns
      Source("crawl_turn_neg"),  # in-place right turns
    ),
  ),
  # Differential-drive (tank): straight forward + straight backward columns + in-place turns.
  "diffdrive": LibrarySpec(
    name="diffdrive",
    sources=(
      Source("crawl_fwd", keep={"vy": 0.0, "wz": 0.0}),
      Source("crawl_bck", keep={"vy": 0.0, "wz": 0.0}),
      Source("crawl_turn_pos"),
      Source("crawl_turn_neg"),
    ),
  ),
}
