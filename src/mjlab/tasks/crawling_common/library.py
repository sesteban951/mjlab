"""Declarative gait-library specs for the library-tracking controllers (single source of truth).

Each library controller (env) tracks a curated subset of an mj-nlp master gait grid
(``<gait_root>/gait_library_<family>/``), converted to mjlab tracking npz. A :class:`LibrarySpec`
names that subset -- per source family, an axis-equality filter on the twist label ``[vx, vy, wz]``
-- plus whether to append a zero-twist idle clip, and where its grids come from: the crawl grids
are ``examples/g1_gait`` at 200 Hz with the prone idle pose, the upright walk grids are
``examples/g1_mimic_periodic/library`` at 100 Hz with a standing one. Because every gait of a spec
was generated at a common period, any mix stacks in one ``LibraryMotionLoader``.

``scripts/build_library.py`` reads these specs to (re)build ``<name>_tracking/``; the env configs
read their ``MOTION_DIR`` from the SAME spec (``LIBRARY_SPECS[key].tracking_dir``), so the
selection and the env can never drift. Central registry: add a controller by adding one entry here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import mjlab

# mjlab repo root and its trajectories dir (where the built libraries live).
_MJLAB_ROOT = Path(mjlab.MJLAB_SRC_PATH).parent.parent
_TRAJ_DIR = _MJLAB_ROOT / "trajectories"
_LIB_DIR = _TRAJ_DIR / "library"  # gait libraries (staged + tracking) live here
# Master gait grids live in the sibling mj-nlp repo (override with build_library --gait-root).
_MJNLP_ROOT = _MJLAB_ROOT.parent / "mj-nlp"
DEFAULT_GAIT_ROOT = _MJNLP_ROOT / "examples" / "g1_gait"  # crawl grids, 200 Hz
WALK_GAIT_ROOT = (
  _MJNLP_ROOT / "examples" / "g1_mimic_periodic" / "library"
)  # upright, 100 Hz
# Idle pose -> zero-twist stop clip (same csv used to re-center the crawl action space). As with
# the crawl, the idle csv lives INSIDE a staged library folder; build_library only clears that
# folder's npz copies on a rebuild, so the csv survives.
DEFAULT_IDLE_CSV = _LIB_DIR / "crawl_ff_loop_180_R_001__A229_library" / "qpos_idle.csv"
# Standing idle for the upright walk library: a verbatim copy of mj-nlp's
# trajectories/g1/poses/stand_idle_qpos.csv (captured static stand, base z 0.776, heading +x,
# COM 3.5 cm inside the support polygon). Re-copy it here if that file changes.
WALK_IDLE_CSV = _LIB_DIR / "walk_diffdrive_library" / "qpos_idle.csv"
# The upright WALK and JOG grids are both written by mj-nlp's g1_mimic_periodic library driver into
# the same folder -- the family name (gait_library_<family>) is what separates them.
PERIODIC_GAIT_ROOT = WALK_GAIT_ROOT
# Standing idle for the jog library. The jog stop is the same static stand the walk library uses;
# replace this csv if the jog task should rest in a different (e.g. ready-stance) pose.
JOG_IDLE_CSV = _LIB_DIR / "jog_unicycle_library" / "qpos_idle.csv"

# The 29 actuated G1 joints in the order the mj-nlp ``state = [qpos | qvel]`` stores them, i.e.
# qpos columns 7:36 of every library clip and of a ``qpos_idle.csv`` row. The converter
# (scripts/library_to_npz.py) and the env configs that read an idle csv all key off this tuple.
QPOS_JOINT_ORDER = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)


def load_idle_qpos(
  idle_csv: Path,
) -> tuple[
  tuple[float, float, float], tuple[float, float, float, float], dict[str, float]
]:
  """One ``qpos_idle.csv`` row -> (base pos, base quat wxyz, {joint name: angle}).

  The joint dict is keyed by :data:`QPOS_JOINT_ORDER`, ready for ``InitialStateCfg.joint_pos``
  so an env can start (and centre its zero action) on the same pose its idle clip tracks."""
  qpos = np.loadtxt(idle_csv, delimiter=",").reshape(-1)
  if qpos.shape[0] != 7 + len(QPOS_JOINT_ORDER):
    raise ValueError(
      f"{idle_csv}: expected {7 + len(QPOS_JOINT_ORDER)} columns "
      f"[pos(3) | quat wxyz(4) | joints({len(QPOS_JOINT_ORDER)})], got {qpos.shape[0]}"
    )
  pos = tuple(float(v) for v in qpos[0:3])
  quat = tuple(float(v) for v in qpos[3:7])
  joints = {n: float(v) for n, v in zip(QPOS_JOINT_ORDER, qpos[7:], strict=True)}
  return pos, quat, joints  # type: ignore[return-value]


@dataclass(frozen=True)
class Source:
  """One master-grid family + an axis-equality filter on its per-clip twist label.

  ``keep`` maps twist axis name (``"vx"``/``"vy"``/``"wz"``) to the value a clip must have (within
  ``tol``) to be selected; an empty ``keep`` takes the whole family. Filtering reads the ``twist``
  array stored in each npz (robust -- not filename parsing).

  ``decimate`` takes every n-th frame while staging, so families solved at different sim_dt can
  share one spec: a spec carries ONE ``input_fps`` for every clip it stages, and the converter
  resamples all of them off that single rate. Decimating to the common (lowest) rate is exact
  rather than lossy -- the converter lands on a 50 Hz grid by interpolation, so at an integer ratio
  the blend weight is zero and it selects source frames; 200 -> 100 -> 50 picks the same frames as
  200 -> 50. Prefer it over interpolating a slower family UP, which would invent frames."""

  family: str
  keep: dict[str, float] = field(default_factory=dict)
  tol: float = 1e-6
  decimate: int = 1


@dataclass(frozen=True)
class LibrarySpec:
  """A controller's tracking library: a filtered union of master-grid sources + an optional idle clip.

  ``name`` is the on-disk stem: ``<name>_library`` (staged inputs) -> ``<name>_tracking``
  (converted output the env loads). ``gait_root``/``input_fps`` say where the master grids are
  and how they are sampled; ``idle_csv``/``idle_name`` give the stop pose and the converted idle
  clip's filename. Every clip of one spec must share a period (the loader stacks them)."""

  name: str
  sources: tuple[Source, ...]
  idle: bool = True
  gait_root: Path = (
    DEFAULT_GAIT_ROOT  # master grids: <gait_root>/gait_library_<family>/
  )
  idle_csv: Path = DEFAULT_IDLE_CSV  # one qpos row -> the zero-twist stop clip
  input_fps: float = 200.0  # sample rate of the master-grid clips (crawl 200, walk 100)
  idle_name: str = "crawl_fwd_vx_000.npz"  # converted idle clip's filename

  @property
  def library_dir(self) -> Path:
    """Staged-inputs dir (mj-nlp [qpos|qvel] schema) the converter reads."""
    return _LIB_DIR / f"{self.name}_library"

  @property
  def tracking_dir(self) -> Path:
    """Converted tracking-npz dir the env's ``MOTION_DIR`` points at."""
    return _LIB_DIR / f"{self.name}_tracking"


# ---- central registry: one entry per library controller -----------------------------------------

LIBRARY_SPECS: dict[str, LibrarySpec] = {
  # Omnidirectional crawl: ALL directions -- full forward + backward grids + in-place turns.
  "omni": LibrarySpec(
    name="crawl_omni",
    sources=(
      Source("crawl_fwd"),  # full forward grid (vx>0, + crab + curve)
      Source("crawl_bck"),  # full backward grid (vx<0)
      Source("crawl_turn_pos"),  # in-place left turns
      Source("crawl_turn_neg"),  # in-place right turns
    ),
  ),
  # Differential-drive crawl (tank): straight forward + backward columns + in-place turns.
  "diffdrive": LibrarySpec(
    name="crawl_diffdrive",
    sources=(
      Source("crawl_fwd", keep={"vy": 0.0, "wz": 0.0}),
      Source("crawl_bck", keep={"vy": 0.0, "wz": 0.0}),
      Source("crawl_turn_pos"),
      Source("crawl_turn_neg"),
    ),
  ),
  # Differential-drive UPRIGHT walking: the g1_mimic_periodic walk grids (T = 1.4 s, 141 frames at
  # 100 Hz -> 70 tracking frames). The grids are already pure -- walk_fwd/bck sweep vx alone, the
  # turns sweep wz alone -- so the filters are documentation; the idle is a standing pose.
  "standing_diffdrive": LibrarySpec(
    name="walk_diffdrive",
    sources=(
      Source("walk_fwd", keep={"vy": 0.0, "wz": 0.0}),  # vx +0.50 .. +1.00
      Source("walk_bck", keep={"vy": 0.0, "wz": 0.0}),  # vx -0.90 .. -0.40
      Source("walk_turn_pos", keep={"vx": 0.0, "vy": 0.0}),  # wz +0.50 .. +1.50 (pivot)
      Source(
        "walk_turn_neg", keep={"vx": 0.0, "vy": 0.0}
      ),  # wz -1.50 .. -0.50 (mirror)
    ),
    gait_root=WALK_GAIT_ROOT,
    idle_csv=WALK_IDLE_CSV,
    input_fps=100.0,
    idle_name="walk_idle.npz",
  ),
  # UNICYCLE JOGGING: a genuine 2-D (vx, wz) grid, not two 1-D columns. The straight/backward
  # families sweep BOTH axes -- that is the whole difference from the diff-drive walk spec above,
  # whose sources pin wz = 0 -- so their filters keep only vy = 0; the pivot families pin vx = 0.
  #
  # THE `family` STRINGS MUST MATCH THE BUILT GRIDS. mj-nlp writes each family to
  # <gait_root>/gait_library_<family>/; trim this tuple to the families that actually exist before
  # building. The jog clips come off run_fwd/run_bck configs at sim_dt = 5 ms -> 200 Hz input.
  # MIXED SOURCE RATES, reconciled by decimation. The arc grids were solved at sim_dt = 5 ms
  # (200 Hz) and the pivots at 10 ms (100 Hz); a spec stages everything at ONE input_fps, so the
  # arcs are decimated by 2 to the pivots' rate. Exact, not lossy: the converter's 50 Hz grid is an
  # integer ratio of both, so 200 -> 100 -> 50 selects the same frames as 200 -> 50. All four
  # families share the 0.860 s stride, which is what lets them stack at all.
  "jog_unicycle": LibrarySpec(
    name="jog_unicycle",
    sources=(
      Source("run_fwd", keep={"vy": 0.0}, decimate=2),  # vx>0 x wz sweep (wz=0 = straight)
      Source("run_bck", keep={"vy": 0.0}, decimate=2),  # vx<0 x wz sweep
      Source("walk_turn_pos_fast", keep={"vx": 0.0, "vy": 0.0}),  # in-place left pivots
      Source("walk_turn_neg_fast", keep={"vx": 0.0, "vy": 0.0}),  # in-place right pivots
    ),
    gait_root=PERIODIC_GAIT_ROOT,
    idle_csv=JOG_IDLE_CSV,
    input_fps=100.0,  # the pivots' native rate; the arcs are decimated to it above
    idle_name="jog_idle.npz",
  ),
}
