"""Convert an mj-nlp gait library into mjlab tracking-motion npz files.

Each source clip is ONE periodic stride as ``state = [qpos(36) | qvel(35)]`` whose closing
frame is frame 0 carried one stride on by an SE(2) map ``g``. The clip is resampled onto
``round(T * output_fps)`` uniform frames, FK-replayed to per-body world kinematics, and
written with ``twist`` and ``stride = [tx, ty, dpsi]`` (``g: p -> Rz(dpsi) p + t``), so
it loops exactly. ``align_phase`` rolls every clip onto one reference clip's gait phase.
An optional ``qpos_idle.csv`` becomes a held zero-twist clip.
  uv run python -m mjlab.scripts.library_to_npz  # output_fps must equal 1 / step_dt
"""

import glob
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.gait_library.library import QPOS_JOINT_ORDER
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_conjugate, quat_mul

_REPO_ROOT = Path(mjlab.MJLAB_SRC_PATH).parent.parent

# The 29 actuated G1 joints, in the order the library `state` stores them (same as csv_to_npz).
JOINT_NAMES = QPOS_JOINT_ORDER

_LOG_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)

# A single-row qpos CSV (``[base_pos(3) | base_quat(4, wxyz) | dof_pos(29)]``) placed in the input
# dir is converted to a held, zero-velocity clip so the policy has an explicit pose to track when
# commanded to stop. It is written with a zero twist ``[vx, vy, wz] = [0, 0, 0]`` so the command's
# ``LibraryMotionLoader`` picks it up as the stop clip (nearest to a zero twist command).
_IDLE_CSV = "qpos_idle.csv"
_IDLE_OUT = "crawl_fwd_vx_000.npz"


def _yaw_about_z(quat_wxyz: np.ndarray) -> float:
  """Heading (yaw about world z) of a wxyz quaternion via swing-twist decomposition."""
  w, _, _, z = (float(v) for v in quat_wxyz)
  if w < 0.0:  # pick the hemisphere so the twist angle is continuous
    w, z = -w, -z
  return 2.0 * float(np.arctan2(z, w))


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Hamilton product of two wxyz quaternions."""
  aw, ax, ay, az = a
  bw, bx, by, bz = b
  return np.array(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    dtype=np.float64,
  )


def _apply_yaw_about_z(quat_wxyz: np.ndarray, dyaw: float) -> np.ndarray:
  """Rotate a wxyz quaternion by ``dyaw`` about the world z axis (world-frame pre-multiply)."""
  qz = np.array([np.cos(dyaw / 2.0), 0.0, 0.0, np.sin(dyaw / 2.0)], dtype=np.float64)
  return _quat_mul(qz, quat_wxyz)


def _rot_z(dyaw: float) -> np.ndarray:
  c, s = np.cos(dyaw), np.sin(dyaw)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _yaw_transform(
  log: dict[str, Any], rows: slice, dyaw: float, t: np.ndarray
) -> None:
  """In place: rotate frames ``rows`` of a tracking log by ``dyaw`` about world z and translate
  by ``t`` (positions), rotating orientations and velocities with it."""
  R = _rot_z(dyaw)
  log["body_pos_w"][rows] = log["body_pos_w"][rows] @ R.T + t
  q = log["body_quat_w"][rows]
  qz = np.array([np.cos(dyaw / 2.0), 0.0, 0.0, np.sin(dyaw / 2.0)], dtype=np.float64)
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  log["body_quat_w"][rows] = np.stack(  # qz (x) q, world-frame pre-multiply
    [
      qz[0] * w - qz[3] * z,
      qz[0] * x - qz[3] * y,
      qz[0] * y + qz[3] * x,
      qz[0] * z + qz[3] * w,
    ],
    axis=-1,
  )
  for k in ("body_lin_vel_w", "body_ang_vel_w"):
    log[k][rows] = log[k][rows] @ R.T


def _slerp(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
  """Batched slerp of wxyz quaternions ``(N, 4)`` at weights ``w`` ``(N, 1)``, falling
  back to a normalized lerp where the two are too close for the sine form."""
  b = np.where((a * b).sum(-1, keepdims=True) < 0.0, -b, b)
  ang = np.arccos(np.clip((a * b).sum(-1, keepdims=True), -1.0, 1.0))
  small = ang < 1e-6
  sin = np.where(small, 1.0, np.sin(ang))
  q = np.where(
    small,
    (1.0 - w) * a + w * b,
    (np.sin((1.0 - w) * ang) * a + np.sin(w * ang) * b) / sin,
  )
  return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _stride_map(b0: np.ndarray, dxy: np.ndarray, dpsi: float) -> np.ndarray:
  """``[tx, ty, dpsi]`` of the stride map ``g: p -> Rz(dpsi) p + t`` that carries the
  frame-0 pelvis ``b0`` (xy) to ``b0 + dxy``."""
  R = _rot_z(dpsi)[:2, :2]
  return np.array([*(b0 + dxy - R @ b0), dpsi], dtype=np.float64)


def _rebased_stride(stride: np.ndarray, yaw: float, t: np.ndarray) -> np.ndarray:
  """The stride map after the whole clip is moved by ``A: p -> Rz(yaw) p + t``, i.e.
  ``A g A^-1``."""
  RA, Rg = _rot_z(yaw)[:2, :2], _rot_z(stride[2])[:2, :2]
  return np.array([*(RA @ stride[:2] + t - Rg @ t), stride[2]], dtype=np.float64)


def _seam_error(log: dict[str, Any], fps: float) -> tuple[float, float]:
  """Worst gap [m/s] between the stored body velocities and a central difference of the
  stored positions, at the two seam frames and over the interior. A wrong ``stride``
  or a non-periodic velocity shows up as a seam gap far above the interior one."""
  P = log["body_pos_w"].astype(np.float64)
  V = log["body_lin_vel_w"].astype(np.float64)
  tx, ty, dpsi = (float(v) for v in log["stride"])
  R, t = _rot_z(dpsi), np.array([tx, ty, 0.0])
  pad = np.concatenate([(P[-1:] - t) @ R, P, P[:1] @ R.T + t], axis=0)
  err = np.linalg.norm(V - (pad[2:] - pad[:-2]) * fps / 2.0, axis=-1).max(axis=1)
  return float(max(err[0], err[-1])), float(err[1:-1].max())


def _circular_lag(profile: np.ndarray, ref: np.ndarray) -> int:
  """Shift ``s`` (frames, in (-T/2, T/2]) maximising the circular correlation of two (2, T)
  foot-height profiles: ``np.roll(profile, s, axis=1)`` lines up with ``ref``."""
  n = ref.shape[1]
  cc = [float(np.sum(np.roll(profile, s, axis=1) * ref)) for s in range(n)]
  s = int(np.argmax(cc))
  return s if s <= n // 2 else s - n


def _feet_profile(log: dict[str, Any], feet: list[int]) -> np.ndarray:
  z = log["body_pos_w"][:, feet, 2].astype(np.float64)
  return (z - z.mean(axis=0)).T


def _phase_reference(logs: dict[str, dict[str, Any]]) -> str:
  """The straight clip nearest the middle of the forward-speed range; the first clip if the
  library has no forward clips."""
  fwd = {
    n: float(lg["twist"][0]) for n, lg in logs.items() if float(lg["twist"][0]) > 1e-6
  }
  if not fwd:
    return next(iter(logs))
  mid = 0.5 * (min(fwd.values()) + max(fwd.values()))
  return min(
    logs,
    key=lambda n: (
      abs(float(logs[n]["twist"][2])) > 1e-6,  # straight first
      abs(float(logs[n]["twist"][0]) - mid),
      n,
    ),
  )


def _resample_periodic(profile: np.ndarray, n: int) -> np.ndarray:
  """A ``(2, T)`` periodic profile linearly resampled onto ``n`` frames of the same cycle."""
  t = profile.shape[1]
  x = np.arange(n) * t / n
  lo = np.floor(x).astype(int)
  frac = x - lo
  return profile[:, lo % t] * (1.0 - frac) + profile[:, (lo + 1) % t] * frac


def _align_phase(
  logs: dict[str, dict[str, Any]],
  strides: dict[str, tuple[np.ndarray, float]],
  feet: list[int],
  pelvis: int,
  ref_profile: np.ndarray | None = None,
) -> None:
  """Roll every clip so its foot-height profile lines up with the reference clip's (module
  docstring). ``strides[name] = (dxy, dpsi)``, the clip's own per-stride SE(2) displacement.

  ``ref_profile`` (a ``(2, T)`` foot profile from another library) replaces the in-library
  reference; it is resampled onto each clip's own length, so clips of any stride align to it.
  """
  if ref_profile is None:
    ref_name = _phase_reference(logs)
    ref_profile = _feet_profile(logs[ref_name], feet)
    print(f"  phase reference: {ref_name}")
  else:
    print("  phase reference: external profile")
  worst = 0
  for name, log in logs.items():
    n = log["body_pos_w"].shape[0]
    ref = _resample_periodic(ref_profile, n)
    s = _circular_lag(_feet_profile(log, feet), ref)
    if s != 0:
      dxy, dpsi = strides[name]
      b0 = (
        log["body_pos_w"][0, pelvis].astype(np.float64).copy()
      )  # pelvis at OLD frame 0
      for k in _LOG_KEYS:
        log[k] = np.roll(log[k], s, axis=0)
      # s > 0 brings old frames T-s..T-1 (previous stride) to the front: map back by g^-1.
      # s < 0 sends old frames 0..|s|-1 (next stride) to the end: map forward by the stride g.
      d3 = np.array([dxy[0], dxy[1], 0.0])
      if s > 0:
        R = _rot_z(-dpsi)
        _yaw_transform(log, slice(0, s), -dpsi, b0 - R @ (b0 + d3))
      else:
        R = _rot_z(dpsi)
        _yaw_transform(log, slice(n + s, n), dpsi, b0 + d3 - R @ b0)
    # re-base: frame 0's pelvis at the origin (xy), facing +x -- the convention every clip had
    p0 = log["body_pos_w"][0, pelvis].astype(np.float64)
    yaw0 = _yaw_about_z(log["body_quat_w"][0, pelvis])
    R = _rot_z(-yaw0)
    t_rebase = -(R @ np.array([p0[0], p0[1], 0.0]))
    _yaw_transform(log, slice(0, n), -yaw0, t_rebase)
    log["stride"] = _rebased_stride(log["stride"], -yaw0, t_rebase[:2])
    for k in _LOG_KEYS:
      log[k] = log[k].astype(np.float32)
    # seam check: the pelvis must move as smoothly across the rolled seam as anywhere else
    step = np.linalg.norm(np.diff(log["body_pos_w"][:, pelvis, :2], axis=0), axis=-1)
    if s != 0 and step.max() > 3.0 * max(float(np.median(step)), 1e-3) + 1e-3:
      raise RuntimeError(
        f"{name}: rolled seam is discontinuous (max step {step.max():.3f} m)"
      )
    res = _circular_lag(_feet_profile(log, feet), ref)
    worst = max(worst, abs(res))
    print(
      f"  {name}: phase shift {s:+d} frames ({100.0 * s / n:+.1f}%), residual {res:+d}"
    )
  print(f"  phase-aligned {len(logs)} clips: worst residual lag {worst} frames")


class LibraryMotionLoader(MotionLoader):
  """MotionLoader that reads one periodic stride from a library ``[qpos|qvel]`` npz.

  The stride is resampled onto ``round(T * output_fps)`` uniform frames (closing frame
  excluded) and velocities are central differences that wrap through the stride map, so
  the clip loops with no seam. A stride that is not a multiple of the output step is
  time-stretched to the nearest one. The library qvel is discarded."""

  def _load_motion(self):
    self.state = np.asarray(np.load(self.motion_file)["state"], dtype=np.float64)
    # The closing frame is frame 0 one stride on: its offset IS the stride map.
    self.stride_dxy = self.state[-1, 0:2] - self.state[0, 0:2]
    self.stride_dpsi = _yaw_about_z(self.state[-1, 3:7]) - _yaw_about_z(
      self.state[0, 3:7]
    )
    self.input_frames = self.state.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt  # one full stride

  def _interpolate_motion(self):
    n = int(math.floor(self.duration * self.output_fps + 0.5))
    # Source-frame coordinate of each output frame. float64 throughout: the base's
    # float32 phase can land a frame early, and its slerp then returns the stale frame.
    x = np.arange(n, dtype=np.float64) * (self.input_frames - 1) / n
    lo = np.floor(x).astype(np.int64)
    w = (x - lo)[:, None]
    a, b = self.state[lo], self.state[lo + 1]
    out = (1.0 - w) * a + w * b  # base position, joints
    out[:, 3:7] = _slerp(a[:, 3:7], b[:, 3:7], w)  # base orientation, already wxyz
    out = torch.from_numpy(out.astype(np.float32)).to(self.device)
    self.output_frames = n
    self.time_stretch = n * self.output_dt / self.duration
    self.motion_base_poss = out[:, 0:3]
    self.motion_base_rots = out[:, 3:7]
    self.motion_dof_poss = out[:, 7:36]  # 29 actuated joints

  def _compute_velocities(self):
    """Central differences whose neighbours across the seam are the adjacent strides."""
    pos, rot, dof = self.motion_base_poss, self.motion_base_rots, self.motion_dof_poss
    two_dt = 2.0 * self.output_dt
    R = torch.tensor(_rot_z(self.stride_dpsi), dtype=pos.dtype, device=pos.device)
    d = torch.tensor([*self.stride_dxy, 0.0], dtype=pos.dtype, device=pos.device)
    b0 = pos[0] * torch.tensor([1.0, 1.0, 0.0], dtype=pos.dtype, device=pos.device)
    nxt = (pos[:1] - b0) @ R.T + b0 + d  # frame 0 of the next stride
    prv = (pos[-1:] - b0 - d) @ R + b0  # last frame of the previous stride
    pad = torch.cat([prv, pos, nxt], dim=0)
    self.motion_base_lin_vels = (pad[2:] - pad[:-2]) / two_dt
    half = 0.5 * self.stride_dpsi
    qz = torch.tensor(
      [[math.cos(half), 0.0, 0.0, math.sin(half)]], dtype=rot.dtype, device=rot.device
    )
    rpad = torch.cat(
      [quat_mul(quat_conjugate(qz), rot[-1:]), rot, quat_mul(qz, rot[:1])], dim=0
    )
    rel = quat_mul(rpad[2:], quat_conjugate(rpad[:-2]))
    self.motion_base_ang_vels = axis_angle_from_quat(rel) / two_dt
    self.motion_dof_vels = (torch.roll(dof, -1, 0) - torch.roll(dof, 1, 0)) / two_dt


def _replay_to_log(
  sim, scene, robot: Entity, joint_indexes, motion: MotionLoader
) -> dict[str, Any]:
  """FK-replay one clip through the scene and return the stacked tracking-npz log dict."""
  log: dict[str, Any] = {"fps": np.asarray([motion.output_fps])}
  frames: dict[str, list] = {k: [] for k in _LOG_KEYS}
  scene.reset()

  done = False
  while not done:
    (base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel), reset_flag = (
      motion.get_next_state()
    )

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = base_rot
    root_states[:, 7:10] = base_lin_vel
    root_states[:, 10:] = base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, joint_indexes] = dof_pos
    joint_vel[:, joint_indexes] = dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    frames["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
    frames["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
    frames["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
    frames["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
    frames["body_lin_vel_w"].append(
      robot.data.body_link_lin_vel_w[0].cpu().numpy().copy()
    )
    frames["body_ang_vel_w"].append(
      robot.data.body_link_ang_vel_w[0].cpu().numpy().copy()
    )

    # Sanity: the sim's base body velocity must equal the finite-diff base velocity we fed in.
    # Loose tol (float32 GPU roundtrip is ~1e-5; a real frame-convention bug would be O(0.1+)).
    torch.testing.assert_close(
      robot.data.body_link_lin_vel_w[0, 0], base_lin_vel[0], rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
      robot.data.body_link_ang_vel_w[0, 0], base_ang_vel[0], rtol=1e-3, atol=1e-3
    )

    if reset_flag:  # wrapped past the last frame -> this frame was the last one
      done = True

  for k in _LOG_KEYS:
    log[k] = np.stack(frames[k], axis=0)
  return log


def _idle_to_log(
  sim,
  scene,
  robot: Entity,
  joint_indexes,
  idle_qpos,
  num_frames: int,
  output_fps: int,
) -> dict[str, Any]:
  """FK a single held pose and tile it to ``num_frames`` frames with zero velocities.

  The idle clip is a static reference -- every frame identical, all velocities exactly zero -- so
  the policy has an explicit pose to imitate at zero commanded speed rather than learning a rest
  from shaping rewards. ``idle_qpos`` is one G1 qpos row:
  ``[base_pos(3) | base_quat(4, wxyz) | dof_pos(29)]``.
  """
  idle_qpos = np.asarray(idle_qpos, dtype=np.float32).reshape(-1)
  base_pos = torch.tensor(idle_qpos[0:3], device=sim.device).unsqueeze(0)
  base_rot = torch.tensor(idle_qpos[3:7], device=sim.device).unsqueeze(0)  # wxyz, as-is
  dof_pos = torch.tensor(idle_qpos[7:36], device=sim.device).unsqueeze(0)

  scene.reset()
  root_states = robot.data.default_root_state.clone()
  root_states[:, 0:3] = base_pos
  root_states[:, :2] += scene.env_origins[:, :2]
  root_states[:, 3:7] = base_rot
  root_states[:, 7:] = 0.0  # zero base linear + angular velocity
  robot.write_root_state_to_sim(root_states)

  joint_pos = robot.data.default_joint_pos.clone()
  joint_vel = robot.data.default_joint_vel.clone()
  joint_pos[:, joint_indexes] = dof_pos
  joint_vel[:, joint_indexes] = 0.0
  robot.write_joint_state_to_sim(joint_pos, joint_vel)

  sim.forward()
  scene.update(sim.mj_model.opt.timestep)

  jp = robot.data.joint_pos[0].cpu().numpy().copy()
  bp = robot.data.body_link_pos_w[0].cpu().numpy().copy()
  bq = robot.data.body_link_quat_w[0].cpu().numpy().copy()
  nj, nb = jp.shape[0], bp.shape[0]

  def _hold(a: np.ndarray) -> np.ndarray:
    return np.repeat(a[None], num_frames, axis=0)

  return {
    "fps": np.asarray([output_fps]),
    "joint_pos": _hold(jp),
    "joint_vel": np.zeros((num_frames, nj), dtype=np.float32),
    "body_pos_w": _hold(bp),
    "body_quat_w": _hold(bq),
    "body_lin_vel_w": np.zeros((num_frames, nb, 3), dtype=np.float32),
    "body_ang_vel_w": np.zeros((num_frames, nb, 3), dtype=np.float32),
    "twist": np.zeros(3, dtype=np.float32),  # [vx, vy, wz] = 0 -> the stop clip
    "stride": np.zeros(3, dtype=np.float64),  # a held pose goes nowhere
    "nominal_speed": np.asarray(0.0, dtype=np.float32),
  }


def main(
  input_dir: str = str(
    _REPO_ROOT
    / "trajectories"
    / "g1"
    / "library"
    / "crawl_ff_loop_180_R_001__A229_library"
  ),
  output_dir: str = str(
    _REPO_ROOT
    / "trajectories"
    / "g1"
    / "library"
    / "crawl_ff_loop_180_R_001__A229_tracking"
  ),
  input_fps: float = 200.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  idle_out: str = _IDLE_OUT,
  align_phase: bool = True,
  phase_ref: str | None = None,
):
  """Convert every ``*.npz`` in ``input_dir`` (mj-nlp ``[qpos|qvel]``) to tracking npz.

  Args:
    input_dir: Folder of mj-nlp clips; an optional ``qpos_idle.csv`` becomes a held idle clip.
    output_dir: Where to write the tracking clips (basenames kept).
    input_fps: Clip sample rate (200 Hz crawl grids, 100 Hz upright grids).
    output_fps: Output rate; must equal the env control rate 1/step_dt (50 Hz).
    align_phase: Roll every clip onto one reference clip's gait phase.
    phase_ref: A built tracking clip to align to instead of the in-library reference.
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA unavailable, falling back to CPU (slow).")
    device = "cpu"

  files = sorted(glob.glob(os.path.join(input_dir, "*.npz")))
  if not files:
    raise FileNotFoundError(f"no .npz clips found in {input_dir}")
  os.makedirs(output_dir, exist_ok=True)

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot: Entity = scene["robot"]
  joint_indexes = robot.find_joints(list(JOINT_NAMES), preserve_order=True)[0]

  print(f"converting {len(files)} clip(s): {input_fps:g} Hz -> {output_fps:g} Hz")
  num_frames = 0
  logs: dict[str, dict[str, Any]] = {}
  strides: dict[str, tuple[np.ndarray, float]] = {}
  for f in files:
    motion = LibraryMotionLoader(
      motion_file=f,
      input_fps=int(input_fps),
      output_fps=int(output_fps),
      device=sim.device,
    )
    log = _replay_to_log(sim, scene, robot, joint_indexes, motion)
    # Carry the per-clip command labels through so the tracking command can select by twist.
    src = np.load(f)
    log["twist"] = np.asarray(
      src["twist"], dtype=np.float32
    )  # [vx, vy, wz] m/s,m/s,rad/s
    if "nominal_speed" in src:
      log["nominal_speed"] = np.asarray(src["nominal_speed"], dtype=np.float32)
    num_frames = log["joint_pos"].shape[0]
    name = os.path.basename(f)
    logs[name] = log
    strides[name] = (motion.stride_dxy, motion.stride_dpsi)
    pelvis0 = log["body_pos_w"][0, robot.body_names.index("pelvis"), :2]
    log["stride"] = _stride_map(pelvis0.astype(np.float64), *strides[name])
    stretch = f", time-stretched x{motion.time_stretch:.4f}" * (
      abs(motion.time_stretch - 1.0) > 1e-9
    )
    print(f"  {name} -> {num_frames} frames{stretch}")
  if align_phase:
    body_names = list(robot.body_names)
    feet = [
      body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    ref_profile = None
    if phase_ref is not None:
      ref_profile = _feet_profile(dict(np.load(phase_ref)), feet)
    _align_phase(logs, strides, feet, body_names.index("pelvis"), ref_profile)
  worst_seam, worst_interior = 0.0, 0.0
  for name, log in logs.items():
    seam, interior = _seam_error(log, output_fps)
    if seam > 3.0 * interior + 1e-3:
      raise RuntimeError(
        f"{name}: the clip does not loop -- body velocity is off by {seam:.3f} m/s at "
        f"the seam vs {interior:.3f} m/s elsewhere"
      )
    worst_seam, worst_interior = max(worst_seam, seam), max(worst_interior, interior)
    np.savez(os.path.join(output_dir, name), **log)
  print(
    f"  saved {len(logs)} clips to {output_dir}; velocity-vs-pose gap: seam "
    f"{worst_seam:.3f} m/s, interior {worst_interior:.3f} m/s"
  )

  # Optional static idle pose -> a held, zero-velocity zero-twist clip. Every clip in a library must
  # share the same frame count, so the idle is held for the gait clips' output length.
  idle_csv = os.path.join(input_dir, _IDLE_CSV)
  has_idle = os.path.exists(idle_csv)
  if has_idle:
    idle_qpos = np.loadtxt(idle_csv, delimiter=",")
    # Align the idle pose's heading (yaw about world z) to the gait clips, so a gait<->idle
    # transition is a posture change only, not a spurious ~50 deg yaw rotation. All clips of a
    # library share one heading; use the first clip as the reference.
    ref_yaw = _yaw_about_z(logs[os.path.basename(files[0])]["body_quat_w"][0, 0])
    idle_yaw = _yaw_about_z(idle_qpos[3:7])
    dyaw = ref_yaw - idle_yaw
    idle_qpos[3:7] = _apply_yaw_about_z(idle_qpos[3:7], dyaw)
    c, s = np.cos(dyaw), np.sin(dyaw)
    x, y = float(idle_qpos[0]), float(idle_qpos[1])
    idle_qpos[0], idle_qpos[1] = c * x - s * y, s * x + c * y
    print(
      f"  idle yaw-aligned to gait heading: {np.degrees(idle_yaw):+.1f} -> "
      f"{np.degrees(ref_yaw):+.1f} deg (dyaw {np.degrees(dyaw):+.1f})"
    )
    log = _idle_to_log(
      sim, scene, robot, joint_indexes, idle_qpos, num_frames, int(output_fps)
    )
    out_path = os.path.join(output_dir, idle_out)
    np.savez(out_path, **log)
    print(
      f"  {_IDLE_CSV} -> {idle_out} (static, held {num_frames} frames, v=0)  "
      f"saved to {out_path}"
    )
  else:
    print(
      f"  [note] no {_IDLE_CSV} in {input_dir}; skipping idle clip "
      "(zero-velocity commands will snap to the slowest crawl clip instead)."
    )

  n_out = len(files) + (1 if has_idle else 0)
  print(f"done: {n_out} clip(s) in {output_dir}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
