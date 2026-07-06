"""Convert the mj-nlp crawl gait library into mjlab tracking-motion npz files.

The library clips (``trajectories/library/crawl_vp0_*.npz``) store per-frame full MuJoCo state
``state = [qpos(36) | qvel(35)]`` for the G1. mjlab's tracking ``MotionLoader`` instead expects
per-body world kinematics (``joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w,
body_ang_vel_w``). This script FK-replays each clip through the mjlab G1 scene to produce that
schema, reusing csv_to_npz's finite-difference velocity recompute + FK logging, and writes one
tracking npz per clip to disk (no Weights & Biases). Run once:

  uv run python -m mjlab.scripts.library_to_npz

Regenerate only if the env's control rate changes (``output_fps`` must equal ``1/step_dt`` =
``1/(decimation*timestep)`` = 50 Hz for the tracking/crawling env).
"""

import glob
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
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

_REPO_ROOT = Path(mjlab.MJLAB_SRC_PATH).parent.parent

# The 29 actuated G1 joints, in the order the library `state` stores them (same as csv_to_npz).
JOINT_NAMES = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
  "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
  "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

_LOG_KEYS = (
  "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
)


class LibraryMotionLoader(MotionLoader):
  """MotionLoader variant that sources base + joint states from a library ``[qpos|qvel]`` npz.

  Overrides only ``_load_motion``: reads the ``state`` array (qpos[0:3]=base pos, qpos[3:7]=base
  quat *already wxyz*, qpos[7:36]=29 joints) and discards the library qvel (the free-joint angular
  qvel is body-local; the base class recomputes world-frame velocities by finite difference)."""

  def _load_motion(self):
    data = np.load(self.motion_file)
    state = torch.from_numpy(np.asarray(data["state"], dtype=np.float32)).to(self.device)
    self.motion_base_poss_input = state[:, 0:3]
    self.motion_base_rots_input = state[:, 3:7]  # already wxyz -- no reorder (unlike the CSV path)
    self.motion_dof_poss_input = state[:, 7:36]  # 29 actuated joints
    self.input_frames = state.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt


def _replay_to_log(sim, scene, robot: Entity, joint_indexes, motion: MotionLoader) -> dict[str, Any]:
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
    frames["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
    frames["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())

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


def main(
  input_dir: str = str(_REPO_ROOT / "trajectories" / "library"),
  output_dir: str = str(_REPO_ROOT / "trajectories" / "library_tracking"),
  input_fps: float = 200.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
):
  """Convert every ``*.npz`` in ``input_dir`` (mj-nlp ``[qpos|qvel]`` schema) to tracking npz.

  Args:
    input_dir: Folder of library clips (e.g. ``trajectories/library``).
    output_dir: Where to write the tracking-format clips (basename preserved).
    input_fps: Sample rate of the library clips (200 Hz for the mj-nlp crawl library).
    output_fps: Output rate; MUST equal the env control rate 1/step_dt (50 Hz).
    device: Torch/sim device.
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
  for f in files:
    motion = LibraryMotionLoader(
      motion_file=f, input_fps=int(input_fps), output_fps=int(output_fps), device=sim.device
    )
    log = _replay_to_log(sim, scene, robot, joint_indexes, motion)
    out_path = os.path.join(output_dir, os.path.basename(f))
    np.savez(out_path, **log)
    print(
      f"  {os.path.basename(f)} -> joint_pos {log['joint_pos'].shape}, "
      f"body_pos_w {log['body_pos_w'].shape}  saved to {out_path}"
    )
  print(f"done: {len(files)} clip(s) in {output_dir}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
