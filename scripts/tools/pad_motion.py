"""Pad a motion npz with a held start pose (frame 0 or a stand) and its last frame."""

import argparse
from pathlib import Path

import numpy as np
import torch

from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.crawling_common.library import load_idle_qpos
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
  quat_conjugate,
  quat_mul,
  yaw_quat,
)

BODY_KEYS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
ZERO_KEYS = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")


def _hold(arr: np.ndarray, n: int, zero: bool) -> np.ndarray:
  return np.repeat(np.zeros_like(arr) if zero else arr, n, axis=0)


class _Fk:
  """Body link states of the G1 tracking scene for given root and joint states."""

  def __init__(self, device: str):
    self.scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
    model = self.scene.compile()
    self.sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)
    self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
    self.robot = self.scene["robot"]
    self.device = device

  def __call__(self, root_pos, root_quat, root_lin, root_ang, joint_pos, joint_vel):
    out = {k: [] for k in BODY_KEYS}
    for i in range(root_pos.shape[0]):
      root = self.robot.data.default_root_state.clone()
      root[0] = torch.cat([root_pos[i], root_quat[i], root_lin[i], root_ang[i]])
      root[:, :2] += self.scene.env_origins[:, :2]
      self.robot.write_root_state_to_sim(root)
      self.robot.write_joint_state_to_sim(joint_pos[i : i + 1], joint_vel[i : i + 1])
      self.sim.forward()
      self.scene.update(self.sim.mj_model.opt.timestep)
      d = self.robot.data
      for k, v in zip(
        BODY_KEYS,
        (
          d.body_link_pos_w,
          d.body_link_quat_w,
          d.body_link_lin_vel_w,
          d.body_link_ang_vel_w,
        ),
        strict=True,
      ):
        out[k].append(v[0].cpu().numpy().copy())
    return {k: np.stack(v) for k, v in out.items()}


def _stand_start(data: dict, stand_csv: Path, hold_s: float, device: str) -> dict:
  """The stand held still for hold_s at frame 0's heading and xy."""
  n_hold = round(hold_s * float(data["fps"][0]))
  fk = _Fk(device)
  t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731

  _, stand_quat, stand_joints = load_idle_qpos(stand_csv)
  q_stand = t([[stand_joints[n] for n in fk.robot.joint_names]])
  pos0, quat0 = t(data["body_pos_w"][:1, 0]), t(data["body_quat_w"][:1, 0])
  # The stand's roll/pitch under frame 0's heading.
  sq = t(stand_quat)[None]
  quat_s = quat_mul(yaw_quat(quat0), quat_mul(quat_conjugate(yaw_quat(sq)), sq))

  zero3, zero_q = torch.zeros(1, 3, device=device), torch.zeros_like(q_stand)
  probe = fk(pos0, quat_s, zero3, zero3, q_stand, zero_q)
  # Place the stand so its lowest body sits at frame 0's lowest body height.
  pos_s = pos0.clone()
  pos_s[0, 2] += data["body_pos_w"][0, :, 2].min() - probe["body_pos_w"][0, :, 2].min()
  frame = fk(pos_s, quat_s, zero3, zero3, q_stand, zero_q)
  frame["joint_pos"] = q_stand.cpu().numpy()
  frame["joint_vel"] = zero_q.cpu().numpy()
  return {k: np.repeat(v, n_hold, axis=0) for k, v in frame.items()}


def pad_motion(
  src: Path,
  dst: Path,
  start_s: float,
  end_s: float,
  stand_csv: Path | None = None,
  device: str = "cuda:0",
) -> None:
  if dst.exists():
    raise FileExistsError(f"{dst} exists; refusing to overwrite.")
  data = dict(np.load(src))
  if "pad_frames" in data:
    raise ValueError(f"{src} is already padded.")
  fps = float(data["fps"][0])
  n_end = round(end_s * fps)
  n_frames = data["joint_pos"].shape[0]
  keys = ("joint_pos", "joint_vel") + BODY_KEYS
  if stand_csv is None:
    n_start = round(start_s * fps)
    start = {k: _hold(data[k][:1], n_start, k in ZERO_KEYS) for k in keys}
  else:
    start = _stand_start(data, stand_csv, start_s, device)
    n_start = start["joint_pos"].shape[0]
  for k in keys:
    end = _hold(data[k][-1:], n_end, k in ZERO_KEYS)
    data[k] = np.concatenate([start[k].astype(data[k].dtype), data[k], end])
  data["pad_frames"] = np.array([n_start, n_end])
  np.savez(dst, **data)
  print(f"{src} -> {dst}: {n_start} + {n_frames} + {n_end} frames at {fps:g} Hz")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument("--output-file", type=Path, default=None)
  parser.add_argument("--start-s", type=float, default=1.0)
  parser.add_argument("--end-s", type=float, default=2.0)
  parser.add_argument("--stand-csv", type=Path, default=None)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  out = args.output_file or args.motion_file.with_name(
    args.motion_file.stem + "_padded.npz"
  )
  pad_motion(
    args.motion_file,
    out,
    args.start_s,
    args.end_s,
    args.stand_csv,
    args.device,
  )


if __name__ == "__main__":
  main()
