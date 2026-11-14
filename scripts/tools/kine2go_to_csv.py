"""Convert a Kine2Go Go2 reference (``motion.npy``) into an mjlab motion CSV.

Kine2Go (https://huggingface.co/datasets/MIMUW-Robotics/kine2go) ships dog mocap
already retargeted onto the Unitree Go2. TrackerLab SuperMimic trains its Go2
gait task on ``ai4_dog_trot_00`` at revision 7f03a3fb7f6b. Each clip's
``motion.npy`` is a (frames, 61) float array:

  0:6    raw root DOFs
  6:18   joint positions grouped by type (4 hips, 4 thighs, 4 calves), each in
         FL, FR, RL, RR order
  18:36  DOF velocities
  36:48  foot positions in FL, RL, FR, RR order
  48:51  root position
  51:55  root quaternion (wxyz)
  55:61  root linear and angular velocity

The output CSV has one row per frame: root xyz, root quaternion (xyzw), then the
12 joints in the Go2 model's joint order, which is the format ``csv_to_npz``
reads. Before writing, the script checks the mapping by running forward
kinematics on mjlab's Go2 model and comparing against the stored foot positions.
Kine2Go is 60 Hz, so pass ``--input-fps 60`` to ``csv_to_npz``.

Clips are licensed by their source (``clip.json``); AI4Animation clips are
CC BY-NC 4.0.

Example:
  uv run python scripts/tools/kine2go_to_csv.py \\
    --input-file motion.npy --output-file ai4_dog_trot_00.csv
"""

from pathlib import Path

import mujoco
import numpy as np
import tyro

from mjlab.asset_zoo.robots.unitree_go2.go2_constants import GO2_XML

_LEGS = ("FL", "FR", "RL", "RR")
_RAW_JOINT_ORDER = tuple(
  f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf") for leg in _LEGS
)
_RAW_FOOT_ORDER = ("FL", "RL", "FR", "RR")
# Foot centre in the calf frame (the Go2 MJCF's foot geom position).
_FOOT_OFFSET = np.array([0.0, 0.0, -0.213])


def convert(raw: np.ndarray, model: mujoco.MjModel) -> tuple[np.ndarray, float]:
  """Return the CSV rows and the largest foot-position FK error in metres."""
  if raw.ndim != 2 or raw.shape[1] != 61:
    raise ValueError(f"Expected a (frames, 61) Kine2Go array, got {raw.shape}")
  quat_wxyz = raw[:, 51:55]
  if not np.allclose(np.linalg.norm(quat_wxyz, axis=1), 1.0, atol=1e-4):
    raise ValueError("Root quaternions are not unit length")

  joint_names = [model.joint(i).name for i in range(1, model.njnt)]
  joint_cols = [6 + _RAW_JOINT_ORDER.index(name) for name in joint_names]
  joint_pos = raw[:, joint_cols]

  data = mujoco.MjData(model)
  calf_ids = [model.body(f"{leg}_calf").id for leg in _RAW_FOOT_ORDER]
  max_err = 0.0
  for t in range(raw.shape[0]):
    data.qpos[:3] = raw[t, 48:51]
    data.qpos[3:7] = quat_wxyz[t]
    data.qpos[7:] = joint_pos[t]
    mujoco.mj_kinematics(model, data)
    feet = raw[t, 36:48].reshape(4, 3)
    for k, body_id in enumerate(calf_ids):
      foot = data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ _FOOT_OFFSET
      max_err = max(max_err, float(np.linalg.norm(foot - feet[k])))

  rows = np.concatenate((raw[:, 48:51], quat_wxyz[:, [1, 2, 3, 0]], joint_pos), axis=1)
  return rows, max_err


def main(input_file: str, output_file: str, fk_tolerance: float = 0.005) -> None:
  """Convert a Kine2Go ``motion.npy`` into an mjlab motion CSV.

  Args:
    input_file: Path to a Kine2Go clip's ``motion.npy``.
    output_file: Path of the CSV to write.
    fk_tolerance: Largest allowed foot-position error (m) between forward
      kinematics on the mjlab Go2 model and the clip's stored foot positions.
  """
  raw = np.load(input_file, allow_pickle=False).astype(np.float64)
  model = mujoco.MjModel.from_xml_path(str(GO2_XML))
  rows, max_err = convert(raw, model)
  print(f"[INFO] {rows.shape[0]} frames, max foot FK error {max_err * 1e3:.2f} mm")
  if max_err > fk_tolerance:
    raise ValueError(
      f"Foot FK error {max_err:.4f} m exceeds {fk_tolerance} m; the joint or "
      "foot layout does not match this Go2 model."
    )
  Path(output_file).parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(output_file, rows, delimiter=",", fmt="%.6f")
  print(f"[INFO] Wrote {output_file}")


if __name__ == "__main__":
  tyro.cli(main)
