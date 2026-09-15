"""Dump a TVLQR export's ``x_bar`` qpos as the CSV ``csv_to_npz`` consumes.

Building the clip from the export keeps it in lockstep with the gain schedule. The root
quaternion is written xyzw, which is the order ``csv_to_npz`` reorders from.
"""

from pathlib import Path

import numpy as np
import tyro

import mjlab
from mjlab.scripts.csv_to_npz import G1_JOINT_ORDER


def _tilt_deg(quat_wxyz: np.ndarray) -> float:
  """Angle between the body +z axis and world +z, which yaw leaves alone."""
  _, x, y, _ = quat_wxyz
  return float(np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))))


def main(
  export_file: str,
  output_file: str,
) -> None:
  """Write a TVLQR export's ``x_bar`` qpos to a CSV for ``csv_to_npz``.

  Args:
    export_file: Path to the TVLQR export npz (needs ``x_bar``, ``nq``, ``dof_names``).
    output_file: Path of the CSV to write.
  """
  export = np.load(export_file)
  x_bar = np.asarray(export["x_bar"], dtype=np.float64)
  nq = int(export["nq"])

  dof_names = [str(n).removeprefix("robot/") for n in export["dof_names"]]
  joint_names = tuple(dof_names[len(dof_names) - (nq - 7) :])
  if joint_names != G1_JOINT_ORDER:
    mismatched = [
      f"{i}: export={a!r} csv_to_npz={b!r}"
      for i, (a, b) in enumerate(zip(joint_names, G1_JOINT_ORDER, strict=False))
      if a != b
    ]
    raise ValueError(
      "the export's joint order does not match the order csv_to_npz reads columns 7: "
      "in, so the CSV would be scrambled:\n  " + "\n  ".join(mismatched)
    )

  qpos = x_bar[:, :nq]
  pos, quat_wxyz, joints = qpos[:, :3], qpos[:, 3:7], qpos[:, 7:]
  rows = np.concatenate([pos, quat_wxyz[:, [1, 2, 3, 0]], joints], axis=1)

  out = Path(output_file)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(out, rows, delimiter=",")

  fps = 1.0 / float(export["dt"])
  print(f"wrote {rows.shape[0]} rows x {rows.shape[1]} cols to {out}")
  print(f"frame 0 root tilt from vertical: {_tilt_deg(quat_wxyz[0]):.1f} deg")
  print(f"the export's dt is {float(export['dt'])}, so pass --input-fps {fps:g}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
