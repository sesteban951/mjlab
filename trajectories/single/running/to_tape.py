##
# Convert this directory's three-file V0 export (state.npz / input.npz / K.npz) into the
# motion + prior pair mjlab's tracking_prior task consumes:
#
#   running_motion.npz        the reference motion -- per-body world states from MuJoCo FK
#   running_motion_prior.npz  the control tape and its LQR schedule for the
#                             JointPositionActionWithPrior term (u / feedforward / gain /
#                             alpha / ref_qpos / ref_qvel), i.e. the `lam`-blended prior
#
# The export splits what mj-nlp's save_trajectory writes as one file, and renames keys:
#
#   state.npz:state_reference -> state, feedback_reference   xbar, the point K linearizes at
#   input.npz:input_plan      -> input, feedforward          ubar, the open-loop tape
#   K.npz:K                   -> gain
#   alpha                      = gain_scale (1.0 here) -- this arm carries no schedule
#
# The perturbed-model rollouts (`state[s]`, `input[s]`) are evaluation output, not the plan,
# so nothing here reads them.
#
# The motion comes from mj-nlp's to_mjlab_motion.convert (FK, resampling, velocities); the
# prior is written here instead of by its save_prior, because that helper indexes the control
# grid with `floor(t / input_dt)` on floats, which lands one step early on frames where the
# division rounds down (frame 29 of this export). The export's tape is already on the sim grid
# and the output grid is an exact integer subsample of it, so the tape is decimated by stride
# `round(sim_fps / output_fps)` with no interpolation: gains and references are held at their
# design nodes, never blended.
#
# Note the tape is emitted pre-clip. The source law clips u to the actuator box (u_lb/u_ub);
# mjlab's prior term applies its own clip to the entity's soft joint limits.
#
# Usage:
#   uv run python trajectories/single/running/to_tape.py
#   uv run python trajectories/single/running/to_tape.py --output-fps 50 --out-dir <dir>
##
import argparse
import importlib.util
import os
import tempfile

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MJ_NLP = os.getenv("TRAJOPT_ROOT_DIR", "/home/jason/humanoid_ws/mj-nlp")


def _load_converter(path):
  spec = importlib.util.spec_from_file_location("to_mjlab_motion", path)
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _load_export(export_dir):
  S = np.load(os.path.join(export_dir, "state.npz"))
  I = np.load(os.path.join(export_dir, "input.npz"))
  G = np.load(os.path.join(export_dir, "K.npz"))
  # P.npz is optional -- the Aug-28 export had no cost-to-go. Without it the tape still
  # drives the prior; it just cannot feed the LQR Lyapunov shaping reward.
  p_path = os.path.join(export_dir, "P.npz")
  P = np.load(p_path, allow_pickle=True) if os.path.exists(p_path) else None

  mode = str(I["actuator_mode"])
  if mode != "position":
    raise ValueError(f"actuator_mode is {mode!r}, not a joint-position target.")
  # The Aug-29 export dropped the sim_dt/gain_scale scalars the Aug-28 one carried, so
  # take the step off the time grid and fall back to an unscaled gain.
  sim_dt = float(S["sim_dt"]) if "sim_dt" in S else float(np.diff(S["time"])[0])
  if not np.allclose(np.diff(S["time"]), sim_dt):
    raise ValueError("state grid is not uniform at sim_dt.")

  xbar = np.asarray(S["state_reference"], dtype=float)  # (N+1, nq+nv)
  ubar = np.asarray(I["input_plan"], dtype=float)  # (N, nu), actuator order
  K = np.asarray(G["K"], dtype=float)  # (N, nu, 2*nv), rows actuator order
  if K.shape[:2] != ubar.shape or xbar.shape[0] != ubar.shape[0] + 1:
    raise ValueError(f"shape mismatch: x {xbar.shape}, u {ubar.shape}, K {K.shape}")
  if P is not None:
    # P is indexed like the state grid (one per state, so one more than the inputs), and
    # its rows and columns share K's tangent layout.
    ndx = K.shape[2]
    if P["P"].shape != (xbar.shape[0], ndx, ndx):
      raise ValueError(
        f"P is {P['P'].shape}, expected ({xbar.shape[0]}, {ndx}, {ndx})."
      )
    if not np.array_equal(P["tangent_start"], G["tangent_start"]) or not np.array_equal(
      P["tangent_stop"], G["tangent_stop"]
    ):
      raise ValueError("P and K disagree on the tangent-state layout.")
  return S, I, G, P, xbar, ubar, K, sim_dt


def pack(xbar, ubar, K, gain_scale, time, packed_npz):
  """Write the export as the one save_trajectory-shaped npz to_mjlab_motion reads."""
  np.savez(
    packed_npz,
    time=time,
    state=xbar,
    feedback_reference=xbar,
    input=ubar,
    feedforward=ubar,
    gain=K,
    alpha=np.full(len(ubar), gain_scale),
    actuator_mode=np.array("position"),
    # One control node per sim step, applied as a held command -- not a ramped spline.
    spline_type=np.array("zero"),
  )


def write_prior(
  prior_npz, m, xbar, ubar, K, P, gain_scale, stride, n_frames, output_fps, source
):
  """Decimate the tape onto the motion's frame grid, in the model's joint order."""
  act_joint = [int(m.actuator_trnid[i, 0]) for i in range(m.nu)]
  hinges = [j for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
  missing = [j for j in hinges if j not in act_joint]
  if missing:
    raise ValueError(
      f"joints {missing} are unactuated; the action term expects all of them"
    )
  perm = np.array(
    [act_joint.index(j) for j in hinges]
  )  # actuator order -> model joint order
  names = np.array(
    [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).split("/")[-1] for j in hinges]
  )

  k = np.arange(n_frames) * stride
  if k[-1] >= len(ubar):
    raise ValueError(f"frame grid runs past the tape: step {k[-1]} of {len(ubar)}")
  u = ubar[k][:, perm].astype(np.float32)
  out = {
    "fps": np.array([output_fps]),
    "u": u,
    "joint_names": names,
    "actuator_mode": np.array("position"),
    "source": np.array(source),
    "feedforward": u,
    "gain": K[k][:, perm, :].astype(np.float32),
    "alpha": np.full(n_frames, gain_scale, dtype=np.float32),
    "ref_qpos": xbar[k, : m.nq].astype(np.float32),
    "ref_qvel": xbar[k, m.nq : m.nq + m.nv].astype(np.float32),
  }
  if P is not None:
    # P's rows/columns are already in the tangent layout the mjlab reward builds its
    # error in, so only the time axis is decimated -- never the matrix itself.
    out["P"] = np.asarray(P["P"], dtype=float)[k].astype(np.float32)
    # Validity radius of the linear analysis, on the input grid. Not used by the shaping
    # reward, but it is the number that says where V stops being trustworthy.
    out["rho_sat"] = np.asarray(P["rho_sat"], dtype=float)[k].astype(np.float32)
  np.savez(prior_npz, **out)
  lo, hi = m.actuator_ctrlrange[perm, 0], m.actuator_ctrlrange[perm, 1]
  print(f"wrote {prior_npz}")
  print(
    f"  u {u.shape}  gain {out['gain'].shape}  stride {stride}  alpha {gain_scale:g}"
  )
  if "P" in out:
    print(f"  P {out['P'].shape}  rho_sat {out['rho_sat'].shape}")
  print(f"  ctrlrange margin: {float(np.minimum(hi - u, u - lo).min()):+.3e}")


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--export-dir", default=HERE)
  ap.add_argument("--out-dir", default=HERE)
  ap.add_argument("--name", default="running")
  ap.add_argument("--output-fps", type=float, default=50.0)
  ap.add_argument(
    "--model", default=os.path.join(MJ_NLP, "models/unitree_g1/g1_29dof.xml")
  )
  ap.add_argument(
    "--converter",
    default=os.path.join(MJ_NLP, "examples/g1_tracking_mpc/to_mjlab_motion.py"),
  )
  a = ap.parse_args()

  S, I, G, P, xbar, ubar, K, sim_dt = _load_export(a.export_dir)
  gain_scale = float(G["gain_scale"]) if "gain_scale" in G else 1.0
  sim_fps = 1.0 / sim_dt
  stride = round(sim_fps / a.output_fps)
  if abs(stride * a.output_fps - sim_fps) > 1e-9:
    raise ValueError(
      f"output fps {a.output_fps} does not divide the {sim_fps:g} Hz sim grid"
    )

  motion = os.path.join(a.out_dir, f"{a.name}_motion.npz")
  prior = os.path.join(a.out_dir, f"{a.name}_motion_prior.npz")
  convert = _load_converter(a.converter).convert
  with tempfile.TemporaryDirectory() as tmp:
    packed = os.path.join(tmp, "packed.npz")
    pack(xbar, ubar, K, gain_scale, S["time"], packed)
    convert(packed, motion, a.model, output_fps=a.output_fps, input_fps=sim_fps)

  n_frames = len(np.load(motion)["joint_pos"])
  write_prior(
    prior,
    mujoco.MjModel.from_xml_path(a.model),
    xbar,
    ubar,
    K,
    P,
    gain_scale,
    stride,
    n_frames,
    a.output_fps,
    f"{os.path.basename(os.path.normpath(a.export_dir))} export (state/input/K.npz)",
  )


if __name__ == "__main__":
  main()
