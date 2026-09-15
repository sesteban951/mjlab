##
# Convert a single-file mj-nlp LQR export into the motion + prior pair mjlab's
# tracking_prior task consumes:
#
#   <name>_motion.npz        the reference motion -- per-body world states from FK
#   <name>_motion_prior.npz  the control tape and its LQR schedule for the
#                            JointPositionActionWithPrior term (u / feedforward / gain /
#                            alpha / ref_qpos / ref_qvel / P)
#
# Unlike running/, which shipped a three-file split (state/input/K.npz), these exports are
# already the one save_trajectory-shaped npz to_mjlab_motion.convert reads -- `state`
# equals `feedback_reference` and `input` equals `feedforward` bitwise. Only the schedule
# keys are named differently (`gain`/`gains` -> gain, `cost_to_go` -> P).
# lqr_mpc.py v3 exports map feedback_reference/input_nominal/gains instead and carry no P.
#
# The tape is written on the export's OWN sim grid by default. A tape solved at 100 Hz and
# replayed as a 50 Hz zero-order hold is a different control signal -- up to 0.62 rad of
# target error on the sideroll clip, enough to lose it open loop on the design plant --
# so decimating now takes --allow-decimation and says what it costs.
#
# The clips are solved on g1_29dof_locked_wrists.xml, which pins the six wrist joints with
# equality constraints but keeps nq/nv/nu at 36/35/29 and the joint order of g1_29dof.xml.
# FK is a function of qpos alone, so the motion is generated against g1_29dof.xml; the
# orderings are asserted rather than assumed.
#
# These exports carry no `rho_sat`, so the prior omits it. That only rules out the
# lqr_lyapunov_shaping clip_to_rho_sat path; the saturating `v_half` form does not use it.
#
# Export the file from mj-nlp, not a copy of it: two npz under the same name in the two
# repos drifted to different r_scale and different gains once already, and the tape looked
# fine either way. The provenance stamped into the tape (`export_path`, `export_sha256`,
# `r_scale`, `gain_max`, `law`) is what play_prior prints so that cannot repeat.
#
# Usage:
#   uv run python scripts/tools/lqr_export_to_tape.py \
#     --export $TRAJOPT_ROOT_DIR/examples/full_run/g1_tracking_mpc/g1_lqr_sideroll_robust_oldK.npz \
#     --out-dir trajectories/single/mpc_sideroll --name sideroll
##
import argparse
import hashlib
import importlib.util
import json
import os

import mujoco
import numpy as np

MJ_NLP = os.getenv("TRAJOPT_ROOT_DIR", "/home/jason/humanoid_ws/mj-nlp")


def _load_converter(path):
  spec = importlib.util.spec_from_file_location("to_mjlab_motion", path)
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _equality_locked_joints(solve):
  """Joint ids the solve model pins with an mjEQ_JOINT constraint, i.e. its dead inputs."""
  locked = []
  for i in range(solve.neq):
    if solve.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
      continue
    jid = int(solve.eq_obj1id[i])
    if solve.eq_obj2id[i] not in (-1, jid):
      continue  # a joint-to-joint coupling still leaves the input live
    locked.append(jid)
  return sorted(set(locked))


def _neutralize_locked(K, ubar, xbar, m, locked):
  """Zero the gain rows and aim the feedforward at the reference for the pinned joints.

  Opt-in, and lossy. An mjEQ_JOINT constraint is a soft constraint, not a weld: at the
  default solref 0.02 / solimp dmin 0.9 the sideroll wrists still travel 0.117 rad under
  the lock, which is exactly their reference, so the actuator keeps real authority
  through the constraint and the solve used it. Dropping those commands costs the clip
  844 -> 657 frames even with the wrists still locked.

  It is only the right move when the replay model does not pin the joint at all AND you
  would rather the wrist hold its reference than chase a 1.77 rad target the unlocked
  plant will actually follow. Matching the solve model is the better fix.
  """
  rows = [i for i in range(m.nu) if int(m.actuator_trnid[i, 0]) in locked]
  K, ubar = K.copy(), ubar.copy()
  K[:, rows, :] = 0.0
  for i in rows:
    ubar[:, i] = xbar[: len(ubar), int(m.jnt_qposadr[int(m.actuator_trnid[i, 0])])]
  return K, ubar, rows


def _pick(z, *names):
  """First of `names` the export actually carries; mj-nlp spells the gain both ways."""
  for name in names:
    if name in z.files:
      return name
  raise KeyError(f"export carries none of {names}; it has {sorted(z.files)}")


def _is_mpc_export(z):
  """The lqr_mpc.py v3 format: the plan lives in *_nominal, `state` is a rollout."""
  return "feedforward" not in z.files and "input_nominal" in z.files


def _load_export(path):
  z = np.load(path, allow_pickle=True)
  mode = str(z["actuator_mode"])
  if mode != "position":
    raise ValueError(f"actuator_mode is {mode!r}, not a joint-position target.")

  time = np.asarray(z["time"], dtype=float)
  sim_dt = float(np.diff(time)[0])
  if not np.allclose(np.diff(time), sim_dt):
    raise ValueError("state grid is not uniform at sim_dt.")

  mpc = _is_mpc_export(z)
  if mpc:
    # u = input + gains (x - feedback_reference); `state` is the closed-loop rollout.
    xbar = np.asarray(z["feedback_reference"], dtype=float)
    ubar = np.asarray(z["input"], dtype=float)
    if not np.array_equal(xbar, z["state_nominal"]):
      raise ValueError("`feedback_reference` is not the `state_nominal` plan.")
  else:
    xbar = np.asarray(z["state"], dtype=float)  # (N+1, nq+nv)
    ubar = np.asarray(z["input"], dtype=float)  # (N, nu), actuator order
    if not np.array_equal(xbar, z["feedback_reference"]):
      raise ValueError("`state` is not the point the gains linearize at.")
    if not np.array_equal(ubar, z["feedforward"]):
      raise ValueError("`input` is not the open-loop tape.")
  K = np.asarray(
    z[_pick(z, "gains", "gain")], dtype=float
  )  # (N, nu, 2*nv), actuator rows
  P = np.asarray(z["cost_to_go"], dtype=float) if "cost_to_go" in z.files else None

  # A law re-aimed at another nominal measures its error against `gain_reference`, which
  # the tape format cannot carry alongside the flown tape: one ref_qpos, one meaning.
  if "gain_reference" in z.files and not np.array_equal(
    xbar, np.asarray(z["gain_reference"], dtype=float)
  ):
    raise ValueError(
      "`gain_reference` differs from `state`: this law feeds back about a different "
      "trajectory than the one it flies, which the single ref_qpos/ref_qvel in the tape "
      "cannot express. Re-aim the gain at its own tape before exporting."
    )
  if K.shape[:2] != ubar.shape or xbar.shape[0] != ubar.shape[0] + 1:
    raise ValueError(f"shape mismatch: x {xbar.shape}, u {ubar.shape}, K {K.shape}")
  ndx = K.shape[2]
  if P is not None and P.shape != (xbar.shape[0], ndx, ndx):
    raise ValueError(f"P is {P.shape}, expected ({xbar.shape[0]}, {ndx}, {ndx}).")

  # alpha: the export's own per-step gain scale when it kept one, else unscaled.
  if "gain_scale" in z.files:
    alpha = np.broadcast_to(
      np.asarray(z["gain_scale"], dtype=float).reshape(-1), (len(ubar),)
    ).astype(np.float32)
  else:
    alpha = np.ones(len(ubar), dtype=np.float32)

  if mpc:
    r_sched = np.asarray(z[_pick(z, "r_scale_sched", "r_scale")], dtype=float).reshape(
      -1
    )
    r_scale = float(r_sched[0]) if np.ptp(r_sched) == 0 else float("nan")
    kind = "alpha" if "svd_modes" in z.files else "theta"
    law = f"LQR x {kind}, risk {z['risk']}, R {r_sched.min():g}..{r_sched.max():g}"
  else:
    r_scale = float(z["r_scale"]) if "r_scale" in z.files else float("nan")
    law = str(z["law"]) if "law" in z.files else "(unlabelled)"
  prov = {
    "export_path": os.path.abspath(path),
    "export_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
    "export_format": "lqr_mpc" if mpc else "lqr",
    "spline_type": str(z["spline_type"]) if "spline_type" in z.files else "?",
    "node_dt": float(z["node_dt"]) if "node_dt" in z.files else float("nan"),
    "sim_dt": sim_dt,
    "r_scale": r_scale,
    "gain_max": float(np.abs(K).max()),
    "law": law,
    # The plant the tape was solved on, so a replay can reproduce it without anyone
    # retyping the numbers. Older exports predate the key.
    "dynamics_config": json.loads(z["dynamics_config"].item())
    if "dynamics_config" in z.files
    else None,
  }
  return xbar, ubar, K, P, alpha, sim_dt, prov


def write_prior(
  prior_npz, m, xbar, ubar, K, P, alpha, stride, n_frames, output_fps, source, prov
):
  """Decimate the tape onto the motion's frame grid, in the model's joint order."""
  act_joint = [int(m.actuator_trnid[i, 0]) for i in range(m.nu)]
  hinges = [j for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
  missing = [j for j in hinges if j not in act_joint]
  if missing:
    raise ValueError(
      f"joints {missing} are unactuated; the action term expects all of them"
    )
  perm = np.array([act_joint.index(j) for j in hinges])
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
    "alpha": alpha[k],
    "provenance": np.array(json.dumps(prov)),
    "ref_qpos": xbar[k, : m.nq].astype(np.float32),
    "ref_qvel": xbar[k, m.nq : m.nq + m.nv].astype(np.float32),
  }
  # P's rows/columns are already in the tangent layout the mjlab reward builds its
  # error in, so only the time axis is decimated -- never the matrix itself.
  if P is not None:
    out["P"] = P[k].astype(np.float32)
  np.savez(prior_npz, **out)
  lo, hi = m.actuator_ctrlrange[perm, 0], m.actuator_ctrlrange[perm, 1]
  print(f"wrote {prior_npz}")
  p_shape = out["P"].shape if "P" in out else "absent"
  print(f"  u {u.shape}  gain {out['gain'].shape}  P {p_shape}  stride {stride}")
  if P is None:
    print(
      "  [WARN] no cost_to_go in the export: the LQR value/decrease rewards need P, so "
      "training must run MJLAB_ABLATION=control or action_prior_exp"
    )
  print(f"  ctrlrange margin: {float(np.minimum(hi - u, u - lo).min()):+.3e}")
  print(
    f"  control {output_fps:g} Hz from a {1 / prov['sim_dt']:g} Hz "
    f"{prov['spline_type']} tape on {prov['node_dt']:g} s nodes"
  )
  print(
    f"  law: {prov['law']}  (R x {prov['r_scale']:g}, |K|max {prov['gain_max']:.1f})"
  )
  print(f"  sha256 {prov['export_sha256'][:16]}  {prov['export_path']}")


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--export", required=True, help="the single-file LQR export")
  ap.add_argument("--out-dir", default=None, help="default: the export's directory")
  ap.add_argument("--name", required=True, help="output prefix, e.g. sideroll")
  ap.add_argument(
    "--output-fps",
    type=float,
    default=None,
    help="default: the export's own sim rate, so the control grid is the "
    "one the gains were designed on",
  )
  ap.add_argument(
    "--allow-decimation",
    action="store_true",
    help="write a tape slower than the export's sim grid anyway",
  )
  ap.add_argument(
    "--neutralize-locked-commands",
    action="store_true",
    help="zero the gain rows and hold the reference for joints the solve model pins by "
    "equality. Lossy: the lock is compliant, so those commands do real work",
  )
  ap.add_argument(
    "--model", default=os.path.join(MJ_NLP, "models/unitree_g1/g1_29dof.xml")
  )
  ap.add_argument(
    "--solve-model",
    default=os.path.join(MJ_NLP, "models/unitree_g1/g1_29dof_locked_wrists.xml"),
  )
  ap.add_argument(
    "--converter",
    default=os.path.join(MJ_NLP, "examples/g1_tracking_mpc/to_mjlab_motion.py"),
  )
  a = ap.parse_args()
  out_dir = a.out_dir or os.path.dirname(os.path.abspath(a.export))

  xbar, ubar, K, P, alpha, sim_dt, prov = _load_export(a.export)
  sim_fps = 1.0 / sim_dt
  output_fps = sim_fps if a.output_fps is None else a.output_fps
  stride = round(sim_fps / output_fps)
  if abs(stride * output_fps - sim_fps) > 1e-9:
    raise ValueError(
      f"output fps {output_fps} does not divide the {sim_fps:g} Hz sim grid"
    )
  if stride > 1 and not a.allow_decimation:
    raise ValueError(
      f"--output-fps {output_fps:g} decimates the {sim_fps:g} Hz tape by {stride}, and "
      f"mjlab holds each sample for the whole control step. A {prov['spline_type']} tape "
      f"is not a staircase: the held signal differs from the one solved for, which on the "
      f"sideroll clip loses the robot open loop on its OWN plant. Export at {sim_fps:g} "
      f"and run play_prior --sim-timestep {sim_dt:g}, or pass --allow-decimation."
    )

  m = mujoco.MjModel.from_xml_path(a.model)
  solve = mujoco.MjModel.from_xml_path(a.solve_model)
  _assert_same_ordering(m, solve)

  locked = _equality_locked_joints(solve)
  prov["locked_joints"] = [
    mujoco.mj_id2name(solve, mujoco.mjtObj.mjOBJ_JOINT, j) for j in locked
  ]
  neutralize = a.neutralize_locked_commands
  prov["locked_commands"] = "neutralized" if neutralize else "kept"
  if locked and neutralize:
    K, ubar, rows = _neutralize_locked(K, ubar, xbar, m, locked)
    print(
      f"[WARN] neutralized {len(rows)} equality-pinned joints (gain row -> 0, "
      f"feedforward -> reference). The lock is compliant, so this drops authority the "
      f"solve used: {', '.join(prov['locked_joints'])}"
    )
  elif locked:
    print(
      f"{len(locked)} joints are pinned by equality in the solve model and are NOT "
      f"pinned in {os.path.basename(a.model)}: {', '.join(prov['locked_joints'])}.\n"
      f"  Their commands are live (the lock is a soft constraint), so the replay model "
      f"must pin them too or the tape asks them for motion the lock was absorbing."
    )

  motion = os.path.join(out_dir, f"{a.name}_motion.npz")
  prior = os.path.join(out_dir, f"{a.name}_motion_prior.npz")
  convert = _load_converter(a.converter).convert
  src = a.export
  if prov["export_format"] == "lqr_mpc":
    # The motion is the plan the gains linearize at, not the export's rollout.
    src = os.path.join(out_dir, f".{a.name}_plan.npz")
    np.savez(src, state=xbar, time=np.arange(len(xbar)) * sim_dt)
  convert(src, motion, a.model, output_fps=output_fps, input_fps=sim_fps)
  if src != a.export:
    os.remove(src)

  n_frames = len(np.load(motion)["joint_pos"])
  write_prior(
    prior,
    m,
    xbar,
    ubar,
    K,
    P,
    alpha,
    stride,
    n_frames,
    output_fps,
    f"{os.path.basename(a.export)} (single-file LQR export)",
    prov,
  )


def _assert_same_ordering(m, solve):
  """The tape's rows are in the solve model's order; FK runs on `m`. They must agree."""
  if (m.nq, m.nv, m.nu, m.njnt) != (solve.nq, solve.nv, solve.nu, solve.njnt):
    raise ValueError(
      f"model is {(m.nq, m.nv, m.nu, m.njnt)} but the solve model is "
      f"{(solve.nq, solve.nv, solve.nu, solve.njnt)} (nq, nv, nu, njnt)"
    )
  for obj, n in (
    (mujoco.mjtObj.mjOBJ_JOINT, m.njnt),
    (mujoco.mjtObj.mjOBJ_ACTUATOR, m.nu),
  ):
    a = [mujoco.mj_id2name(m, obj, i) for i in range(n)]
    b = [mujoco.mj_id2name(solve, obj, i) for i in range(n)]
    if a != b:
      raise ValueError(f"{obj} ordering differs between the FK and solve models")


if __name__ == "__main__":
  main()
