"""Convert an mj-nlp MPC/LQR solve into the TVLQR export schema mjlab's CLF rewards read.

``tracking.mdp.tvlqr.TvlqrGuidedJointPositionAction`` consumes the ``*_tvlqr.npz`` that mj-nlp's
``export_tvlqr.py`` writes. A raw solve npz carries the same four quantities under mj-nlp's own
names and is missing the metadata the loader needs, so this script renames and fills in the rest:

    state       -> x_bar   (n+1, nq+nv)   the nominal state trajectory
    input       -> u_bar   (n,   nu)      the feedforward command
    gains       -> K       (n, nu, 2*nv)  the time-varying gain schedule
    cost_to_go  -> P       (n+1, 2*nv, 2*nv)   the Riccati cost-to-go

``dof_names``, ``actuator_dof_index``, ``u_lb`` and ``u_ub`` are read off the MuJoCo model the
solve names in its own ``model`` field, so the row and column orders are the solve's, not a guess.
The loader re-maps K's rows to the action term's joint order by name at load time.

ALPHA IS A CONSTANT YOU PASS IN, and it is NOT the solve's ``alpha`` array. mjlab's ``alpha`` is
the per-step CLF decay rate in ``viol = max(V' - V + alpha_k V, 0)``; an mj-nlp solve's ``alpha``
is a per-SVD-mode scaling on the gain matrix (one column per ``svd_modes``, already applied in
``gains``), which is a different quantity entirely and is deliberately not carried over. Since
these solves ship no decay rate, pass ``--alpha-per-step``: a continuous rate of ``a`` 1/s is
``1 - exp(-a*dt)`` per step, so 0.5 1/s at dt=0.01 is 0.005. The default of 0 reduces the
condition to "V must not increase", which is the weakest useful form.

The export's ``dt`` must equal the env's ``physics_dt`` -- the loader refuses otherwise, since a
gain schedule applied at a rate it was not designed for is a different controller. This script
writes the SOLVE's own timestep, so the env must be configured to match it.
"""

from pathlib import Path

import mujoco
import numpy as np
import tyro

import mjlab

# mj-nlp solve key -> mjlab export key, for the four arrays that carry the design.
_RENAME = {
  "state": "x_bar",
  "input": "u_bar",
  "gains": "K",
  "cost_to_go": "P",
}


def main(
  solve_file: str,
  output_file: str,
  model_file: str | None = None,
  alpha_per_step: float = 0.0,
  cost_to_go_key: str = "cost_to_go",
  like_export: str | None = None,
) -> None:
  """Convert an mj-nlp solve npz to mjlab's TVLQR export schema.

  Args:
    solve_file: The mj-nlp solve npz (must carry state, input, gains, cost_to_go).
    output_file: Where to write the converted ``*_tvlqr.npz``.
    model_file: MuJoCo XML the solve was built against. Defaults to the path recorded in
      the solve's own ``model`` field.
    alpha_per_step: Constant per-step CLF decay rate written to ``alpha``. The default of
      0 asks only that V not increase; see the module docstring.
    cost_to_go_key: Which array in the solve becomes ``P``. Solves that ship more than one
      cost-to-go (e.g. a closed-loop Lyapunov P alongside ``cost_to_go_optimal``) select
      between them here.
    like_export: Borrow ``dof_names``, ``actuator_dof_index``, ``u_lb`` and ``u_ub`` from
      an existing export instead of the XML, for when the solve's model is not on this
      machine. Shapes are checked against it, so a solve from a different model is refused.
  """
  solve = np.load(solve_file, allow_pickle=True)
  rename = dict(_RENAME)
  if cost_to_go_key != "cost_to_go":
    del rename["cost_to_go"]
    rename[cost_to_go_key] = "P"
  missing = [k for k in rename if k not in solve.files]
  if missing:
    raise KeyError(f"{solve_file} has no {missing}; it is not an mj-nlp LQR solve.")

  if like_export is not None:
    ref = np.load(like_export)
    nq, nv, nu = int(ref["nq"]), int(ref["nv"]), int(ref["nu"])
    xml = str(ref["model"])
  else:
    xml = model_file or str(solve["model"])
    model = mujoco.MjModel.from_xml_path(xml)
    nq, nv, nu = model.nq, model.nv, model.nu

  out = {k: np.asarray(solve[src], dtype=np.float64) for src, k in rename.items()}
  n = out["u_bar"].shape[0]

  # Shapes the loader relies on. Checked here so a mismatched solve fails with a sentence
  # rather than as a silent misread of K's columns at training time.
  expect = {
    "x_bar": (n + 1, nq + nv),
    "u_bar": (n, nu),
    "K": (n, nu, 2 * nv),
    "P": (n + 1, 2 * nv, 2 * nv),
  }
  for key, want in expect.items():
    if out[key].shape != want:
      raise ValueError(
        f"{key} is {out[key].shape}, expected {want} for a model with "
        f"nq={nq}, nv={nv}, nu={nu} and n={n} steps."
      )

  if like_export is not None:
    # Model-derived metadata borrowed wholesale; only P differs in the P-swap case.
    dof_names = list(ref["dof_names"])
    actuator_dof_index = ref["actuator_dof_index"]
    u_lb, u_ub = ref["u_lb"].copy(), ref["u_ub"].copy()
    if len(dof_names) != nv or len(u_lb) != nu:
      raise ValueError(
        f"{like_export} does not describe a model with nv={nv}, nu={nu}."
      )
  else:
    # The tangent basis: the free joint's 6 dofs, then one entry per hinge, in dof order.
    # The loader strips the prefix and checks dof_names[6:nv] against the entity's joints.
    joint_names = [
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)
    ]
    dof_names = []
    for j, name in enumerate(joint_names):
      width = nv - model.jnt_dofadr[j] if j + 1 == model.njnt else 0
      width = (
        model.jnt_dofadr[j + 1] - model.jnt_dofadr[j] if j + 1 < model.njnt else width
      )
      dof_names.extend([f"{name}[{i}]" for i in range(width)] if width > 1 else [name])

    # Which dof each actuator drives, in the solve's actuator (and so u_bar/K row) order.
    actuator_dof_index = np.array(
      [model.jnt_dofadr[model.actuator_trnid[a, 0]] for a in range(nu)], dtype=np.int64
    )
    u_lb = model.actuator_ctrlrange[:, 0].copy()
    u_ub = model.actuator_ctrlrange[:, 1].copy()

  dt = (
    float(np.diff(np.asarray(solve["time"])).mean()) if "time" in solve.files else 0.0
  )
  if dt <= 0.0:
    raise ValueError(f"{solve_file} has no usable 'time' array to derive dt from.")

  np.savez(
    output_file,
    **out,
    alpha=np.full(n, float(alpha_per_step)),
    u_lb=u_lb,
    u_ub=u_ub,
    dof_names=np.array(dof_names),
    actuator_dof_index=actuator_dof_index,
    nq=nq,
    nv=nv,
    nu=nu,
    ndx=2 * nv,
    n_steps=n,
    dt=dt,
    # Not periodic in general, and the loader only reads it; one full pass is the honest value.
    stride_len=n,
    model=xml,
    source_run=str(Path(solve_file).name),
    actuator_mode=str(solve["actuator_mode"])
    if "actuator_mode" in solve.files
    else "position",
    cost_to_go_kind=str(solve["cost_to_go_kind"])
    if "cost_to_go_kind" in solve.files and cost_to_go_key == "cost_to_go"
    else cost_to_go_key,
  )
  borrowed = f", metadata from {like_export}" if like_export else ""
  print(
    f"[INFO] wrote {output_file}\n"
    f"       n={n} steps at dt={dt} ({n * dt:.3f} s), nq={nq} nv={nv} nu={nu}\n"
    f"       alpha = {alpha_per_step} per step (constant)\n"
    f"       P from '{cost_to_go_key}'{borrowed}\n"
    f"       the env's physics timestep MUST be {dt} or the loader will refuse it."
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
