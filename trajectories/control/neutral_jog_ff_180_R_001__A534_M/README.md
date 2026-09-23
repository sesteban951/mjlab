# G1 periodic running gait + time-varying LQR

A dynamically-feasible periodic running gait for the Unitree G1 (29 dof), and the finite-horizon
time-varying LQR designed around it. One `.npz`: nominal trajectory, gain schedule `K_k`,
cost-to-go `P_k`, time axis, plus the metadata needed to interpret them.

Produced by `examples/g1_mimic_periodic_v0/export_tvlqr.py` from a solved run. Every array is
copied out of that run unchanged — nothing here is recomputed.

```python
import numpy as np, mujoco

d = np.load("g1_mimic_periodic_v0_run_fwd_<stamp>_tvlqr.npz", allow_pickle=True)
t, x_bar, u_bar, K, P = d["time"], d["x_bar"], d["u_bar"], d["K"], d["P"]

# THE TWO PLANT OVERRIDES -- see "Reproducing the plant". Without them the law does not work.
m = mujoco.MjModel.from_xml_path(str(d["model"]))  # relative to the repo root
m.opt.timestep = float(d["sim_timestep"])  # 5 ms, not the XML's 2 ms
m.opt.integrator = {"euler": 0, "rk4": 1, "implicit": 2, "implicitfast": 3}[
  str(d["sim_integrator"])
]  # not the XML's euler
```

## Contents

| key | shape | units | what |
|---|---|---|---|
| `time` | (1721,) | s | uniform, `dt` = 5 ms (200 Hz) |
| `x_bar` | (1721, 71) | rad, m, rad/s, m/s | the nominal state trajectory |
| `u_bar` | (1720, 29) | **rad** | the nominal command — joint *position targets* |
| `K` | (1720, 29, 70) | — | gain schedule, `u = u_bar + K δx` |
| `P` | (1721, 70, 70) | — | LQR cost-to-go, `V_k(δx) = δxᵀ P_k δx` |

`nq`=36, `nv`=35, `nu`=29, `ndx`=70, `n_steps`=1720. Metadata: `actuator_names`, `dof_names`,
`actuator_dof_index`, `u_lb`/`u_ub`, `servo_kp`/`servo_kd`/`tau_max`, `stride_len`, `repeat`,
`twist_arc`, `periodicity`, `model`, `source_run`, `gain_scale`, `lqr_r_scale`, `lqr_reg`.

## The control law

```python
u = np.clip(u_bar[k] + K[k] @ state_diff(x, x_bar[k], m), d["u_lb"], d["u_ub"])
```

**`state_diff` is not subtraction.** The state lives on a manifold — the base orientation is a
unit quaternion — so the 70-dim error is a *tangent* vector:

```python
def state_diff(xa, xb, model):  # -> (70,)
  dq = np.zeros(model.nv)
  mujoco.mj_differentiatePos(model, dq, 1.0, xb[: model.nq], xa[: model.nq])
  return np.concatenate([dq, xa[model.nq :] - xb[model.nq :]])
```

`x - x_bar` would be 71-dim, wrong in the 4 quaternion components, and silently wrong everywhere
downstream. This is the single most common way to misuse this dataset.

### Layouts

`x` (71) is `[base_pos(3), base_quat_wxyz(4), joints(29) | base_linvel(3), base_angvel(3), joint_vel(29)]`.

`δx` (70) is `[base_pos(3), base_rotvec(3), joints(29) | base_linvel(3), base_angvel(3), joint_vel(29)]`
— the quaternion's 4 components collapse to a 3-vector.

Frames follow MuJoCo's free-joint convention: **base linear velocity is world-frame, base angular
velocity is body-frame**, and the rotation tangent is body-frame. (Confirmed independently — the
SE(2) tiling that built this trajectory rotates exactly rows `0:3` and `35:38` and leaves the rest
alone.)

## Two ordering traps

**1. `K`'s rows and columns are in different orders.** Rows are *actuators*, in the XML's
declaration order. Columns are the *tangent*, in dof order. On this robot those disagree:

```
dof order    (columns 6:35): left leg(6), right leg(6), waist(3), left arm(7), right arm(7)
actuator order     (rows):   left arm(5), right arm(5), then legs and waist interleaved
```

`actuator_dof_index[i]` is the dof driven by actuator `i` — `[21, 22, 23, 24, 25, 28, ...]`, i.e.
row 0 is `left_shoulder_pitch` while column 0 is the base. To view `K` with both axes aligned:

```python
order = np.argsort(d["actuator_dof_index"])  # actuators, sorted into dof order
K_aligned = K[:, order, :]  # rows now match columns
u = np.empty(29)
u[order] = u_sorted  # ... and to undo it on a command
```

**2. `u` is a joint position target, not a torque.** The actuators are position servos:

```
tau = clamp( kp * (u - q) - kd * qd,  ±tau_max )
```

with `servo_kp` ∈ [14, 99] N·m/rad, `servo_kd` ∈ [0.9, 6.3], `tau_max` ∈ {5, 25, 50, 88, 139} N·m.
So `u_lb`/`u_ub` are *not* joint limits — they are `jnt_range ± tau_max/kp`, which is why they
reach ±5 rad. **Always clip**: `u_bar + K δx` can leave the box, and the plant clips whether you
do or not, so an unclipped command makes your bookkeeping disagree with the physics.

There is feedback inside the plant regardless of `K`. Setting `K = 0` is *not* open loop — it
leaves the servos tracking a fixed tape at 200 Hz.

## Reproducing the plant

**The model file is not the plant.** `g1_29dof.xml` authors `timestep = 0.002` and
`integrator = "euler"`; the example overrides both before designing anything, so `K` was built
around a *different* simulator than a naive `from_xml_path` gives you:

| | XML says | `K` was designed on |
|---|---|---|
| `opt.timestep` | 0.002 | **0.005** (`sim_timestep`) |
| `opt.integrator` | `euler` | **`implicitfast`** (`sim_integrator`) |

Everything else — friction, `cone` (`pyramidal`), `condim`, joint limits — is the XML's own. Skip
the two overrides and the discrepancy is 0.68 on a *single* step and 15 in tangent norm after 3 s:
the robot stays up, but you are no longer running the controller that was designed.

### Validated

The full 1720-step closed loop, driven from this file alone with the overrides applied:

| block | max \|·\|∞ | RMS |
|---|---|---|
| base position | 0.0077 m | 0.0036 m |
| base rotation | 0.0762 rad | 0.0238 rad |
| joint angles | 0.1253 rad | 0.0207 rad |
| base linear velocity | 0.2428 m/s | 0.0370 m/s |
| base angular velocity | 2.90 rad/s | 0.2973 rad/s |
| joint velocity | 16.86 rad/s | 0.4363 rad/s |

Upright for all 8.6 s, travelling 11.596 m against the nominal's 11.601 m, with **0.01%** of
(step, actuator) commands on the control box. The large joint-velocity peak is a contact-transition
spike, not drift — the RMS is 0.44 rad/s and the position blocks stay in the millimetre/centiradian
range throughout.

## `P` — what it is, and what it is not

`P_k` is the optimal cost-to-go of the LQR problem that produced `K`, under the convention
`J_LQR = (2/dt) J` (the stage `dt` and the ½ are stripped), so it is not in raw cost units. It
satisfies the Riccati identity `V_k(δx) = stage cost + V_{k+1}(δx')` along the optimal closed
loop, and `P[n] = Q_f`.

It is **valid for the gain in this file**: `gain_scale = 1.0` and there is no feature mask or
gravity projection, so `K` *is* the unmasked optimal law. Measured: symmetric to 0.0, PSD at every
`k` (min eig 3.1e-3, max 2.5e5, median condition number 2.5e6). If you ever rescale `K`, `P` stops
being its cost-to-go — the certificate does not survive `K → αK`.

It is a *finite-horizon* value function, not a Lyapunov certificate for running the gait forever.
The periodic fixed point is what would give you that; see the convergence note below.

## Stride structure

The trajectory is **one 172-step stride tiled 10 times** under its own SE(2) advance. The stride
closes to machine precision — `periodicity` max |residual| = **1.3e-17** — so the seams are exact,
not merely small.

Per stride the base advances by `twist_arc` = (1.1694 m, −0.0176 m, −0.0444 rad); over 10 strides
that is 11.6 m in 8.6 s (~1.35 m/s). **Everything is periodic except base x, y and heading**, which
grow. For a phase-indexed policy use `phase = (k % 172) / 172` and work in a base-relative frame,
or subtract `r * twist_arc`.

**Do not use the last stride's gains.** The backward pass starts from an arbitrary `Q_f`, and only
the interior strides have forgotten it. Measured gap between adjacent strides (rotated into a
common frame):

```
stride pair   0-1    1-2    2-3    3-4    4-5    5-6    6-7    7-8    8-9
dK          0.034  0.030  0.022  0.002  0.028  0.028  0.040  0.036  0.834
```

The last pair is 20× the others — that is the `Q_f` transient. Strides 0–7 are usable; the
interior plateau at ~3% is the residual distance from the true periodic gain.

## Using it for learning

- **Behaviour cloning / warm start.** `(x_bar, u_bar)` is an expert tape at 200 Hz. Feasible to
  within a shooting defect of 1.75e-4 (tangent norm) — close, but *not* exactly on-manifold, so a
  pure open-loop replay drifts.
- **Residual policy.** `π(x) = u_bar[k] + K[k] @ state_diff(x, x_bar[k]) + Δ_θ(x)`. The LQR gives
  you a stabilizing base to learn a correction on top of, which is far easier than learning
  stabilization from scratch. Measured on 100 held-out randomized models: open loop **100/100
  fell**, with this law **1/100**.
- **Value initialization / reward shaping.** `δxᵀ P_k δx` is a ready-made quadratic value estimate
  around the gait; usable as a potential for shaping, or to initialize a critic.
- **Gain scheduling features.** `phase`, plus the per-step `‖K_k‖_F` (range [29, 514]) which peaks
  at the contact transitions — a cheap signal for where the gait is stiff.

Pitfalls beyond the two ordering traps: this law is tied to **this model file** (`model` key —
meshes, foot geoms, servo gains, armature). A gain designed on one plant is not valid on another;
the source example randomizes ±10% on `kp`/`kd` and finds the law survives, but that is a measured
result, not a guarantee.

## Caveat on the gain magnitude

This export came from a run with **`lqr_r_scale = 1.0`**, which gives `max|K| = 405`,
`‖K‖_F = 3063`. The example's default is `100`, which gives `max|K| ≈ 32.6` — a 12× softer gain.
`r_scale` multiplies the control weight `R = r_scale · Rtau · kp²`; at 1 the torque penalty that is
a sensible *cost* term makes an aggressive *gain*. The stiff version is markedly more robust here
(1/100 falls vs 40/100 at `r_scale = 100`) but commands much harder and saturates more. If you want
the soft law, re-run without `--r-scale 1` and re-export.

## Regenerating

```bash
export TRAJOPT_ROOT_DIR=/path/to/mj-nlp
python examples/g1_mimic_periodic/g1_mimic_periodic.py run_fwd        # the periodic stride
python examples/g1_mimic_periodic_v0/g1_mimic_periodic_v0.py run_fwd  # tile + design + score
python examples/g1_mimic_periodic_v0/export_tvlqr.py                 # this file
```

`export_tvlqr.py --float32` halves the size (P keeps ~7 significant digits); `--run FRAGMENT`
picks a specific saved run instead of the newest. `source_run` records which run this came from.
