# `output/` — V0 `F2@r1000` export

Four arrays pulled out of one solved run, at the simulation time resolution, for use outside this
repo. Three describe the controller (`state`, `input`, `K`); the fourth, `P`, is the quadratic
Lyapunov function of that controller.

**Provenance.** `examples/g1_gait_v0/g1_gait_v0_run_fwd_20260829T163422.npz`, arm `F2@r1000`:

```bash
python examples/g1_gait_v0/g1_gait_v0.py run_fwd          # the run (F0/F1G/F1/F2 x r 1/100/1000)
python output/export_arm.py                               # newest run, F2@r1000 -> these files
python examples/replay.py g1_gait_v0 --arm F2@r1000       # the run itself, all arms
```

`F2@r1000` is **not** this run's winner — `F2@r100` is, at 0.229 held-out against 0.561. It is
exported because it is the arm this folder has always carried. `export_arm.py <run.npz> <arm>`
takes any other.

`V0` is **one nominal solve plus an LQR schedule designed around its own rollout, played back on
perturbed models.** There is no robust solve and no feedback inside the optimization — the gain here
was designed *after* the solve and applied at evaluation only.

| | |
|---|---|
| robot | Unitree G1, 29 dof — `models/unitree_g1/g1_29dof.xml` (27 floor-contact geoms) |
| motion | `run_fwd`, 0.86 s |
| `sim_dt` | **0.005 s** — every array below is on this grid |
| `n_sim` | 172 steps, so 173 states |
| feature set | `F2` — the law reads the **whole** tangent state, base position included |
| `r_scale` | 1000 (multiplier on the LQR control weight; larger = softer gain) |
| `gain_scale` | 1.0 |
| ensemble | 5 models (`seed = 0`): row 0 is the **nominal, unperturbed** robot |
| held out | 100 further models at `test_seed = 10000`, not exported |

## The law

$$u^{(s)}_k = \mathrm{clip}\!\left(\bar u_k + K_k\,\left(x^{(s)}_k \ominus \bar x_k\right),\ u_{lb},\ u_{ub}\right)$$

with $\bar u$ = `input_plan`, $\bar x$ = `state_reference`, $K$ = `K`. Everything on the right is in
these files.

## Files

### `state.npz`

| key | shape | what |
|---|---|---|
| `time` | (173,) | seconds, `0, 0.005, … 0.86` |
| `state` | (5, 173, 71) | the **closed-loop** trajectory of each of the 5 models under the law |
| `state_reference` | (173, 71) | $\bar x$ — the trajectory the gain measures error against |
| `state_plan` | (173, 71) | the optimizer's own stage-1 trajectory (node-restarted) |
| `nq`, `nv`, `nx` | () | 36, 35, 71 |

State layout is `[qpos(36) | qvel(35)]`, and `qpos = [base_pos(3) | base_quat(4, wxyz) | joints(29)]`.

`state_reference` is the nominal model's open-loop rollout of `input_plan` — the point the LQR was
linearized about. It is the same array for every arm of the run (it depends on the tape and the model
only, not on the feature set or `r_scale`); verified bitwise against the open-loop arm's nominal row.

### `input.npz`

| key | shape | what |
|---|---|---|
| `time` | (172,) | the input grid — one shorter than the state grid |
| `input` | (5, 172, 29) | the command each model **actually received**, after clipping |
| `input_plan` | (172, 29) | $\bar u = Hp$, the open-loop tape from the solve |
| `u_lb`, `u_ub` | (29,) | the actuator box the commands are clipped to |
| `actuator` | (29,) | actuator names, in the column order of `input` and of `K`'s rows |
| `actuator_mode` | () | `position` — see below |

**`actuator_mode = position` means `u` is a joint-angle target, not a torque.** The XML's `<general>`
actuators are PD servos and the plant applies $\tau = k_p(u - q) - k_d\dot q$ itself. Feeding these
numbers to a torque-controlled model would be wrong by orders of magnitude.

### `K.npz`

| key | shape | what |
|---|---|---|
| `K` | (172, 29, 70) | the gain schedule, one $29 \times 70$ matrix per sim step |
| `time` | (172,) | as above |
| `actuator` | (29,) | `K`'s **rows** |
| `tangent_block`, `tangent_start`, `tangent_stop` | (6,) | `K`'s **columns**, see below |
| `ndx`, `nv` | () | 70, 35 |

`K`'s columns live on the **tangent** state, $n_{dx} = 2n_v = 70$, which is *not* the 71-wide full
state — the base quaternion contributes 3 tangent components, not 4:

```
cols  0.. 3   base_pos        cols 35..38   base_linvel
cols  3.. 6   base_rot        cols 38..41   base_angvel
cols  6..35   joint_pos       cols 41..70   joint_vel
```

### `P.npz`

The cost-to-go of the gain in `K.npz` — a **Lyapunov function** for the deployed law.

| key | shape | what |
|---|---|---|
| `P` | (173, 70, 70) | $P_k$, one symmetric PSD $70 \times 70$ per **state** step |
| `time` | (173,) | the state grid, as in `state.npz` |
| `rho_sat` | (172,) | saturation radius — see below |
| `Q_diag`, `Qf_diag` | (70,) | the stage / terminal weight $P$ was built with |
| `R_diag` | (29,) | the control weight, already carrying `r_scale` |
| `r_scale` | () | 1000 |
| `tangent_block`, `tangent_start`, `tangent_stop` | (6,) | `P`'s rows **and** columns — same layout as `K`'s columns |
| `ndx`, `nv` | () | 70, 35 |
| `use_E` | () | `True` — the log-map factor is included, see below |

$P$ is indexed like `state`, not like `input`: there are 173 of them, and $P_{172}$ is the terminal
weight $Q_f$ itself.

**What it certifies.** With $M_k = K_k E_k$ and $A^{cl}_k = A_k + B_k M_k$, $P$ satisfies

$$P_k = Q + M_k^\top R\, M_k + A_k^{cl\top} P_{k+1} A^{cl}_k$$

which makes $V_k(\delta x) = \delta x^\top P_k\, \delta x$ decrease along the **linearized** closed
loop, for every $\delta x$, at every $k$:

$$V_{k+1}\!\left(A^{cl}_k \delta x\right) - V_k(\delta x) = -\,\delta x^\top\!\left(Q + M_k^\top R\, M_k\right)\delta x \;\le\; 0$$

That is an identity, not a search — it holds to $3.6\times10^{-12}$ here, `P` is exactly symmetric,
and $\lambda_{\min}(P_k) \ge 0.0104$ over all $k$.

**$E_k$ is not the identity.** $E_k = \partial\,\mathrm{state\_diff}(x, \bar x_k)/\partial x$ is the
$SO(3)$ log-map right-Jacobian on the `base_rot` block and $I$ elsewhere. The control deviation as a
function of a tangent state perturbation is $\delta u = K E\,\delta x$, so $P$ is built against
$A + BKE$, not $A + BK$. On this run that distinction moves $A^{cl}$ by $\sim\!3\times10^{-3}$
relative.

**`rho_sat` is a validity radius, not part of the Lyapunov function.** The linear analysis holds only
while the command stays inside the actuator box. Given headroom
$m_{k,i} = \min(u_{ub,i} - \bar u_{k,i},\; \bar u_{k,i} - u_{lb,i})$,

$$\rho^{\mathrm{sat}}_k = \min_i \frac{m_{k,i}^2}{k_i^\top P_k^{-1} k_i}$$

is the largest $\rho$ for which $\lVert\delta x\rVert^2_{P_k} \le \rho$ guarantees no clipping
($k_i^\top$ = row $i$ of $M_k$). Here it runs 0.066 … 8.21, median 5.84. It is a **necessary**
condition only: on this motion the true dynamics leave the linear regime *before* saturation binds,
at roughly $\rho \approx 0.5$–$2.7$ — measured by sampling, not certified. Do not read
$\rho^{\mathrm{sat}}$ as a region of attraction.

## Applying `K` — the one thing that will bite you

$\ominus$ is **not** subtraction. `state - state_reference` is 71-wide and its base-orientation part
is a quaternion difference, which is meaningless as a tangent vector. The error must be built with
MuJoCo's own difference:

```python
import numpy as np, mujoco

m = mujoco.MjModel.from_xml_path("models/unitree_g1/g1_29dof.xml")
nq, nv = m.nq, m.nv


def state_diff(x, x_ref):
  """x (-) x_ref in tangent coordinates: (2*nv,)."""
  dq = np.zeros(nv)
  mujoco.mj_differentiatePos(
    m, dq, 1.0, x_ref[:nq], x[:nq]
  )  # log map on the free joint
  return np.concatenate([dq, x[nq:] - x_ref[nq:]])


S = np.load("output/state.npz")
I = np.load("output/input.npz")
G = np.load("output/K.npz")
s = 1  # a perturbed model
for k in range(len(G["time"])):
  e = state_diff(S["state"][s, k], S["state_reference"][k])
  u = np.clip(I["input_plan"][k] + G["K"][k] @ e, I["u_lb"], I["u_ub"])
  assert np.allclose(u, I["input"][s, k], atol=1e-9)  # reproduces the saved command
```

That assertion was run over all 5 models and all 172 steps of this export and passes **exactly**
(`max|Δu| = 0`), so the three files are mutually consistent and the recipe is the one that produced
them. Swapping `state_diff` for plain subtraction of the first 70 entries — the obvious shortcut —
puts `max|Δu|` at **1.039** instead, which is the whole reason this section exists.

## Using `P`

`P` needs the same tangent error `K` does, so the `state_diff` from the previous section is the
only ingredient not already in the files:

```python
Pz = np.load("output/P.npz")
s = 1  # a perturbed model
V = [
  (lambda e: e @ Pz["P"][k] @ e)(state_diff(S["state"][s, k], S["state_reference"][k]))
  for k in range(len(Pz["time"]))
]
# V[k] is the Lyapunov value of model s at step k; compare against Pz["rho_sat"][k]
```

On the five ensemble rows:

```
model    V max    steps with V > rho_sat        (model 0 = the NOMINAL robot)
  0     0.7986            1 / 172
  1     3.0788            2 / 172
  2     5.5443            4 / 172
  3     1.3540            1 / 172
  4     3.4429            3 / 172
```

Model 0 is no longer $V \equiv 0$: it is the nominal robot, but evaluated on a *different collision
model* than the reference was built on — see the solve/eval section below.

Worth noting where that sits: the sampled breakdown radius for this class of controller is
$\rho \approx 0.5$–$2.7$, so **the models spend part of the horizon at or past the edge of the region
where $V$ is trustworthy.** The law still recovers them — held-out mean 0.561 against 12.93 open loop
— but that recovery is not something $P$ certifies. It is evidence the funnel estimate is
conservative, not that the models stayed inside it.

**Regenerating.** `P` is a function of the tape, the model and `K` — not of a solve — so it is
rebuilt from the other three files plus `src.feedback.lyapunov_cost_to_go`, without re-running
`g1_gait_v0.py`. See that function's docstring for the recursion and
`fb.cost_to_go(dyn, X_bar, U_bar, K, cfg)` for the wrapper that assembles $A$, $B$, $E$ and the
weights.

## Historical note: the Aug-28 export

An earlier version of these files, exported 2026-08-28, carried a `K` that could **not** be
regenerated from the tree — re-designing against its own tape gave a gain ~1.8x stiffer, and no
`r_scale` reproduced it. Ruled out at the time: the tape (bitwise), the model, the weights, `Rrate`,
warm-start ordering, and any $(Q, R)$ pair. $A$ and $B$ validated against the true one-step map to
$\sim\!10^{-9}$. The cause was never found, and the files have since been replaced by the run above,
which is self-consistent (`max|du| = 0` exactly). Recorded because it was never explained, not
because it still affects anything here.

## The solve and eval models disagree on foot contact

The solve runs on `g1_29dof_feet.xml` (4 spheres per foot, $\mu = 1.0$, 8 floor-contact geoms); every
evaluation runs on `g1_29dof.xml`, which now carries mjlab's original 7 capsules per foot (33
floor-contact geoms). These used to be the same geometry — the sphere-footed `g1_29dof.xml` was
*dynamically identical* to the solve model on this motion, since only the feet ever reach the floor.

They are not any more, and the gap is large: replaying `input_plan` through the two models diverges
by **4.67 m**. That is what makes row 0's feedback non-trivial (see below), and it is why `K` and `P`
— both built on the solve model — describe a linearization the evaluation never exactly sees.

Note the config's `friction=1.0` override still applies to every geom at load, so the capsules'
authored $\mu = 0.6$ is not what runs; only the geometry changed.

## Two things to know before reading the numbers

**Row 0's feedback is NO LONGER inert** — this changed, and it is worth understanding. `state_reference`
is the rollout on the **solve** model (`g1_29dof_feet.xml`, 4 spheres per foot) while every evaluation
runs on `g1_29dof.xml`, which now carries mjlab's 7 capsules. The two no longer agree on foot contact,
so even the unperturbed robot drifts off the reference and the law acts on it:
`max|input[0] - input_plan| = 1.8e-01`, against `7.7e-15` when both models had spheres. The perturbed
rows reach **0.302**.

That offset is real, not an artifact — it is the "collision-model offset that the design never saw"
the run prints — but it means row 0 is now a *sim-to-sim transfer* row rather than a control. To
recover a true no-feedback baseline, use the `open` arm.

**`F2` is an upper bound, not a deployable law.** It reads base position and base linear velocity,
which no onboard sensor gives you — those need a state estimator. The deployable rung is `F1G`
(projected gravity + body-frame angular rate + the joints, no base pose), and on this run it is
**materially worse**, not marginally:

```
held-out mean over the same 100 models, r_scale = 1000
  F0    55.2610      <- worse than open loop
  F1G    0.9851
  F1     0.7704
  F2     0.5607      <- this export
  open  12.9280

F1G - F2 = +0.4244 +/- 0.1039   (F1G cheaper on 11/100)
```

So at `r_scale = 1000` the base-position signals are worth a factor of 1.8 in held-out mean, and the
gap is resolved at ~4.1 sigma. Do not read this export as an achievable on-robot result. `F0` — joint
states only, what the servo already sees — is worse than no feedback at all here.

## Cost of this arm

Scored with the solve's own objective, $J = \Delta t \sum_k \ell + \ell_f$:

```
ensemble (5 models, seed 0)   0.0584  0.9715  1.6935  0.2293  0.7852
held out (100 models)         mean 0.5607   worst 2.7183
the same 100, law OFF         mean 12.9280
```

So the gain is worth roughly a factor of 23 in held-out mean cost on this motion.
(`F2@r100`, this run's winner, reaches 0.2292 — a factor of 56.)

## Note

`*.npz` is in the repo's `.gitignore`, so these four files are local only — this README is the part
that gets committed. Regenerate `state` / `input` / `K` by re-running the two commands at the top of
this file; `P` is rebuilt from those three, see **Using `P`**.
