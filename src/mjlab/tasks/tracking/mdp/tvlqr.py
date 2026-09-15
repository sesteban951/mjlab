"""TVLQR guidance for CLF-RL: a model-based controller that shapes reward, never the action.

The trajectory optimizer that produced the tracked motion also produced a time-varying LQR around
it -- a gain schedule ``K_k``, the nominal command ``u_bar_k``, the cost-to-go ``P_k``, and the
per-step decay rate ``alpha_k`` that the backward pass is entitled to. This module makes all four
available inside the env so two rewards can use them (see ``rewards.clf_decrease_rbf`` and
``rewards.qdes_imitation_rbf``).

GUIDE ONLY. The LQR is computed alongside the policy and never applied: ``apply_actions`` still
sends the POLICY's target to the sim. That is CLF-RL as robot_rl implements it -- no safety filter,
no min-norm QP, no projection -- so the policy remains free to disagree and simply earns less
reward for it. ``residual = True`` switches to putting the controller in the loop, which is a
different (and strictly easier) problem; it is here to A/B against, not as the default.

WHY THIS LIVES ON THE ACTION TERM. It needs three things at once: the 200 Hz physics rate (the
schedule was designed at the physics timestep, and ``apply_action`` is the only hook called per
substep -- see ``ManagerBasedRlEnv.step``), the robot state, and ``qdes_policy``, which is this
term's own ``_processed_actions``. A command term would have the first two but not the third.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  matrix_from_quat,
  quat_conjugate,
  quat_mul,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from .commands import MotionCommand


def state_diff(
  pos: torch.Tensor,
  quat: torch.Tensor,
  joint_pos: torch.Tensor,
  vel: torch.Tensor,
  pos_bar: torch.Tensor,
  quat_bar: torch.Tensor,
  joint_pos_bar: torch.Tensor,
  vel_bar: torch.Tensor,
) -> torch.Tensor:
  """MuJoCo's ``mj_differentiatePos`` on the position half, plain difference on the velocity half.

  Returns the (B, 2*nv) tangent error the gain schedule multiplies. Verified to 5.6e-17 against
  ``mj_differentiatePos`` on the exported trajectory.

  THE ROTATION BLOCK IS LOCAL: ``log(q_bar^-1 * q)``, not ``log(q * q_bar^-1)``. mjlab's
  ``quat_box_minus`` is the latter -- the WORLD-frame difference -- and using it here is wrong by
  up to 6.8e-3 rad on this trajectory. Small, systematic, and it silently corrupts both V and
  qdes_ctrl, so it is spelled out rather than delegated.
  """
  d_rot = axis_angle_from_quat(quat_mul(quat_conjugate(quat_bar), quat))
  return torch.cat(
    [pos - pos_bar, d_rot, joint_pos - joint_pos_bar, vel - vel_bar], dim=-1
  )


def reference_offset(export_path: str, mode: str = "mean") -> dict[str, float]:
  """{joint name: action offset} taken from the trajectory itself, for the action term's `offset`.

  WHY NOT `use_default_offset`. With that, a zero action commands the robot's default pose -- the
  HOME keyframe -- while the reference is a running gait, so the imitation reward starts at its
  worst and a zero-initialized actor is maximally wrong. Setting `offset` to 0 is worse still: it
  targets 0 rad, a straight-legged pose the gait never visits. Re-centring the offset on the
  trajectory's own posture is what makes a zero action land ON the gait.

  `mode="mean"` averages u_bar over the whole tape, which is the right choice BECAUSE EPISODES
  START AT A RANDOM PHASE (MotionCommand samples `time_steps`): no fixed offset can match a
  time-varying command at every phase, and the mean is the one that minimizes the expected error
  over them. `mode="first"` pins it to phase 0 instead, which is only sensible with
  `sampling_mode="start"`.
  """
  d = np.load(export_path, allow_pickle=True)
  u_bar = np.asarray(d["u_bar"], dtype=float)
  if mode == "mean":
    ref = u_bar.mean(axis=0)
  elif mode == "first":
    ref = u_bar[0]
  else:
    raise ValueError(f"mode must be 'mean' or 'first', got {mode!r}")
  dof_names = [str(s).split("/")[-1] for s in d["dof_names"]]
  act_dof = np.asarray(d["actuator_dof_index"], dtype=int)
  return {dof_names[act_dof[i]]: float(ref[i]) for i in range(len(ref))}


@dataclass(kw_only=True)
class TvlqrGuidedJointPositionActionCfg(JointPositionActionCfg):
  """A joint-position action that also evaluates a TVLQR schedule for the reward terms."""

  export_path: str
  """The ``*_tvlqr.npz`` written by mj-nlp's ``export_tvlqr.py``. Must carry `alpha`."""

  motion_command_name: str = "motion"
  """Command term whose ``time_steps`` indexes the motion; the schedule index is derived from it."""

  motion_pad_start: int = 0
  """Motion frames of standing hold before the schedule's first entry."""

  alpha_scale: float = 1.0
  """Multiplier on the exported per-step decay ``alpha_k``. 1.0 asks for exactly the rate the
  design guarantees; below 1 is slack, above 1 demands more than the LQR itself achieves."""

  clip_to_ctrl_box: bool = True
  """Clip ``qdes_ctrl`` to the actuator ctrlrange, as the deployed law does."""

  residual: bool = False
  """Put the controller IN THE LOOP (``target = qdes_ctrl + scale * action``) instead of guiding by
  reward. Off by default -- see the module docstring."""

  v_ref_path: str | None = None
  """``.npy`` of per-schedule-entry reference V, shape ``(n,)``; enables ``v_ref``."""

  v_floor_scale: float = 0.0
  """Adds ``v_floor_scale * V_ref,k`` to ``violation_rel``'s denominator; 0 keeps the plain ratio."""

  def build(self, env: ManagerBasedRlEnv) -> TvlqrGuidedJointPositionAction:
    return TvlqrGuidedJointPositionAction(self, env)


class TvlqrGuidedJointPositionAction(JointPositionAction):
  """Joint-position control, plus a per-substep TVLQR evaluation exposed for reward terms."""

  cfg: TvlqrGuidedJointPositionActionCfg

  def __init__(self, cfg: TvlqrGuidedJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    dev = self.device
    d = np.load(cfg.export_path, allow_pickle=True)
    if "alpha" not in d.files:
      raise ValueError(
        f"{cfg.export_path} has no `alpha`; re-run export_tvlqr.py without --no-alpha. The CLF "
        f"reward needs the decay rate the design prescribes, and hand-picking one is exactly the "
        f"mistake that a continuous-time alpha ported to a 200 Hz discrete condition makes."
      )

    self._nv = int(d["nv"])
    self._nq = int(d["nq"])
    self._dt_export = float(d["dt"])
    self._stride = int(d["stride_len"])

    # ---- ROW ORDER. The export's K rows are ACTUATORS in XML declaration order; this term's
    # action vector is `self._target_names`, which find_joints_by_actuator_names returns in the
    # model's natural JOINT order. Those disagree on this robot (export row 0 is
    # left_shoulder_pitch, tangent column 0 is the floating base). Mapped BY NAME rather than by
    # the argsort the export README suggests, so a model that reordered either one still lands
    # correctly instead of scrambling silently.
    dof_names = [str(s).split("/")[-1] for s in d["dof_names"]]
    act_dof = np.asarray(d["actuator_dof_index"], dtype=int)
    act_joint_name = [
      dof_names[i] for i in act_dof
    ]  # the joint each exported actuator drives
    name_to_row = {n: i for i, n in enumerate(act_joint_name)}
    missing = [n for n in self._target_names if n not in name_to_row]
    if missing:
      raise ValueError(
        f"the export has no actuator driving {missing}; it was built for a different robot"
      )
    rows = torch.tensor(
      [name_to_row[n] for n in self._target_names], dtype=torch.long, device=dev
    )

    # ---- COLUMN ORDER. K's columns are the tangent: [base(6), dofs 6..nv-1]. The joint half must
    # be this entity's joint order, or the gain reads the wrong error component.
    ent_joint_names = list(self._entity.joint_names)
    if dof_names[6 : self._nv] != ent_joint_names:
      raise ValueError(
        "the export's tangent joint order does not match the entity's joint order, so K's columns "
        f"would be misread. export={dof_names[6 : self._nv][:4]}... entity={ent_joint_names[:4]}..."
      )

    def t(key: str) -> torch.Tensor:
      return torch.as_tensor(np.asarray(d[key]), dtype=torch.float32, device=dev)

    self._x_bar = t("x_bar")  # (n+1, nq+nv)
    self._u_bar = t("u_bar")[:, rows]  # (n,   nu) in target order
    self._K = t("K")[:, rows, :]  # (n, nu, ndx) rows permuted
    self._P = t("P")  # (n+1, ndx, ndx)
    self._alpha = t("alpha") * float(cfg.alpha_scale)  # (n,)
    self._u_lb = t("u_lb")[rows]
    self._u_ub = t("u_ub")[rows]
    self._n_steps = int(self._u_bar.shape[0])

    self._v_ref: torch.Tensor | None = None
    if cfg.v_ref_path is not None:
      v_ref = torch.as_tensor(np.load(cfg.v_ref_path), dtype=torch.float32, device=dev)
      if v_ref.shape != (self._n_steps,) or not bool((v_ref > 0).all()):
        raise ValueError(
          f"{cfg.v_ref_path} must be positive with shape ({self._n_steps},), "
          f"got {tuple(v_ref.shape)} with min {float(v_ref.min()):.3g}"
        )
      self._v_ref = v_ref
    if cfg.v_floor_scale < 0 or (cfg.v_floor_scale > 0 and self._v_ref is None):
      raise ValueError(
        f"v_floor_scale={cfg.v_floor_scale} must be >= 0, and > 0 needs a v_ref_path"
      )

    # ---- rates. The schedule is per PHYSICS step and the motion index advances per ENV step, so
    # the two are related by the decimation. A resampled artifact would desync them silently.
    dec = int(env.cfg.decimation)
    # The schedule advances one entry per STRIDE physics substeps, so physics may integrate finer
    # than the rate K was designed for; qdes_ctrl is then held between updates, as the 100 Hz
    # deployed law does against a 200 Hz plant. stride == 1 is the designed-rate case.
    stride = round(self._dt_export / env.physics_dt)
    if stride < 1 or abs(self._dt_export - stride * env.physics_dt) > 1e-12:
      raise ValueError(
        f"the export is at dt={self._dt_export} but physics_dt={env.physics_dt}; the export rate "
        f"must be an integer multiple of the physics rate, so the schedule lands on substeps"
      )
    if dec % stride:
      raise ValueError(
        f"decimation={dec} is not divisible by the schedule stride {stride} "
        f"(dt_export={self._dt_export} / physics_dt={env.physics_dt}); the env step would end "
        f"part-way through a schedule entry"
      )
    self._decimation = dec
    self._stride = stride

    # ---- per-env buffers
    n_envs = self.num_envs
    self._substep = 0
    self._k0 = torch.zeros(n_envs, dtype=torch.long, device=dev)
    self._in_clip = torch.ones(n_envs, dtype=torch.bool, device=dev)
    self._qdes_ctrl = torch.zeros(n_envs, self.action_dim, device=dev)
    self._v = torch.zeros(n_envs, device=dev)
    self._v_prev = torch.zeros(n_envs, device=dev)
    self._violation = torch.zeros(n_envs, device=dev)
    self._violation_rel = torch.zeros(n_envs, device=dev)
    self._have_prev = torch.zeros(n_envs, dtype=torch.bool, device=dev)
    self._v_ref_now = torch.ones(n_envs, device=dev)

  ##
  # Properties read by the reward terms.
  ##

  @property
  def qdes_ctrl(self) -> torch.Tensor:
    """(B, nu) the model-based command, in this term's action order, ctrl-box clipped."""
    return self._qdes_ctrl

  @property
  def qdes_policy(self) -> torch.Tensor:
    """(B, nu) what the POLICY asked for: raw action * scale + offset, i.e. already un-normalized.
    Comparing against the raw action instead would compare a dimensionless number to radians."""
    return self._processed_actions

  @property
  def in_clip(self) -> torch.Tensor:
    """(B,) True while the motion is inside the schedule, False in its standing pads."""
    return self._in_clip

  @property
  def v(self) -> torch.Tensor:
    """(B,) the Lyapunov value at the latest substep."""
    return self._v

  @property
  def violation(self) -> torch.Tensor:
    """(B,) worst CLF-decrease violation over the last env step, >= 0.

    ``max(V_{k+1} - V_k + alpha_k V_k, 0)``, the discrete form of robot_rl's
    ``clf_decreasing_condition``. ONE-SIDED, as theirs is: beating the required rate earns nothing,
    so the policy is not pushed to over-stabilize. Reduced by MAX over the decimation window rather
    than mean, so one bad substep is not averaged away by three good ones.
    """
    return self._violation

  @property
  def violation_rel(self) -> torch.Tensor:
    """(B,) the violation as a FRACTION of V: ``max(V_{k+1}/V_k - 1 + alpha_k, 0)``.

    The scale-free form, and the one the CLF reward should use. Measured on a zero-action rollout
    the RAW violation spans 0.3 to 833 (p5 to p99) -- four orders of magnitude, because V itself
    ranges over three -- so no single sigma can cover it. This ratio lands at p50 0.08, p95 0.24,
    which is what makes a fixed sigma meaningful. It is also directly interpretable: the fraction
    of the decay the design asked for that the policy failed to deliver.

    Same purpose as robot_rl dividing by ``2||P|| eta_max eta_dot_max + alpha lam_max eta_max^2``,
    but measured against the ACTUAL V rather than an assumed error bound, so there is no eta_max
    to guess. Accumulated per substep, since normalizing the max afterwards would divide by the
    wrong V.
    """
    return self._violation_rel

  @property
  def v_ref(self) -> torch.Tensor:
    """(B,) the reference V at the latest substep's schedule entry."""
    if self._v_ref is None:
      raise ValueError("v_ref needs TvlqrGuidedJointPositionActionCfg.v_ref_path")
    return self._v_ref_now

  ##
  # ActionTerm hooks.
  ##

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    cmd = cast(
      "MotionCommand", self._env.command_manager.get_term(self.cfg.motion_command_name)
    )
    # the motion index of the step ABOUT TO RUN; the command manager advances it after the rewards
    k0 = (cmd.time_steps - self.cfg.motion_pad_start) * (
      self._decimation // self._stride
    )
    self._in_clip = (k0 >= 0) & (k0 < self._n_steps)
    self._k0 = torch.clamp(k0, min=0, max=self._n_steps - 1)
    self._substep = 0
    self._violation.zero_()
    self._violation_rel.zero_()

  def apply_actions(self) -> None:
    substep = self._substep
    self._substep += 1
    if substep % self._stride:
      self._write_target()
      return
    k = torch.clamp(self._k0 + substep // self._stride, max=self._n_steps - 1)

    dx = self._tangent_error(k)
    qdes = self._u_bar[k] + torch.einsum("bij,bj->bi", self._K[k], dx)
    if self.cfg.clip_to_ctrl_box:
      qdes = torch.clamp(qdes, min=self._u_lb, max=self._u_ub)
    self._qdes_ctrl = qdes

    v = torch.einsum("bi,bij,bj->b", dx, self._P[k], dx)
    step_viol = torch.clamp(v - self._v_prev + self._alpha[k] * self._v_prev, min=0.0)
    floor = 0.0
    if self._v_ref is not None:
      self._v_ref_now = self._v_ref[k]
      floor = self.cfg.v_floor_scale * self._v_ref_now
    step_rel = step_viol / (self._v_prev + floor + 1e-6)
    # the first substep after a reset has no predecessor to decrease from
    live = self._have_prev
    self._violation = torch.maximum(self._violation, torch.where(live, step_viol, 0.0))
    self._violation_rel = torch.maximum(
      self._violation_rel, torch.where(live, step_rel, 0.0)
    )
    self._v, self._v_prev = v, v
    self._have_prev = torch.ones_like(self._have_prev)

    self._write_target()

  def _write_target(self) -> None:
    """Write the joint target from the CURRENT ``qdes_ctrl``, held between schedule updates."""
    if self.cfg.residual:
      # qdes + (action * scale), recovered as processed - offset so a zero action is exactly the
      # controller. encoder_bias is subtracted for the same reason the parent does it: it is a
      # SENSING perturbation, so the servo compares the target against a biased measurement and
      # the compensation has to be in the target. Omitting it leaves a per-joint offset that looks
      # like a steady-state tracking error.
      encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
      self._entity.set_joint_position_target(
        self._qdes_ctrl + self._processed_actions - self._offset - encoder_bias,
        joint_ids=self._target_ids,
      )
    else:
      super().apply_actions()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    idx = slice(None) if env_ids is None else env_ids
    self._have_prev[idx] = False
    self._v_prev[idx] = 0.0
    self._violation[idx] = 0.0
    self._violation_rel[idx] = 0.0

  ##
  # Internals.
  ##

  def _tangent_error(self, k: torch.Tensor) -> torch.Tensor:
    """(B, 2*nv) state_diff(robot, x_bar[k]), in the export's own frames."""
    data = self._entity.data
    # x_bar is in the trajectory's absolute frame; the robot is offset to its env's origin
    pos = data.root_link_pos_w - self._env.scene.env_origins
    quat = data.root_link_quat_w
    # MuJoCo's free joint stores LINEAR velocity in world and ANGULAR velocity in the BODY frame,
    # while mjlab reports both in world -- so only the angular half is rotated back.
    ang_b = torch.einsum("bji,bj->bi", matrix_from_quat(quat), data.root_link_ang_vel_w)
    vel = torch.cat([data.root_link_lin_vel_w, ang_b, data.joint_vel], dim=-1)
    xb = self._x_bar[k]
    return state_diff(
      pos,
      quat,
      data.joint_pos,
      vel,
      xb[:, 0:3],
      xb[:, 3:7],
      xb[:, 7 : self._nq],
      xb[:, self._nq :],
    )
