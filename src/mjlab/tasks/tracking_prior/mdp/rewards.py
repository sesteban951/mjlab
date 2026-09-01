"""Rewards for the tracking task with a control prior."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from mjlab.envs.mdp.actions import JointPositionActionWithPrior
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking_prior.mdp.priors import tangent_state_error

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg


class action_prior_deviation:
  """Penalize ``||pi(o) - u_prior||^2``, faded out as the prior weight goes to zero.

  While ``lam`` is large the policy's own output barely reaches the sim, so nothing
  shapes it and the handoff at ``lam = 0`` is a step change. This pulls ``pi(o)``
  onto the prior exactly while the prior is the thing driving: the penalty is scaled
  by ``lam**power``, the prior's share of the command, so it decays with the same
  curriculum that anneals ``lam`` and is identically zero once ``lam`` is.

  Both terms are absolute joint targets in radians, so the error is a joint angle.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    action_name: str = cfg.params.get("action_name", "joint_pos")
    term = env.action_manager.get_term(action_name)
    if not isinstance(term, JointPositionActionWithPrior):
      raise TypeError(
        f"Action term '{action_name}' is a {type(term).__name__}, which has no prior "
        f"to deviate from. Expected JointPositionActionWithPrior."
      )
    self._term = term
    self._zero = torch.zeros(env.num_envs, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
    power: float = 1.0,
  ) -> torch.Tensor:
    del env, action_name  # Resolved at init.
    weight = self._term.prior_weight
    if weight <= 0.0:
      return self._zero
    error = self._term.processed_actions - self._term.prior_target
    return weight**power * torch.sum(torch.square(error), dim=1)


class action_prior_deviation_exp:
  """Reward ``exp(-mean((pi(o) - u_prior)^2) / std^2)``, closeness to the prior's command.

  The exponential-kernel sibling of :class:`action_prior_deviation`, in the same form as
  the task's motion-tracking rewards, and different from that squared penalty in three
  ways.

  It is a **bonus**, not a penalty: bounded in ``(0, 1]``, worth 1 when the policy
  reproduces the prior exactly and decaying smoothly to 0 as it walks away. An unbounded
  squared penalty can be reduced without limit by any behavior that keeps the robot
  near the prior's command, which at the start of training is most easily bought by
  going limp; a bounded bonus cannot be farmed that way, and it sits on the same scale
  as the tracking terms it competes with.

  Its gradient is **local**. Beyond a few ``std`` the kernel is flat, so a policy that
  is nowhere near the prior is not dragged toward it and is free to be shaped by the
  tracking rewards instead; the pull only switches on once it is already close. The
  squared penalty does the opposite, pulling hardest exactly where the policy is most
  wrong.

  It does **not** fade with the prior's share of the command, so it stays live at
  ``lam = 0`` and under ``blend="nominal"``, where the prior never drives the robot and
  this is the only thing tying the policy to it -- a behavior-cloning pull toward a
  controller the actor cannot see. Set ``fade_with_prior=True`` to recover the scaling
  of :class:`action_prior_deviation`, which only means something under
  ``blend="convex"``.

  ``std`` is in **radians** and is a per-joint RMS deviation: the squared error is
  averaged over joints, not summed, so the same ``std`` means the same thing on a 23-
  and a 29-dof robot. At ``std = 0.2`` a uniform 0.2 rad miss on every joint scores
  ``1/e``.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    action_name: str = cfg.params.get("action_name", "joint_pos")
    term = env.action_manager.get_term(action_name)
    if not isinstance(term, JointPositionActionWithPrior):
      raise TypeError(
        f"Action term '{action_name}' is a {type(term).__name__}, which has no prior "
        f"to deviate from. Expected JointPositionActionWithPrior."
      )
    if float(cfg.params.get("std", 0.0)) <= 0.0:
      raise ValueError(f"std must be positive, got {cfg.params.get('std')}.")
    self._term = term
    self._zero = torch.zeros(env.num_envs, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    action_name: str = "joint_pos",
    fade_with_prior: bool = False,
  ) -> torch.Tensor:
    del env, action_name  # Resolved at init.
    error = self._term.processed_actions - self._term.prior_target
    bonus = torch.exp(-torch.mean(torch.square(error), dim=1) / std**2)
    if not fade_with_prior:
      return bonus
    weight = self._term.prior_weight
    if weight <= 0.0:
      return self._zero
    return weight * bonus


class lqr_lyapunov_shaping:
  r"""Potential-based shaping on the nominal LQR cost-to-go.

  Adds ``F(s, s') = gamma * Phi(s') - Phi(s)`` to the reward with the potential

  .. math:: \Phi(s) = -\kappa\, s^\top P s

  where ``s`` is the tangent error against the tape's reference trajectory and ``P``
  solves the Riccati equation of the nominal LQR problem the tape's gains came from.
  Expanded, the term this returns is

  .. math:: F(s, s') = \kappa\,(s^\top P s - \gamma\, s'^\top P s')

  i.e. a dense per-step bonus for decreasing the LQR Lyapunov function. ``P`` is
  positive semidefinite, so ``Phi <= 0``: the potential is a well whose floor is the
  reference trajectory, and the agent is paid on every step for descending it rather
  than waiting for a sparse progress signal to arrive.

  **The potential must be bounded, and there are two ways to do it.** With ``v_half`` set
  (preferred), the potential saturates smoothly::

      Phi(s) = -kappa * V / (V + v_half)

  which lies in ``(-kappa, 0]``, is steepest at ``V = 0``, half-saturates at
  ``V = v_half``, and keeps a gradient at every ``V``. Set ``v_half`` near the ``V`` the
  policy actually operates at or the term is flat where it matters -- and note that value
  is a property of the *policy*, not of ``P``. On the G1 running tape a trained tracking
  policy sits at ``V ~ 43`` (median) while ``rho_sat`` is 5.87, because ``P`` was designed
  around an LQR holding deviations an order of magnitude tighter: a 5 cm base-position
  error alone is worth ``V = 25`` under that ``P``.

  The alternative, ``clip_to_rho_sat``, hard-clips ``V`` at the tape's ``rho_sat``. It
  bounds the leak equally well but goes **flat** past the threshold, so a policy living
  entirely outside ``rho_sat`` sees a constant potential, a constant ``F``, and no
  gradient whatsoever -- measured as exactly no effect over a full 7000-iteration run.
  Prefer ``v_half`` unless the policy is genuinely inside the funnel.

  **Why it must be bounded at all.** Expanding the term around a steady ``V`` leaves
  ``F -> kappa * (1 - gamma) * V``, a *positive* payment proportional to how far the robot
  is from the reference, collected forever. Unbounded, with ``P``'s eigenvalues reaching
  1e4, leaving the reference and staying there outearns tracking it: on this tape
  ``kappa = 0.1`` and ``kappa = 1.0`` both drove mean episode length from ~490 to under 15
  within a few hundred iterations, with every tracking reward at ~0.01. Bounding the
  potential caps that residual -- at ``kappa * (1 - gamma)`` for the saturating form, at
  ``kappa * (1 - gamma) * rho_sat`` for the clip. Neither costs the invariance:
  ``V / (V + v_half)`` and ``min(V, rho)`` are both functions of state, so both are still
  potentials.

  Because it is a potential difference, this does not change the optimal policy of the
  underlying MDP (Ng, Harada & Russell, 1999) -- it only changes how fast the agent
  finds it. Two caveats on that guarantee as it lands here:

  * The manager scales every reward term by ``step_dt``. That is a uniform scaling of
    the base reward and the shaping alike, so the invariance survives with the
    potential ``step_dt * Phi``; the effective ``kappa`` is ``weight * kappa * step_dt``.
  * The guarantee assumes ``Phi = 0`` at a terminal state. This emits **zero** on the
    first step of an episode (there is no ``s`` to have moved from) but scores the last
    step of an episode normally, so an early termination pockets ``-Phi(s_terminal) >= 0``
    -- a bonus for dying far from the reference. Clipping bounds that bonus too, but give
    the task a termination penalty that dominates it if ``kappa`` is large.

  ``gamma`` must be the discount the policy is actually trained with, or the telescoping
  that makes this policy-invariant does not hold. It does not default to anything for
  that reason.

  ``rho_scale`` tightens the clip below ``rho_sat``. On the G1 running tape ``rho_sat``
  runs 0.066 to 8.21 while the true breakdown radius is nearer 0.5 to 2.7 (sampled, not
  certified), so a scale below 1 is defensible if the leak still shows.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    entity_name: str = cfg.params.get("entity_name", "robot")
    self._entity = env.scene[entity_name]
    njoint = self._entity.data.joint_pos.shape[1]
    ndx = 2 * (6 + njoint)

    tape_file: str = cfg.params["tape_file"]
    p_file: str | None = cfg.params.get("p_file")
    tape = np.load(tape_file)
    source = np.load(p_file) if p_file is not None else tape
    if "P" not in source:
      raise KeyError(
        f"{p_file or tape_file} has no 'P' array, so there is no Riccati solution to "
        f"shape with. Export the cost-to-go alongside the gain schedule -- it is the "
        f"P of the same LQR solve that produced 'gain' -- as either a ({ndx}, {ndx}) "
        f"matrix or one per tape frame, then point 'tape_file' or 'p_file' at it."
      )

    def _to(src, key: str) -> torch.Tensor:
      return torch.tensor(src[key], dtype=torch.float32, device=env.device)

    P = _to(source, "P")
    if P.shape[-2:] != (ndx, ndx):
      raise ValueError(
        f"P is {tuple(P.shape)}, expected ({ndx}, {ndx}) or (frames, {ndx}, {ndx}) for "
        f"a floating base with {njoint} joints."
      )
    if P.ndim not in (2, 3):
      raise ValueError(f"P must be 2- or 3-dimensional, got {P.ndim} dimensions.")
    # The Riccati solution is symmetric; enforce it so an asymmetric export cannot make
    # the quadratic form disagree with the cost it is supposed to represent.
    self._P = 0.5 * (P + P.transpose(-1, -2))
    self._time_varying = P.ndim == 3

    qpos_ref, qvel_ref = _to(tape, "ref_qpos"), _to(tape, "ref_qvel")
    if qpos_ref.shape[1] != 7 + njoint or qvel_ref.shape[1] != 6 + njoint:
      raise ValueError(
        f"Tape reference is {qpos_ref.shape[1]}/{qvel_ref.shape[1]} wide, expected "
        f"{7 + njoint}/{6 + njoint} for a floating base with {njoint} joints."
      )
    if self._time_varying and self._P.shape[0] != qpos_ref.shape[0]:
      raise ValueError(
        f"P has {self._P.shape[0]} frames but the tape reference has "
        f"{qpos_ref.shape[0]}; they must be on the same grid."
      )
    self._qpos_ref, self._qvel_ref = qpos_ref, qvel_ref
    self._num_frames = qpos_ref.shape[0]

    # Validity radius of the linear analysis, per frame. Clipping the potential at it is
    # what keeps the (1 - gamma) leak bounded; see the class docstring.
    # A finite v_half switches the potential to the saturating form and makes the
    # rho_sat clip redundant -- the normalization already bounds it.
    self._v_half: float | None = cfg.params.get("v_half")
    if self._v_half is not None and self._v_half <= 0.0:
      raise ValueError(f"v_half must be positive, got {self._v_half}.")

    self._rho: torch.Tensor | None = None
    if self._v_half is None and cfg.params.get("clip_to_rho_sat", True):
      if "rho_sat" not in tape:
        raise KeyError(
          f"clip_to_rho_sat is on but {tape_file} has no 'rho_sat' array. Regenerate "
          f"the tape with a to_tape.py that carries it, or pass "
          f"clip_to_rho_sat=False and accept an unbounded potential."
        )
      rho = _to(tape, "rho_sat")
      if rho.shape != (self._num_frames,):
        raise ValueError(
          f"rho_sat is {tuple(rho.shape)}, expected ({self._num_frames},)."
        )
      self._rho = rho

    self._prev_quad = torch.zeros(env.num_envs, device=env.device)
    self._has_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._zero = torch.zeros(env.num_envs, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    gamma: float,
    tape_file: str,
    kappa: float = 1.0,
    command_name: str = "motion",
    entity_name: str = "robot",
    p_file: str | None = None,
    clip_to_rho_sat: bool = True,
    rho_scale: float = 1.0,
    v_half: float | None = None,
  ) -> torch.Tensor:
    del tape_file, entity_name, p_file, clip_to_rho_sat, v_half  # Resolved at init.
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    k = command.time_steps.clamp(max=self._num_frames - 1)

    error = tangent_state_error(env, self._entity, self._qpos_ref[k], self._qvel_ref[k])
    if self._time_varying:
      quad = torch.einsum("bi,bij,bj->b", error, self._P[k], error)
    else:
      quad = torch.einsum("bi,ij,bj->b", error, self._P, error)

    # Squash the quadratic into the potential. Both forms bound it -- an unbounded
    # potential lets the (1 - gamma) residual pay for leaving the reference -- but they
    # bound it differently. The hard clip is flat past the threshold, so a policy that
    # lives entirely outside it sees a constant potential and gets no gradient at all.
    # The saturating form keeps a gradient everywhere, steepest around v_half.
    if self._v_half is not None:
      quad = quad / (quad + self._v_half)
    elif self._rho is not None:
      quad = torch.minimum(quad, rho_scale * self._rho[k])

    # F = gamma * Phi(s') - Phi(s) = kappa * (s^T P s - gamma * s'^T P s'), with the
    # previous step's quadratic standing in for s. Zero on the first step of an episode,
    # where there is no previous state to have descended from.
    shaping = kappa * (self._prev_quad - gamma * quad)
    out = torch.where(self._has_prev, shaping, self._zero)

    self._prev_quad = quad
    self._has_prev.fill_(True)
    return out

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._has_prev[env_ids] = False


class lqr_lyapunov_decrease:
  """Reward the discrete-time Lyapunov condition ``V' <= (1 - alpha) * V``."""

  FORMS = ("hinge", "margin", "indicator")

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    entity_name: str = cfg.params.get("entity_name", "robot")
    self._entity = env.scene[entity_name]
    njoint = self._entity.data.joint_pos.shape[1]
    ndx = 2 * (6 + njoint)

    tape_file: str = cfg.params["tape_file"]
    p_file: str | None = cfg.params.get("p_file")
    tape = np.load(tape_file)
    source = np.load(p_file) if p_file is not None else tape
    if "P" not in source:
      raise KeyError(
        f"{p_file or tape_file} has no 'P' array; export the Riccati cost-to-go of the "
        f"same LQR solve that produced 'gain'."
      )

    def _to(src, key: str) -> torch.Tensor:
      return torch.tensor(src[key], dtype=torch.float32, device=env.device)

    P = _to(source, "P")
    if P.ndim not in (2, 3):
      raise ValueError(f"P must be 2- or 3-dimensional, got {P.ndim} dimensions.")
    if P.shape[-2:] != (ndx, ndx):
      raise ValueError(
        f"P is {tuple(P.shape)}, expected ({ndx}, {ndx}) or (frames, {ndx}, {ndx})."
      )
    # The Riccati solution is symmetric; enforce it so an asymmetric export cannot
    # make the quadratic form disagree with the cost it represents.
    self._P = 0.5 * (P + P.transpose(-1, -2))
    self._time_varying = P.ndim == 3

    qpos_ref, qvel_ref = _to(tape, "ref_qpos"), _to(tape, "ref_qvel")
    if qpos_ref.shape[1] != 7 + njoint or qvel_ref.shape[1] != 6 + njoint:
      raise ValueError(
        f"Tape reference is {qpos_ref.shape[1]}/{qvel_ref.shape[1]} wide, expected "
        f"{7 + njoint}/{6 + njoint} for a floating base with {njoint} joints."
      )
    if self._time_varying and self._P.shape[0] != qpos_ref.shape[0]:
      raise ValueError(
        f"P has {self._P.shape[0]} frames but the tape reference has "
        f"{qpos_ref.shape[0]}; they must be on the same grid."
      )
    self._qpos_ref, self._qvel_ref = qpos_ref, qvel_ref
    self._num_frames = qpos_ref.shape[0]

    self._form: str = cfg.params.get("form", "hinge")
    if self._form not in self.FORMS:
      raise ValueError(f"form must be one of {self.FORMS}, got {self._form!r}.")
    # Dividing the margin by V makes the term dimensionless and independent of how P
    # is scaled, so alpha is the only knob that sets the required contraction rate.
    self._normalize: bool = cfg.params.get("normalize", True)
    self._eps: float = cfg.params.get("eps", 1e-6)
    if self._eps <= 0.0:
      raise ValueError(f"eps must be positive, got {self._eps}.")

    self._prev_quad = torch.zeros(env.num_envs, device=env.device)
    self._has_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._zero = torch.zeros(env.num_envs, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    tape_file: str,
    alpha: float = 0.01,
    clip: float = 1.0,
    command_name: str = "motion",
    entity_name: str = "robot",
    p_file: str | None = None,
    form: str = "hinge",
    normalize: bool = True,
    eps: float = 1e-6,
  ) -> torch.Tensor:
    del tape_file, entity_name, p_file, form, normalize, eps  # Resolved at init.
    if not 0.0 < alpha <= 1.0:
      raise ValueError(f"alpha must lie in (0, 1], got {alpha}.")
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    k = command.time_steps.clamp(max=self._num_frames - 1)

    error = tangent_state_error(env, self._entity, self._qpos_ref[k], self._qvel_ref[k])
    if self._time_varying:
      quad = torch.einsum("bi,bij,bj->b", error, self._P[k], error)
    else:
      quad = torch.einsum("bi,ij,bj->b", error, self._P, error)

    # Margin of the Lyapunov decrease condition: >= 0 iff V' <= (1 - alpha) * V.
    margin = (1.0 - alpha) * self._prev_quad - quad
    if self._normalize:
      margin = margin / (self._prev_quad + self._eps)

    if self._form == "hinge":
      out = -torch.clamp(-margin, min=0.0, max=clip)
    elif self._form == "margin":
      out = torch.clamp(margin, min=-clip, max=clip)
    else:
      out = (margin >= 0.0).to(margin.dtype)

    # Zero on the first step of an episode: no previous V to have contracted from.
    out = torch.where(self._has_prev, out, self._zero)

    self._prev_quad = quad
    self._has_prev.fill_(True)
    return out

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._has_prev[env_ids] = False
