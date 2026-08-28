"""Joint position action blended with a full-state control prior.

The applied joint-position target is a convex combination of the policy's
target and a prior controller's target::

    u = 1 / (1 + lam) * pi(o) + lam / (1 + lam) * u_prior(s)

Both terms are in radians (the policy's raw action is mapped to radians by the
usual ``scale``/``offset`` of :class:`JointPositionAction`, so the blend needs no
extra unit conversion). ``lam >= 0`` weights the prior: ``lam = 0`` is the pure
policy, ``lam = 5`` gives the prior 5/6 of the authority, ``lam -> inf`` is the
pure prior. Anneal it during training with
:func:`mjlab.envs.mdp.curriculums.action_curriculum`.

The prior is evaluated *inside the decimation loop* at its own rate
(``prior_frequency_hz``, zero-order held between evaluations), so it can be a
genuine state-feedback controller rather than a per-control-step feedforward
reference. It reads the environment directly and therefore has access to the
full simulator state, independent of what the actor observation group exposes to
the policy.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, runtime_checkable

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@runtime_checkable
class JointPriorTerm(Protocol):
  """Protocol for a class-based prior.

  Class-based priors are instantiated once with ``(cfg, env, term)`` where
  ``term`` is the owning :class:`JointPositionActionWithPrior`. Use it to resolve
  joint indices once via ``term.target_ids`` / ``term.target_names`` so that
  ``__call__`` returns targets in the term's joint order.
  """

  def __init__(
    self,
    cfg: JointPositionActionWithPriorCfg,
    env: ManagerBasedRlEnv,
    term: JointPositionActionWithPrior,
  ) -> None: ...

  def __call__(self, env: ManagerBasedRlEnv, **kwargs: Any) -> torch.Tensor: ...


def default_joint_pos_prior(
  env: ManagerBasedRlEnv,
  entity_name: str,
  joint_ids: torch.Tensor,
) -> torch.Tensor:
  """Trivial prior that holds the entity's default joint positions.

  Useful as a smoke-test prior and as a "stand still" regularizer. ``joint_ids``
  is injected by the action term; do not set it in ``prior_params``.
  """
  return env.scene[entity_name].data.default_joint_pos[:, joint_ids]


@dataclass(kw_only=True)
class JointPositionActionWithPriorCfg(JointPositionActionCfg):
  """Configuration for joint position control blended with a control prior."""

  prior: Callable[..., torch.Tensor] = default_joint_pos_prior
  """The prior controller.

  Either a function ``f(env, **prior_params) -> (num_envs, action_dim)`` or a
  class following :class:`JointPriorTerm`, instantiated once with
  ``(cfg, env, term)``. Must return joint-position targets in **radians**,
  ordered to match the action term's ``target_names``.

  The prior is called with the environment, so it may read any privileged state
  (``env.scene[...].data``, ``env.command_manager``, sensors). It is *not*
  restricted to the actor observation group.
  """

  prior_params: dict[str, Any] = field(default_factory=dict)
  """Keyword arguments forwarded to ``prior`` on every evaluation.

  ``SceneEntityCfg`` values are resolved against the scene at init. If the prior
  declares a ``joint_ids`` or ``entity_name`` parameter and does not get one
  here, the action term injects its own resolved values, so a prior can gather
  the term's joints without re-deriving the ordering."""

  prior_frequency_hz: float | None = None
  """Rate at which the prior is re-evaluated, in Hz.

  ``None`` (the default) evaluates it once per control step, in lockstep with
  the policy. A higher rate re-evaluates it inside the decimation loop, which
  only matters for a state-feedback prior; a feedforward prior gains nothing
  from it. Must divide the physics rate (``1 / sim.mujoco.timestep``) into an
  integer number of substeps, and cannot be slower than the control rate. With
  the default 200 Hz physics and 50 Hz control, ``100.0`` runs the prior on
  every 2nd substep and ``200.0`` on every substep."""

  lam: float = 5.0
  """Initial prior weight ``lam >= 0``. The prior gets ``lam / (1 + lam)`` of the
  authority and the policy ``1 / (1 + lam)``. Annealed by
  :func:`mjlab.envs.mdp.curriculums.action_curriculum`."""

  blend: Literal["convex", "residual"] = "convex"
  """How the policy and prior are combined.

  ``"convex"``: ``u = 1 / (1 + lam) * pi(o) + lam / (1 + lam) * u_prior``.

  ``"residual"``: ``u = u_prior + pi(o)``, with ``lam`` unused -- the policy learns a
  correction on top of the prior rather than competing with it for authority. Set
  ``use_default_offset=False`` alongside it, or the default pose is added on top of
  the prior's target and the residual is biased by a whole standing posture."""

  def build(self, env: ManagerBasedRlEnv) -> JointPositionActionWithPrior:
    return JointPositionActionWithPrior(self, env)


class JointPositionActionWithPrior(JointPositionAction):
  """Joint position control blended with a full-state prior controller.

  The blended target is recomputed on every decimation substep. The policy's
  contribution is held constant across the control step (it is only updated in
  :meth:`process_actions`), while the prior is refreshed at
  ``cfg.prior_frequency_hz``.

  Note that :class:`~mjlab.managers.action_manager.ActionManager` records the
  *raw policy output* in its ``action`` / ``prev_action`` history, so
  ``mdp.last_action`` observations and action-rate rewards see ``pi(o)`` alone,
  not the blended command. Read :attr:`blended_target` if you need the latter.
  """

  def __init__(self, cfg: JointPositionActionWithPriorCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    if cfg.lam < 0.0:
      raise ValueError(f"lam must be non-negative, got {cfg.lam}.")
    self._lam = float(cfg.lam)

    physics_hz = 1.0 / env.physics_dt
    if cfg.prior_frequency_hz is None:
      # Lockstep with the policy: evaluate once per control step.
      self._prior_period = env.cfg.decimation
    else:
      if cfg.prior_frequency_hz <= 0.0:
        raise ValueError(
          f"prior_frequency_hz must be positive, got {cfg.prior_frequency_hz}."
        )
      ratio = physics_hz / cfg.prior_frequency_hz
      self._prior_period = int(round(ratio))
      if self._prior_period < 1 or abs(ratio - self._prior_period) > 1e-6:
        raise ValueError(
          f"prior_frequency_hz={cfg.prior_frequency_hz} does not evenly divide "
          f"the physics rate of {physics_hz} Hz (ratio {ratio}). Choose a rate "
          f"of the form physics_hz / k for integer k >= 1."
        )
      if self._prior_period > env.cfg.decimation:
        raise ValueError(
          f"prior_frequency_hz={cfg.prior_frequency_hz} is slower than the "
          f"control rate of {physics_hz / env.cfg.decimation} Hz, so the prior "
          f"would not be re-evaluated every control step."
        )

    # Resolve scene entity configs in the prior params, mirroring how the
    # managers treat term params.
    for value in cfg.prior_params.values():
      if isinstance(value, SceneEntityCfg):
        value.resolve(env.scene)
    self._prior_params = dict(cfg.prior_params)
    # Inject the resolved joint indices and entity name for priors that declare
    # them, so a prior does not have to re-derive the term's joint ordering.
    injected = {"joint_ids": self._target_ids, "entity_name": cfg.entity_name}
    call = cfg.prior.__call__ if inspect.isclass(cfg.prior) else cfg.prior
    accepted = inspect.signature(call).parameters
    for key, value in injected.items():
      if key in accepted and key not in self._prior_params:
        self._prior_params[key] = value

    if inspect.isclass(cfg.prior):
      self._prior = cfg.prior(cfg=cfg, env=env, term=self)
    else:
      self._prior = cfg.prior

    self._prior_target = torch.zeros_like(self._raw_actions)
    self._blended_target = torch.zeros_like(self._raw_actions)
    self._substep = 0

  # Properties.

  @property
  def lam(self) -> float:
    """Prior weight. The prior gets ``lam / (1 + lam)`` of the authority."""
    return self._lam

  @lam.setter
  def lam(self, value: float) -> None:
    value = float(value)
    if value < 0.0:
      raise ValueError(f"lam must be non-negative, got {value}.")
    self._lam = value

  @property
  def prior_weight(self) -> float:
    """``lam / (1 + lam)``, the fraction of the command coming from the prior."""
    return self._lam / (1.0 + self._lam)

  @property
  def prior_target(self) -> torch.Tensor:
    """Most recent prior target, in radians. Shape ``(num_envs, action_dim)``."""
    return self._prior_target

  @property
  def blended_target(self) -> torch.Tensor:
    """Most recently applied blended target, in radians (before encoder bias)."""
    return self._blended_target

  # Methods.

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    # Restart the substep clock so the prior is always evaluated on the first
    # substep of every control step, making its phase independent of resets.
    self._substep = 0

  def apply_actions(self) -> None:
    if self._substep % self._prior_period == 0:
      prior = self._prior(self._env, **self._prior_params)
      if prior.shape != self._prior_target.shape:
        raise ValueError(
          f"Prior returned shape {tuple(prior.shape)}, expected "
          f"{tuple(self._prior_target.shape)} (num_envs, action_dim)."
        )
      self._prior_target[:] = prior
    self._substep += 1

    if self.cfg.blend == "residual":
      torch.add(self._prior_target, self._processed_actions, out=self._blended_target)
    else:
      policy_weight = 1.0 / (1.0 + self._lam)
      torch.lerp(
        self._processed_actions,
        self._prior_target,
        1.0 - policy_weight,
        out=self._blended_target,
      )

    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      self._blended_target - encoder_bias, joint_ids=self._target_ids
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    # The target buffers are caches, fully rewritten on the first substep of
    # every control step before they are read, so they are deliberately left
    # alone here: zeroing them would only make the accessors read as zero for
    # anything inspecting them between a reset and the next step.
    super().reset(env_ids)
    prior_reset = getattr(self._prior, "reset", None)
    if callable(prior_reset):
      prior_reset(env_ids)
