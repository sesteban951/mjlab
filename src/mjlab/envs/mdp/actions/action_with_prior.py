"""Joint position actions that carry a full-state control prior.

Two terms share one prior plumbing:

* :class:`JointPositionActionWithPrior` evaluates the prior once per control step and
  combines it with the policy target as ``cfg.blend`` selects: ``"nominal"``
  (``u = pi(o)``, prior published for rewards only), ``"convex"``
  (``u = (1 - lam) * pi(o) + lam * u_prior``) or ``"residual"``
  (``u = u_prior + pi(o)``).
* :class:`JointPositionPriorReplayAction` applies the prior instead, at its own rate
  inside the decimation loop, for scoring a controller with no policy in the loop.

A prior reads the environment directly, so it has access to the full simulator state,
independent of what the actor observation group exposes to the policy.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, runtime_checkable

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@runtime_checkable
class JointPriorTerm(Protocol):
  """Protocol for a class-based prior.

  Class-based priors are instantiated once with ``(cfg, env, term)`` where ``term``
  is the owning action term. Use it to resolve joint indices once via
  ``term.target_ids`` / ``term.target_names`` so that ``__call__`` returns targets in
  the term's joint order.
  """

  def __init__(
    self,
    cfg: JointPriorActionCfg,
    env: ManagerBasedRlEnv,
    term: JointPriorAction,
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
class JointPriorActionCfg(JointPositionActionCfg):
  """Shared configuration for the joint position actions that carry a prior."""

  prior: Callable[..., torch.Tensor] | type[JointPriorTerm] = default_joint_pos_prior
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
  """Keyword arguments forwarded to ``prior`` on every evaluation."""

  @classmethod
  def from_joint_position_cfg(cls, cfg: JointPositionActionCfg, **kwargs: Any):
    """Rebuild a plain joint-position action cfg as this class, keeping every field.

    Lets a task reuse an upstream action config (scale, actuator names, offsets)
    without restating it, so tuning the upstream one cannot silently drift.
    """
    base = {f.name: getattr(cfg, f.name) for f in fields(JointPositionActionCfg)}
    base.update(kwargs)
    return cls(**base)

  def build(self, env: ManagerBasedRlEnv) -> JointPriorAction:
    raise NotImplementedError("Use a concrete prior action config.")


@dataclass(kw_only=True)
class JointPositionActionWithPriorCfg(JointPriorActionCfg):
  """Configuration for joint position control with a prior evaluated alongside."""

  blend: Literal["convex", "residual", "nominal"] = "residual"
  """How the policy and prior are combined.

  ``"nominal"``: ``u = pi(o)``; the prior is evaluated but never applied.
  ``"convex"``: ``u = (1 - lam) * pi(o) + lam * u_prior``.
  ``"residual"``: ``u = u_prior + pi(o)``; pair with ``use_default_offset=False``.
  """

  lam: float = 5.0 / 6.0
  """Initial prior weight in ``[0, 1]`` under ``"convex"``; ignored by the other modes."""

  def build(self, env: ManagerBasedRlEnv) -> JointPositionActionWithPrior:
    return JointPositionActionWithPrior(self, env)


@dataclass(kw_only=True)
class JointPositionPriorReplayActionCfg(JointPriorActionCfg):
  """Configuration for applying the prior as the sole controller."""

  prior_frequency_hz: float | None = None
  """Rate at which the prior is re-evaluated, in Hz. Default: once per control step."""

  def build(self, env: ManagerBasedRlEnv) -> JointPositionPriorReplayAction:
    return JointPositionPriorReplayAction(self, env)


class JointPriorAction(JointPositionAction):
  """Joint position action that owns a control prior. Not used directly."""

  def __init__(self, cfg: JointPriorActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

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

  @property
  def prior(self) -> Callable[..., torch.Tensor]:
    """The prior controller instance (or function) this term evaluates."""
    return self._prior

  @property
  def prior_target(self) -> torch.Tensor:
    """Most recent prior target, in radians. Shape ``(num_envs, action_dim)``."""
    return self._prior_target

  def evaluate_prior(self) -> None:
    """Refresh :attr:`prior_target` from the prior controller."""
    prior = self._prior(self._env, **self._prior_params)
    if prior.shape != self._prior_target.shape:
      raise ValueError(
        f"Prior returned shape {tuple(prior.shape)}, expected "
        f"{tuple(self._prior_target.shape)} (num_envs, action_dim)."
      )
    self._prior_target[:] = prior

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    # The prior target is a cache, fully rewritten before it is read, so it is
    # deliberately left alone here.
    super().reset(env_ids)
    prior_reset = getattr(self._prior, "reset", None)
    if callable(prior_reset):
      prior_reset(env_ids)


class JointPositionActionWithPrior(JointPriorAction):
  """Joint position control with a privileged prior evaluated alongside the policy.

  The prior is evaluated once per control step, at the same state the policy acted on,
  and published on :attr:`prior_target` for rewards to read. ``cfg.blend`` then decides
  whether it also reaches the robot; the applied command is :attr:`blended_target`.

  Note that :class:`~mjlab.managers.action_manager.ActionManager` records the *raw
  policy output* in its ``action`` / ``prev_action`` history, so ``mdp.last_action``
  observations and action-rate rewards see ``pi(o)`` alone, not the blended command.
  """

  def __init__(self, cfg: JointPositionActionWithPriorCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    if not 0.0 <= cfg.lam <= 1.0:
      raise ValueError(f"lam must lie in [0, 1], got {cfg.lam}.")
    self._lam = float(cfg.lam)
    self._blend = cfg.blend
    self._blended_target = torch.zeros_like(self._raw_actions)

  # Properties.

  @property
  def lam(self) -> float:
    """Prior weight in ``[0, 1]``. The prior gets ``lam`` of the authority."""
    return self._lam

  @lam.setter
  def lam(self, value: float) -> None:
    value = float(value)
    if not 0.0 <= value <= 1.0:
      raise ValueError(f"lam must lie in [0, 1], got {value}.")
    self._lam = value

  @property
  def prior_weight(self) -> float:
    """``lam``, the fraction of the command coming from the prior."""
    return self._lam

  @property
  def blended_target(self) -> torch.Tensor:
    """Most recently applied target, in radians (before encoder bias)."""
    return self._blended_target

  # Methods.

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    # Same instant as the policy's own output: the rewards compare the two.
    self.evaluate_prior()

  def apply_actions(self) -> None:
    if self._blend == "nominal":
      self._blended_target[:] = self._processed_actions
    elif self._blend == "residual":
      torch.add(self._prior_target, self._processed_actions, out=self._blended_target)
    else:
      torch.lerp(
        self._processed_actions,
        self._prior_target,
        self._lam,
        out=self._blended_target,
      )

    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      self._blended_target - encoder_bias, joint_ids=self._target_ids
    )


class JointPositionPriorReplayAction(JointPriorAction):
  """Joint position control driven entirely by the prior, ignoring the policy.

  The prior is refreshed inside the decimation loop at ``cfg.prior_frequency_hz``
  (zero-order held between evaluations), so it can close a state-feedback loop at
  the physics rate rather than acting as a per-control-step feedforward reference.
  """

  def __init__(self, cfg: JointPositionPriorReplayActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

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
    self._substep = 0

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    # Restart the substep clock so the prior is always evaluated on the first
    # substep of every control step, making its phase independent of resets.
    self._substep = 0

  def apply_actions(self) -> None:
    if self._substep % self._prior_period == 0:
      self.evaluate_prior()
    self._substep += 1

    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      self._prior_target - encoder_bias, joint_ids=self._target_ids
    )
