"""Diagnostics of the policy's input against the pieces of a tape prior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPriorAction
from mjlab.tasks.tracking_prior.mdp.priors import motion_tape_prior

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.metrics_manager import MetricsTermCfg


class prior_input_kernel:
  """``exp(-||qdes_policy - target|| / sigma^2)`` against one piece of the tape prior.

  The same kernel as :class:`qdes_imitation_rbf`, evaluated as a measurement rather
  than optimized, so every arm reports what that reward *would* score against
  ``"feedforward"`` (``u_ff[k]``) or ``"closed_loop"`` (``u_ff[k] + alpha K e``).
  """

  TARGETS = ("feedforward", "closed_loop")

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    action_name: str = cfg.params.get("action_name", "joint_pos")
    term = env.action_manager.get_term(action_name)
    if not isinstance(term, JointPriorAction):
      raise TypeError(
        f"Action term '{action_name}' is a {type(term).__name__}; expected a "
        f"JointPriorAction carrying a motion_tape_prior."
      )
    if not isinstance(term.prior, motion_tape_prior):
      raise TypeError(
        f"Action term '{action_name}' has prior {type(term.prior).__name__}; only "
        f"motion_tape_prior exposes feedforward and feedback pieces."
      )
    target = cfg.params.get("target")
    if target not in self.TARGETS:
      raise ValueError(f"target must be one of {self.TARGETS}, got {target!r}.")
    if float(cfg.params.get("sigma", 0.0)) <= 0.0:
      raise ValueError(f"sigma must be positive, got {cfg.params.get('sigma')}.")
    self._term, self._prior = term, term.prior

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    target: str,
    sigma: float,
    action_name: str = "joint_pos",
    squared: bool = False,
  ) -> torch.Tensor:
    del env, action_name  # Resolved at init.
    if target == "feedforward":
      ref = self._prior.feedforward_target
    else:
      ref = self._prior.closed_loop_target
    err = torch.linalg.norm(self._term.processed_actions - ref, dim=-1)
    return torch.exp(-(err**2 if squared else err) / sigma**2)


class prior_feedback_ratio:
  """``||qdes_policy - u_ff|| / ||alpha K e||``: how much correction the policy applies.

  1 means the policy deviates from the feedforward by as much as the LQR would
  correct, 0 means it plays the feedforward. Clipped at ``clip`` since the ratio is
  unbounded where the controller's correction vanishes.
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    action_name: str = cfg.params.get("action_name", "joint_pos")
    term = env.action_manager.get_term(action_name)
    if not isinstance(term, JointPriorAction):
      raise TypeError(
        f"Action term '{action_name}' is a {type(term).__name__}; expected a "
        f"JointPriorAction carrying a motion_tape_prior."
      )
    if not isinstance(term.prior, motion_tape_prior):
      raise TypeError(
        f"Action term '{action_name}' has prior {type(term.prior).__name__}; only "
        f"motion_tape_prior exposes the feedback correction."
      )
    if float(cfg.params.get("clip", 2.0)) <= 0.0:
      raise ValueError(f"clip must be positive, got {cfg.params.get('clip')}.")
    self._term, self._prior = term, term.prior

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
    clip: float = 2.0,
    eps: float = 1e-6,
  ) -> torch.Tensor:
    del env, action_name  # Resolved at init.
    applied = torch.linalg.norm(
      self._term.processed_actions - self._prior.feedforward_target, dim=-1
    )
    wanted = torch.linalg.norm(self._prior.feedback_correction, dim=-1)
    return torch.clamp(applied / (wanted + eps), max=clip)
