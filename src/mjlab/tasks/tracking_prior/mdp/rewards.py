"""Rewards for the tracking task with a control prior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionActionWithPrior

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg


class action_prior_deviation:
  """Penalize ``||pi(o) - u_prior||^2``, faded out as the prior weight goes to zero.

  While ``lam`` is large the policy's own output barely reaches the sim, so nothing
  shapes it and the handoff at ``lam = 0`` is a step change. This pulls ``pi(o)``
  onto the prior exactly while the prior is the thing driving: the penalty is scaled
  by ``(lam / (1 + lam))**power``, the prior's share of the command, so it decays with
  the same curriculum that anneals ``lam`` and is identically zero once ``lam`` is.

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
