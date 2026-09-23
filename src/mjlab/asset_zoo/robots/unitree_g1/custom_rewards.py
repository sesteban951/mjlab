"""Shared reward tweaks for the custom Unitree G1 environments.

Companion to ``custom_dr.py``: reward-term overrides that every custom G1 env
(tracking, contact-rich, crawling, velocity, control) layers on top of its
inherited base config.

The action-rate penalty is the main one. mjlab's built-in ``action_rate_l2``
sums over the WHOLE action vector, so it cannot weight groups of joints
differently. The custom envs instead split it into two scoped terms -- limbs
(arms + legs) and waist (the three roll/pitch/yaw DOFs) -- so the torso can be
penalized harder than the limbs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions.actions import BaseAction
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.lab_api.string import resolve_matching_names

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# The three waist DOFs. Everything else the policy commands is a limb joint
# (arms + legs), so matching on this expression and its complement partitions
# the full action vector -- no commanded joint loses its action-rate penalty.
WAIST_JOINT_EXPR: tuple[str, ...] = ("waist_.*",)


def _action_indices(
  env: ManagerBasedRlEnv,
  action_term_name: str,
  joint_expr: Sequence[str],
  invert: bool,
) -> torch.Tensor:
  """Indices into the concatenated action vector for the matching joints.

  ``target_names`` is ordered to match the term's action entries, so resolving
  the regexes against it yields action-vector indices directly (shifted by the
  term's offset when several action terms are concatenated). Cached per env,
  since the mapping is fixed after construction.
  """
  cache_key = (action_term_name, tuple(joint_expr), invert)
  cache = getattr(env, "_g1_action_rate_idx_cache", None)
  if cache is None:
    cache = {}
    env._g1_action_rate_idx_cache = cache  # type: ignore[attr-defined]
  cached = cache.get(cache_key)
  if cached is not None:
    return cached

  manager = env.action_manager
  term = manager.get_term(action_term_name)
  assert isinstance(term, BaseAction), (
    f"action term {action_term_name!r} must expose joint target names"
  )
  names = term.target_names
  matched, _ = resolve_matching_names(list(joint_expr), names)
  matched_set = set(matched)
  local = [i for i in range(len(names)) if (i in matched_set) != invert]
  if not local:
    raise ValueError(
      f"action-rate group matched no joints (expr={list(joint_expr)}, "
      f"invert={invert}) among {names}"
    )

  # Offset of this term inside the concatenated action vector.
  offset = 0
  for name in manager.active_terms:
    if name == action_term_name:
      break
    offset += manager.get_term(name).action_dim

  idx = torch.tensor([offset + i for i in local], device=env.device, dtype=torch.long)
  cache[cache_key] = idx
  return idx


def action_rate_l2_joints(
  env: ManagerBasedRlEnv,
  joint_expr: Sequence[str],
  invert: bool = False,
  action_term_name: str = "joint_pos",
) -> torch.Tensor:
  """``action_rate_l2`` restricted to a subset of the commanded joints.

  Like the built-in, this operates on the RAW policy output (before the action
  term's per-joint scale).

  Args:
    env: The environment.
    joint_expr: Regex patterns matched against the action term's joint names.
    invert: If True, penalize the joints that do NOT match ``joint_expr``.
    action_term_name: Action term whose joints are being scoped.
  """
  idx = _action_indices(env, action_term_name, joint_expr, invert)
  delta = env.action_manager.action[:, idx] - env.action_manager.prev_action[:, idx]
  return torch.sum(torch.square(delta), dim=1)


def add_custom_g1_action_rate_split(
  cfg: ManagerBasedRlEnvCfg,
  limb_weight: float = -0.1,
  waist_weight: float = -0.3,
  action_term_name: str = "joint_pos",
) -> None:
  """Replace the single ``action_rate_l2`` term with limb- and waist-scoped ones.

  The two groups partition the commanded joints: ``action_rate_l2_waist`` covers
  the three waist DOFs, ``action_rate_l2_limbs`` covers every other commanded joint
  (arms + legs). A heavier waist weight keeps the torso quiet while leaving the
  limbs free to move.

  Args:
    cfg: An already-built env config containing an ``action_rate_l2`` reward.
    limb_weight: Weight for the arm + leg action-rate penalty.
    waist_weight: Weight for the waist action-rate penalty.
    action_term_name: Action term whose joints are being scoped.
  """
  cfg.rewards.pop("action_rate_l2", None)
  common = {"joint_expr": WAIST_JOINT_EXPR, "action_term_name": action_term_name}
  cfg.rewards["action_rate_l2_limbs"] = RewardTermCfg(
    func=action_rate_l2_joints,
    weight=limb_weight,
    params={**common, "invert": True},
  )
  cfg.rewards["action_rate_l2_waist"] = RewardTermCfg(
    func=action_rate_l2_joints,
    weight=waist_weight,
    params={**common, "invert": False},
  )
