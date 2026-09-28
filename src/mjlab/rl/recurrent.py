"""Inference helpers for recurrent (RNN) policies."""

from typing import Any, Protocol

import torch


class _HasEpisodeLength(Protocol):
  episode_length_buf: torch.Tensor


class ResetOnEpisodeStart:
  """Wrap a recurrent inference policy so its hidden state is zeroed for every env that
  starts a new episode -- what PPO does with ``actor.reset(dones)`` during training. The
  viewers call ``policy(obs)`` in their own step loop and call ``policy.reset()`` only
  on a MANUAL full reset, so under auto-reset a fallen env's memory would otherwise leak
  into its next episode. Every reset path (auto-reset inside ``step``, a manual full
  reset, a partial ``reset(env_ids)`` from the GUI) zeroes ``episode_length_buf`` for
  the envs it reset, and the buffer only increments inside ``step``, so
  ``episode_length_buf == 0`` at call time marks exactly the envs about to take the
  first action of a fresh episode.
  """

  def __init__(self, policy: Any, env: _HasEpisodeLength):
    self.policy = policy
    self._env = env

  def __call__(self, obs: Any) -> torch.Tensor:
    fresh = self._env.episode_length_buf == 0
    if bool(fresh.any()):
      self.policy.reset(fresh)
    return self.policy(obs)

  def reset(self, dones: torch.Tensor | None = None) -> None:
    self.policy.reset(dones)

  def __getattr__(self, name: str) -> Any:
    return getattr(self.policy, name)  # e.g. is_recurrent, input_size


def wrap_for_inference(policy: Any, env: _HasEpisodeLength) -> Any:
  """``policy`` wrapped in :class:`ResetOnEpisodeStart` if recurrent, else unchanged."""
  if getattr(policy, "is_recurrent", False):
    return ResetOnEpisodeStart(policy, env)
  return policy
