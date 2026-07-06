"""Crawling v2 terminations.

The tracking terminations are all reference-relative (anchor/body position & orientation error), so
the moment a stopped env lowers off the frozen crawl reference into belly rest they would fire.
``gate_moving`` wraps such a termination so it can only trigger for MOVING (non-stopped) envs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

from .commands import LibraryMotionCommand

if TYPE_CHECKING:
  import torch

  from mjlab.envs import ManagerBasedRlEnv


def gate_moving(base_func: Callable) -> Callable:
  """Wrap a termination func so it can only trigger for MOVING (non-stopped) envs.

  Called by the termination manager as ``func(env, **params)``; ``command_name`` must be in params.
  """

  def wrapped(env: ManagerBasedRlEnv, **params) -> "torch.Tensor":
    cmd = cast(LibraryMotionCommand, env.command_manager.get_term(params["command_name"]))
    return base_func(env, **params) & (~cmd.is_stopped)

  wrapped.__name__ = f"gated_{getattr(base_func, '__name__', 'termination')}"
  return wrapped
