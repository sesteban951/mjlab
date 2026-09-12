"""Runner for the CLF-RL tracking task: zero the actor's output layer at construction.

WHY. With the action offset re-centred on the trajectory (see ``tvlqr.reference_offset``), a zero
action commands the gait's own mean posture -- so an actor that outputs zero starts already close
to ``qdes_ctrl`` and the imitation reward starts HIGH instead of at its floor. That turns the
model-based controller into the policy's initialization rather than a distant target it has to
find by exploration.

ONLY THE OUTPUT LAYER. Zeroing every layer makes all hidden units identical and their gradients
identical, so the network can never break symmetry and never learns -- the standard failure of
"initialize to zeros". Zeroing the last linear layer alone gives an exactly-zero mean action while
leaving the hidden layers at their normal init, which is trainable from the first step. The policy
still explores: PPO's action noise is a separate learned/​configured std, untouched here.
"""

from __future__ import annotations

import torch
from rsl_rl.env import VecEnv

from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner


def _zero_output_layer(module: torch.nn.Module) -> torch.nn.Linear | None:
  """Zero the weight and bias of the LAST nn.Linear inside `module`. Returns it, or None."""
  last: torch.nn.Linear | None = None
  for m in module.modules():
    if isinstance(m, torch.nn.Linear):
      last = m
  if last is not None:
    with torch.no_grad():
      last.weight.zero_()
      if last.bias is not None:
        last.bias.zero_()
  return last


class ControlTrackingOnPolicyRunner(MotionTrackingOnPolicyRunner):
  """MotionTrackingOnPolicyRunner with a zero-initialized actor output layer."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device, registry_name)
    # get_policy() is rsl_rl's own accessor for the raw actor module (PPO._raw_actor); reaching
    # for .policy.actor instead depends on an internal layout that has already moved once.
    layer = _zero_output_layer(self.alg.get_policy())
    if layer is None:
      print("[WARNING]: found no nn.Linear in the actor; skipping the zero-init.")
    else:
      print(
        f"[INFO]: zeroed the actor's output layer "
        f"({layer.out_features}x{layer.in_features}), so the initial "
        f"mean action is exactly 0 and qdes_policy starts at the trajectory's mean posture."
      )
