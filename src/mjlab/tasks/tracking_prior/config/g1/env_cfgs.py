"""Unitree G1 tracking with a reference-pose control prior blended into the action.

Reuses ``Mjlab-Tracking-Flat-Unitree-G1``'s env config verbatim, then swaps its
joint-position action for the prior-blended variant and adds a curriculum that
anneals the prior weight ``lam`` to zero. Everything else -- scene, motion
command, observations, rewards, terminations -- is identical.
"""

import os
from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import (
  JointPositionActionCfg,
  JointPositionActionWithPriorCfg,
)
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking_prior import mdp

# Initial prior weight: the prior gets LAM_START / (1 + LAM_START) of the command.
LAM_START = 5.0

# Environment step at which lam reaches zero (pure policy).  so 5000 steps
# is roughly 200 iterations out of 30k -- a fast drop, meant as a warm start rather than a long-lived crutch.
LAM_ZERO_STEP = 50000

# Weight of the behavior-cloning pull toward the prior. The penalty is a sum of squared
# joint-angle errors in radians, on the same scale as the tracking rewards, and it carries
PRIOR_DEVIATION_WEIGHT = -0.2


def unitree_g1_tracking_prior_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the G1 tracking config with a reference-pose control prior."""
  cfg = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation, play=play
  )

  # MJLAB_PRIOR_TAPE points at a *_prior.npz to replay a solved control tape as the
  # prior instead of the reference pose; MJLAB_PRIOR_FEEDBACK=1 closes its LQR loop
  # (off by default -- RSI randomizes the base the gains were designed against).
  tape = os.environ.get("MJLAB_PRIOR_TAPE")
  if tape:
    prior = mdp.motion_tape_prior
    prior_params = {
      "command_name": "motion",
      "tape_file": tape,
      "feedback": os.environ.get("MJLAB_PRIOR_FEEDBACK") == "1",
    }
    print(f"[INFO] control prior: tape {tape} (feedback={prior_params['feedback']})")
  else:
    prior = mdp.motion_reference_joint_pos
    prior_params = {"command_name": "motion"}

  # Swap the joint position action for the prior-blended variant, carrying over
  # all of its fields (notably the per-joint action scale).
  old = cfg.actions["joint_pos"]
  assert isinstance(old, JointPositionActionCfg)
  kwargs = {f.name: getattr(old, f.name) for f in fields(old)}
  cfg.actions["joint_pos"] = JointPositionActionWithPriorCfg(
    **kwargs,
    prior=prior,
    prior_params=prior_params,
    # Evaluate the prior once per control step, in lockstep with the policy.
    prior_frequency_hz=None,
    # Play runs the trained policy alone; training starts leaning on the prior.
    lam=0.0 if play else LAM_START,
  )

  if not play:
    # Shape pi(o) while the prior is the thing actually driving, so it is not handed full
    # authority cold at lam = 0. Fades with the prior's share of the command, so no second
    # curriculum is needed.
    cfg.rewards["action_prior_deviation"] = RewardTermCfg(
      func=mdp.action_prior_deviation,
      params={"action_name": "joint_pos", "power": 1.0},
      weight=PRIOR_DEVIATION_WEIGHT,
    )

    cfg.curriculum["prior_blend"] = CurriculumTermCfg(
      func=mdp.action_curriculum,
      params={
        "action_name": "joint_pos",
        "attribute": "lam",
        "stages": [
          {"step": 0, "value": LAM_START},
          {"step": LAM_ZERO_STEP, "value": 0.0},
        ],
      },
    )

  return cfg
