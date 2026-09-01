"""G1 tracking with a reference-pose control prior blended into the action."""

import os
from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import (
  JointPositionActionCfg,
  JointPositionActionWithPriorCfg,
  default_joint_pos_prior,
)
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking_prior import mdp
from mjlab.tasks.tracking_prior.config.g1.rl_cfg import (
  unitree_g1_tracking_prior_ppo_runner_cfg,
)

# Initial prior weight: prior gets LAM_START of the command, policy 1 - LAM_START.
LAM_START = 5.0 / 6.0

# Environment step at which lam reaches zero (pure policy).
LAM_ZERO_STEP = 50000

# Scale of the saturating LQR potential shaping; ~10% of the other rewards per step.
LYAPUNOV_KAPPA = 50  # 26.0

# Half-saturation point of Phi = -kappa * V / (V + v_half); near the operating point.
LYAPUNOV_V_HALF = 50.0

# Required contraction per step for V' <= (1 - alpha) V; a parked robot pays -alpha.
LYAPUNOV_ALPHA = 0.01


def unitree_g1_tracking_prior_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the G1 tracking config with a reference-pose control prior."""
  cfg = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation, play=play
  )

  # MJLAB_PRIOR_TAPE: path to a *_prior.npz control tape to use as the prior.
  tape = os.environ.get("MJLAB_PRIOR_TAPE")
  if tape == "off":
    prior = default_joint_pos_prior
    prior_params = {}
    tape = None
    print("[INFO] control prior: default joint pose (tape disabled)")
  elif tape:
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

  # MJLAB_PRIOR_BLEND picks how the policy and prior are combined.
  blend = os.environ.get("MJLAB_PRIOR_BLEND", "nominal")
  if blend not in ("convex", "residual", "nominal"):
    raise ValueError(
      f"MJLAB_PRIOR_BLEND={blend!r} is not one of 'convex', 'residual', 'nominal'."
    )
  residual = blend == "residual"

  # Swap in the prior-blended action, carrying over all fields (e.g. action scale).
  old = cfg.actions["joint_pos"]
  assert isinstance(old, JointPositionActionCfg)
  kwargs = {f.name: getattr(old, f.name) for f in fields(old)}
  if residual:
    kwargs["use_default_offset"] = False
    print("[INFO] control blend: residual (u = u_prior + pi(o); lam unused)")
  elif blend == "nominal":
    print("[INFO] control blend: nominal (u = pi(o); prior evaluated, never applied)")
  else:
    print("[INFO] control blend: convex (u = (1 - lam) * pi(o) + lam * u_prior)")
  cfg.actions["joint_pos"] = JointPositionActionWithPriorCfg(
    **kwargs,
    prior=prior,
    prior_params=prior_params,
    # Evaluate the prior once per control step, in lockstep with the policy.
    prior_frequency_hz=None,
    blend=blend,
    # Play runs the trained policy alone; training starts leaning on the prior.
    lam=0.0 if play else LAM_START,
  )

  if not play and blend == "convex":
    # Shape pi(o) while the prior is the thing actually driving.
    cfg.rewards["action_prior_deviation"] = RewardTermCfg(
      func=mdp.action_prior_deviation,
      params={"action_name": "joint_pos", "power": 1.0},
      weight=-0.2,
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

  # Behavior-cloning pull toward the prior, currently off: exp(-mean sq error / std^2).
  # cfg.rewards["action_prior_exp"] = RewardTermCfg(
  #   func=mdp.action_prior_deviation_exp,
  #   params={
  #     "action_name": "joint_pos",
  #     "std": 0.2,
  #     "fade_with_prior": False,
  #   },
  #   weight=0.2,
  # )
  # print("[INFO] prior BC: exp kernel at weight 2.0, std 0.2 rad")

  # MJLAB_PRIOR_LYAPUNOV: "tape" to read P from the prior tape, else a path to an npz.
  # if not play:
  #   if tape is None:
  #     raise ValueError(
  #       "MJLAB_PRIOR_LYAPUNOV needs MJLAB_PRIOR_TAPE set to a *_prior.npz: the shaping "
  #       "scores the tangent error against that tape's reference trajectory, which is "
  #       "the trajectory P was solved around. There is no reference to measure against "
  #       "with the default reference-pose prior."
  #     )
    # Must match the discount the policy is trained with, or the telescoping breaks.
    # gamma = unitree_g1_tracking_prior_ppo_runner_cfg().algorithm.gamma
    # params = {
    #   "tape_file": tape,
    #   "command_name": "motion",
    #   "entity_name": "robot",
    #   "gamma": gamma,
    #   "kappa": LYAPUNOV_KAPPA,
    #   # Bounds the (1 - gamma) residual so the policy cannot farm it by straying.
    #   "v_half": LYAPUNOV_V_HALF,
    # }
    # cfg.rewards["lqr_lyapunov"] = RewardTermCfg(
    #   func=mdp.lqr_lyapunov_shaping,
    #   params=params,
    #   # kappa carries the scale; leave the weight at 1 so there is one knob, not two.
    #   weight=1.0,
    # )
    # print(
    #   f"[INFO] LQR shaping: kappa {LYAPUNOV_KAPPA}, gamma {gamma}, "
    #   f"P from {params.get('p_file', tape)}"
    # )

    # Discrete-time Lyapunov condition V' <= (1 - alpha) V, hinged and normalized by V.
    # cfg.rewards["lqr_lyapunov_decrease"] = RewardTermCfg(
    #   func=mdp.lqr_lyapunov_decrease,
    #   params={
    #     "tape_file": tape,
    #     "command_name": "motion",
    #     "entity_name": "robot",
    #     "alpha": LYAPUNOV_ALPHA,
    #     "form": "hinge",
    #     "normalize": True,
    #     "clip": 1.0,
    #   },
    #   weight=3.0,
    # )

  return cfg
