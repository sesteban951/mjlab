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
from mjlab.terrains import BoxObstacleTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg

# Initial prior weight: prior gets LAM_START of the command, policy 1 - LAM_START.
LAM_START = 5.0 / 6.0

# Environment step at which lam reaches zero (pure policy). Overridable via
# MJLAB_LAM_ZERO_STEP for one-off runs without touching the default schedule.
LAM_ZERO_STEP = int(os.environ.get("MJLAB_LAM_ZERO_STEP", 2500))

# Scale of the saturating LQR potential shaping; ~10% of the other rewards per step.
LYAPUNOV_KAPPA = 50  # 26.0

# Half-saturation point of Phi = -kappa * V / (V + v_half); near the operating point.
LYAPUNOV_V_HALF = 50.0

# Required contraction per step for V' <= (1 - alpha) V; a parked robot pays -alpha.
LYAPUNOV_ALPHA = 0.01

# RBF widths and weights of the CLF decrease and qdes imitation kernels.
CLF_DECREASE_SIGMA = 0.5
CLF_DECREASE_WEIGHT = 1.0
QDES_IMITATION_SIGMA = 3.0
QDES_IMITATION_WEIGHT = 1.0


def unitree_g1_tracking_prior_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
  kappa: float = LYAPUNOV_KAPPA,
  v_half: float = LYAPUNOV_V_HALF,
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
  blend = os.environ.get("MJLAB_PRIOR_BLEND", "residual")
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
    blend=blend,
    # Play runs the trained policy alone; training starts leaning on the prior.
    lam=0.0 if play else LAM_START,
  )

  if not play and blend == "convex":
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

  # MJLAB_PRIOR_REWARD selects at most one prior-linked reward term, so each can be
  # ablated one at a time against the same blend + curriculum setup:
  #   "input"        -> action_prior_deviation: penalize ||pi(o) - u_prior||^2, faded
  #                      by lam**power. Only meaningful under blend="convex".
  #   "lyapunov"      -> lqr_lyapunov_shaping: potential-based LQR cost-to-go shaping.
  #   "lyapunov_dec"  -> lqr_lyapunov_decrease: discrete Lyapunov decrease condition.
  #   "clf_rbf"       -> clf_decrease_rbf: RBF on the same decrease violation.
  #   "qdes_rbf"      -> qdes_imitation_rbf: RBF on ||qdes_policy - qdes_prior||.
  #   "none" (default) -> none of the above.
  reward_mode = os.environ.get("MJLAB_PRIOR_REWARD", "none")
  if reward_mode not in (
    "none",
    "input",
    "lyapunov",
    "lyapunov_dec",
    "clf_rbf",
    "qdes_rbf",
  ):
    raise ValueError(
      f"MJLAB_PRIOR_REWARD={reward_mode!r} is not one of 'none', 'input', "
      f"'lyapunov', 'lyapunov_dec', 'clf_rbf', 'qdes_rbf'."
    )

  if reward_mode == "input":
    if not play and blend == "convex":
      # Shape pi(o) while the prior is the thing actually driving.
      cfg.rewards["action_prior_deviation"] = RewardTermCfg(
        func=mdp.action_prior_deviation,
        params={"action_name": "joint_pos", "power": 1.0},
        weight=-0.2,
      )
      print("[INFO] prior reward: action_prior_deviation (weight -0.2)")
    else:
      print(
        "[INFO] MJLAB_PRIOR_REWARD=input has no effect outside training with "
        "blend='convex'."
      )

  elif reward_mode == "lyapunov":
    if play:
      print("[INFO] MJLAB_PRIOR_REWARD=lyapunov has no effect during play.")
    else:
      if tape is None:
        raise ValueError(
          "MJLAB_PRIOR_REWARD=lyapunov needs MJLAB_PRIOR_TAPE set to a *_prior.npz: "
          "the shaping scores the tangent error against that tape's reference "
          "trajectory, which is the trajectory P was solved around. There is no "
          "reference to measure against with the default reference-pose prior."
        )
      # Must match the discount the policy is trained with, or the telescoping breaks.
      gamma = unitree_g1_tracking_prior_ppo_runner_cfg().algorithm.gamma
      params = {
        "tape_file": tape,
        "command_name": "motion",
        "entity_name": "robot",
        "gamma": gamma,
        "kappa": kappa,
        # Bounds the (1 - gamma) residual so the policy cannot farm it by straying.
        "v_half": v_half,
      }
      cfg.rewards["lqr_lyapunov"] = RewardTermCfg(
        func=mdp.lqr_lyapunov_shaping,
        params=params,
        # kappa carries the scale; leave the weight at 1 so there is one knob, not two.
        weight=1.0,
      )
      print(
        f"[INFO] prior reward: lqr_lyapunov_shaping (kappa {kappa}, "
        f"gamma {gamma}, P from {params.get('p_file', tape)})"
      )

  elif reward_mode == "lyapunov_dec":
    if play:
      print("[INFO] MJLAB_PRIOR_REWARD=lyapunov_dec has no effect during play.")
    else:
      if tape is None:
        raise ValueError(
          "MJLAB_PRIOR_REWARD=lyapunov_dec needs MJLAB_PRIOR_TAPE set to a "
          "*_prior.npz: the Lyapunov condition is scored against that tape's "
          "reference trajectory."
        )
      cfg.rewards["lqr_lyapunov_decrease"] = RewardTermCfg(
        func=mdp.lqr_lyapunov_decrease,
        params={
          "tape_file": tape,
          "command_name": "motion",
          "entity_name": "robot",
          "alpha": LYAPUNOV_ALPHA,
          "form": "hinge",
          "normalize": True,
          "clip": 1.0,
        },
        weight=3.0,
      )
      print(f"[INFO] prior reward: lqr_lyapunov_decrease (alpha {LYAPUNOV_ALPHA})")

  elif reward_mode == "clf_rbf":
    if play:
      print("[INFO] MJLAB_PRIOR_REWARD=clf_rbf has no effect during play.")
    else:
      if tape is None:
        raise ValueError(
          "MJLAB_PRIOR_REWARD=clf_rbf needs MJLAB_PRIOR_TAPE set to a *_prior.npz: "
          "the decrease violation is scored against that tape's reference trajectory."
        )
      clf_sigma = float(os.environ.get("MJLAB_CLF_SIGMA", CLF_DECREASE_SIGMA))
      cfg.rewards["clf_decrease"] = RewardTermCfg(
        func=mdp.clf_decrease_rbf,
        params={
          "tape_file": tape,
          "command_name": "motion",
          "entity_name": "robot",
          "alpha": LYAPUNOV_ALPHA,
          "sigma": clf_sigma,
          "normalize": True,
        },
        weight=CLF_DECREASE_WEIGHT,
      )
      print(
        f"[INFO] prior reward: clf_decrease_rbf (sigma {clf_sigma}, "
        f"alpha {LYAPUNOV_ALPHA})"
      )

  elif reward_mode == "qdes_rbf":
    if play:
      print("[INFO] MJLAB_PRIOR_REWARD=qdes_rbf has no effect during play.")
    else:
      qdes_sigma = float(os.environ.get("MJLAB_QDES_SIGMA", QDES_IMITATION_SIGMA))
      cfg.rewards["qdes_imitation"] = RewardTermCfg(
        func=mdp.qdes_imitation_rbf,
        params={"action_name": "joint_pos", "sigma": qdes_sigma},
        weight=QDES_IMITATION_WEIGHT,
      )
      print(f"[INFO] prior reward: qdes_imitation_rbf (sigma {qdes_sigma})")

  # Behavior-cloning pull toward the prior, not part of the MJLAB_PRIOR_REWARD switch
  # above: exp(-mean sq error / std^2).
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

  return cfg


# box_rolldown: the clip's box under every robot, at the solve's offset.
BOX_ROLLDOWN_SIZE = (1.20, 1.22, 0.41)
BOX_ROLLDOWN_OFFSET = (0.0, 0.41)
BOX_ROLLDOWN_PATCH_SIZE = (8.0, 8.0)
BOX_ROLLDOWN_KAPPA = 23.0
BOX_ROLLDOWN_V_HALF = 95.0

# sideroll: stand, roll sideways 2.1 m, stand up; solved on bare ground.
SIDEROLL_KAPPA = 47.0
SIDEROLL_V_HALF = 17.8


def box_rolldown_terrain_cfg(
  num_rows: int = 10, num_cols: int = 10
) -> TerrainEntityCfg:
  """Grid of flat patches, each carrying the clip's box at the solve's offset."""
  return TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      size=BOX_ROLLDOWN_PATCH_SIZE,
      num_rows=num_rows,
      num_cols=num_cols,
      border_width=0.0,
      sub_terrains={
        "box_rolldown": BoxObstacleTerrainCfg(
          box_size=BOX_ROLLDOWN_SIZE,
          box_offset=BOX_ROLLDOWN_OFFSET,
        )
      },
    ),
  )


def unitree_g1_box_rolldown_prior_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """The prior task on box_rolldown, with the clip's box under every robot."""
  cfg = unitree_g1_tracking_prior_env_cfg(
    has_state_estimation=has_state_estimation,
    play=play,
    kappa=float(os.environ.get("MJLAB_LYAP_KAPPA", BOX_ROLLDOWN_KAPPA)),
    v_half=float(os.environ.get("MJLAB_LYAP_V_HALF", BOX_ROLLDOWN_V_HALF)),
  )
  cfg.scene.terrain = box_rolldown_terrain_cfg()
  return cfg


def unitree_g1_sideroll_prior_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """The prior task on the sideroll clip, on flat ground."""
  return unitree_g1_tracking_prior_env_cfg(
    has_state_estimation=has_state_estimation,
    play=play,
    kappa=float(os.environ.get("MJLAB_LYAP_KAPPA", SIDEROLL_KAPPA)),
    v_half=float(os.environ.get("MJLAB_LYAP_V_HALF", SIDEROLL_V_HALF)),
  )
