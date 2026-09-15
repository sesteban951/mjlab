"""RL configuration for the Unitree G1 robust (sim2real) CLF-guided tracking task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1_custom.rl_cfg import (
  unitree_g1_custom_tracking_ppo_runner_cfg,
)


def unitree_g1_robust_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """G1-Tracking-Custom's PPO config under its own experiment name."""
  cfg = unitree_g1_custom_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_robust_tracking"
  return cfg


def unitree_g1_clf_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """G1-Robust-Tracking's PPO config under its own experiment name."""
  cfg = unitree_g1_robust_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_clf_tracking"
  cfg.wandb_group = "ICRA"
  return cfg
