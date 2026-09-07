"""RL configuration for the Unitree G1 CLF-RL tracking task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1_custom.rl_cfg import (
  unitree_g1_custom_tracking_ppo_runner_cfg,
)


def unitree_g1_control_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """G1-Tracking-Custom's PPO config under its own experiment name."""
  cfg = unitree_g1_custom_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_tracking_control"
  return cfg


def unitree_g1_control_baseline_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The baseline arm's PPO config: identical apart from the experiment name.

  Same hyperparameters on purpose -- if the two arms differed in PPO as well as in the reward,
  a difference in outcome would not be attributable to either.
  """
  cfg = unitree_g1_custom_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_tracking_control_baseline"
  return cfg
