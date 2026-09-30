"""RL config for the Unitree G1 unicycle walking LSTM task on the backpack robot."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.walk_unicycle.config.g1_lstm.rl_cfg import (
  unitree_g1_walk_unicycle_lstm_ppo_runner_cfg,
)


def unitree_g1_walk_unicycle_lstm_backpack_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The LSTM runner cfg, verbatim, under its own experiment name."""
  cfg = unitree_g1_walk_unicycle_lstm_ppo_runner_cfg()
  cfg.experiment_name = "g1_walk_unicycle_lstm_backpack"
  return cfg
