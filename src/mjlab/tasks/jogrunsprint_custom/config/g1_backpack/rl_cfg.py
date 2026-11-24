"""RL config for the Unitree G1 jog + run + sprint task on the backpack robot."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.jogrunsprint_custom.config.g1.rl_cfg import (
  unitree_g1_jogrunsprint_ppo_runner_cfg,
)


def unitree_g1_jogrunsprint_backpack_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The parent's runner cfg, verbatim, under its own experiment name."""
  cfg = unitree_g1_jogrunsprint_ppo_runner_cfg()
  cfg.experiment_name = "g1_jogrunsprint_custom_backpack"
  return cfg
