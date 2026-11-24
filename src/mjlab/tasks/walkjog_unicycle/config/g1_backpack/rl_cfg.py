"""RL config for the Unitree G1 walk+jog unicycle task on the backpack robot."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.walkjog_unicycle.config.g1.rl_cfg import (
  unitree_g1_walkjog_unicycle_ppo_runner_cfg,
)


def unitree_g1_walkjog_unicycle_backpack_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """The parent's runner cfg, verbatim, under its own experiment name."""
  cfg = unitree_g1_walkjog_unicycle_ppo_runner_cfg()
  cfg.experiment_name = "g1_walkjog_unicycle_backpack"
  return cfg
