"""RL config for the Unitree G1 unicycle walking task with swing-foot penalties."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.walk_unicycle.config.g1.rl_cfg import (
  unitree_g1_walk_unicycle_ppo_runner_cfg,
)


def unitree_g1_walk_unicycle_swingfoot_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_walk_unicycle_ppo_runner_cfg()
  cfg.experiment_name = "g1_walk_unicycle_swingfoot"
  return cfg
