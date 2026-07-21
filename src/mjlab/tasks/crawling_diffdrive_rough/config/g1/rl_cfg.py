"""RL config for the Unitree G1 differential-drive rough-terrain crawling task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.crawling_diffdrive.config.g1.rl_cfg import (
  unitree_g1_crawling_diffdrive_ppo_runner_cfg,
)


def unitree_g1_crawling_diffdrive_rough_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_crawling_diffdrive_ppo_runner_cfg()
  cfg.experiment_name = "g1_crawling_diffdrive_rough"
  return cfg
