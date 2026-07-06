"""RL configuration for the Unitree G1 crawling task (reuses the tracking PPO config)."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg


def unitree_g1_crawling_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_crawling_v1"
  return cfg
