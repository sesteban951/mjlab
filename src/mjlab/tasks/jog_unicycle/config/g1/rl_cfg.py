"""RL config for the Unitree G1 unicycle jogging task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg


def unitree_g1_jog_unicycle_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_jog_unicycle"
  return cfg
