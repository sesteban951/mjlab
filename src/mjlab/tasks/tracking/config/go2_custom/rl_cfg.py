"""RL configuration for the Unitree Go2 custom tracking task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg


def unitree_go2_custom_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for the custom Go2 tracking task.

  Reuses the G1 tracking PPO setup unchanged apart from the experiment name.
  """
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "go2_tracking_custom"
  return cfg
