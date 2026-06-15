"""RL configurations for the Unitree G1 custom tracking tasks (29- and 23-DoF)."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

#########################################################
# G1 29 DOF
#########################################################


def unitree_g1_29dof_custom_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for the custom G1 (29-DoF) tracking task."""
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_29dof_tracking_custom"
  # Match unitree_rl_mjlab (mjlab uses 30_000).
  cfg.max_iterations = 30001
  return cfg


#########################################################
# G1 29 DOF
#########################################################


def unitree_g1_23dof_custom_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for the custom G1 (23-DoF) tracking task."""
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_23dof_tracking_custom"
  # Match unitree_rl_mjlab (mjlab uses 30_000).
  cfg.max_iterations = 30001
  return cfg
