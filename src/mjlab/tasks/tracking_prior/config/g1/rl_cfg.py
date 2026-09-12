"""RL config for the Unitree G1 prior-blended tracking task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg


def unitree_g1_tracking_prior_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_tracking_prior"
  cfg.wandb_group = "ICRA"
  # Policy starts at zero action, so with a residual/convex blend the initial
  # command is the prior alone -- the policy learns a correction on top of it.
  cfg.actor.zero_init_last_layer = True
  return cfg
