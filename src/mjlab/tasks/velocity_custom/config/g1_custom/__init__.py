from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_23dof_flat_env_cfg,
  unitree_g1_flat_env_cfg,
)
from .rl_cfg import (
  unitree_g1_23dof_ppo_runner_cfg,
  unitree_g1_ppo_runner_cfg,
)

#########################################################
# G1 29 DOF
#########################################################

register_mjlab_task(
  task_id="G1-29Dof-Velocity-Custom",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

#########################################################
# G1 23 DOF
#########################################################

register_mjlab_task(
  task_id="G1-23Dof-Velocity-Custom",
  env_cfg=unitree_g1_23dof_flat_env_cfg(),
  play_env_cfg=unitree_g1_23dof_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
