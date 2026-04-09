from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
  unitree_g1_flat_env_cfg_custom,
  unitree_g1_flat_env_cfg_unitree_rl_gym,
)
from .rl_cfg import unitree_g1_ppo_runner_cfg, unitree_g1_ppo_runner_cfg_custom

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-G1",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


#########################################################################
# CUSTOM CONFIGS
#########################################################################


register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-Custom",
  env_cfg=unitree_g1_flat_env_cfg_custom(),
  play_env_cfg=unitree_g1_flat_env_cfg_custom(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-G1-UnitreeRlGym",
  env_cfg=unitree_g1_flat_env_cfg_unitree_rl_gym(),
  play_env_cfg=unitree_g1_flat_env_cfg_unitree_rl_gym(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg_custom(),
  runner_cls=VelocityOnPolicyRunner,
)
