from mjlab.tasks.crawling_common.runner import LibraryTrackingOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import unitree_g1_jogrunsprint_env_cfg
from .rl_cfg import unitree_g1_jogrunsprint_ppo_runner_cfg

register_mjlab_task(
  task_id="G1-JogRunSprint-Custom",
  env_cfg=unitree_g1_jogrunsprint_env_cfg(),
  play_env_cfg=unitree_g1_jogrunsprint_env_cfg(play=True),
  rl_cfg=unitree_g1_jogrunsprint_ppo_runner_cfg(),
  runner_cls=LibraryTrackingOnPolicyRunner,
)
