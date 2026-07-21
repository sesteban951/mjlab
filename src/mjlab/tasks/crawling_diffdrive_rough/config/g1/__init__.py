from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import unitree_g1_crawling_diffdrive_rough_env_cfg
from .rl_cfg import unitree_g1_crawling_diffdrive_rough_ppo_runner_cfg

register_mjlab_task(
  task_id="G1-Crawling-DiffDrive-Rough",
  env_cfg=unitree_g1_crawling_diffdrive_rough_env_cfg(),
  play_env_cfg=unitree_g1_crawling_diffdrive_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_crawling_diffdrive_rough_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
