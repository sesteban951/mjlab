from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import unitree_go2_custom_flat_tracking_env_cfg
from .rl_cfg import unitree_go2_custom_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Go2-Tracking-Custom",
  env_cfg=unitree_go2_custom_flat_tracking_env_cfg(),
  play_env_cfg=unitree_go2_custom_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_go2_custom_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
