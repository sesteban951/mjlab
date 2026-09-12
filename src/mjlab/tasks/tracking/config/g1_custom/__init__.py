from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  unitree_g1_custom_flat_tracking_env_cfg,
)
from .rl_cfg import (
  unitree_g1_custom_tracking_ppo_runner_cfg,
)

_env_cfg = unitree_g1_custom_flat_tracking_env_cfg(has_state_estimation=False)
_play_env_cfg = unitree_g1_custom_flat_tracking_env_cfg(
  has_state_estimation=False, play=True
)

# Action-rate history: this task once carried a single, elevated action-rate penalty (-0.5, up from
# the base -0.1) to combat high-frequency action chatter on hardware. That is now handled by the
# shared limb/waist split applied in the env builder (see custom_rewards), which keeps the torso
# heavily damped (waist -0.5) while freeing the limbs (-0.15).

register_mjlab_task(
  task_id="G1-Tracking-Custom",
  env_cfg=_env_cfg,
  play_env_cfg=_play_env_cfg,
  rl_cfg=unitree_g1_custom_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
