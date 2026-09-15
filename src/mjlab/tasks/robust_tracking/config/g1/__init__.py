"""Registration for the robust (sim2real) CLF-guided tracking task.

Guarded like ``g1_control``'s: the cfg is built at import time and ``mjlab.tasks`` imports
every task eagerly, so a missing trajectory would take the whole registry down with it.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1_control.runner import ControlTrackingOnPolicyRunner

from .env_cfgs import (
  CLF_ABLATION_ARMS,
  MOTION_FILE,
  TVLQR_EXPORT,
  unitree_g1_clf_ablation_env_cfg,
  unitree_g1_clf_tracking_env_cfg,
  unitree_g1_robust_tracking_env_cfg,
)
from .rl_cfg import (
  unitree_g1_clf_tracking_ppo_runner_cfg,
  unitree_g1_robust_tracking_ppo_runner_cfg,
)

_missing = [str(p) for p in (MOTION_FILE, TVLQR_EXPORT) if not p.exists()]
if _missing:
  print(
    f"[WARNING]: skipping G1-Robust-Tracking -- missing {_missing}. "
    "See the rebuild commands in the FileNotFoundError in env_cfgs.py; the ones for "
    "G1-Tracking-Control's jog do NOT apply to this clip."
  )
else:
  # Zero-init runner from G1-Tracking-Control: a zero action is the clip's first frame, so an
  # actor starting at zero starts at a pose the robot can hold.
  register_mjlab_task(
    task_id="G1-Robust-Tracking",
    env_cfg=unitree_g1_robust_tracking_env_cfg(),
    play_env_cfg=unitree_g1_robust_tracking_env_cfg(play=True),
    rl_cfg=unitree_g1_robust_tracking_ppo_runner_cfg(),
    runner_cls=ControlTrackingOnPolicyRunner,
  )
  register_mjlab_task(
    task_id="G1-CLF-Tracking",
    env_cfg=unitree_g1_clf_tracking_env_cfg(),
    play_env_cfg=unitree_g1_clf_tracking_env_cfg(play=True),
    rl_cfg=unitree_g1_clf_tracking_ppo_runner_cfg(),
    runner_cls=ControlTrackingOnPolicyRunner,
  )
  for arm in CLF_ABLATION_ARMS:
    register_mjlab_task(
      task_id=f"G1-CLF-Ablation-{arm}",
      env_cfg=unitree_g1_clf_ablation_env_cfg(arm),
      play_env_cfg=unitree_g1_clf_ablation_env_cfg(arm, play=True),
      rl_cfg=unitree_g1_clf_tracking_ppo_runner_cfg(),
      runner_cls=ControlTrackingOnPolicyRunner,
    )
