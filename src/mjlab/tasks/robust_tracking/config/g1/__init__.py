"""Registration for the robust (sim2real) CLF-guided tracking task.

REGISTRATION IS GUARDED, for the same reason ``g1_control``'s is: the env cfg is built at
import time and ``mjlab.tasks`` imports every task package eagerly, so a missing trajectory
would otherwise raise straight out of ``import mjlab.tasks`` and take the whole registry down
with it. Checking the data first keeps the failure local.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1_control.runner import ControlTrackingOnPolicyRunner

from .env_cfgs import (
  MOTION_FILE,
  TVLQR_EXPORT,
  unitree_g1_robust_tracking_env_cfg,
)
from .rl_cfg import unitree_g1_robust_tracking_ppo_runner_cfg

_missing = [str(p) for p in (MOTION_FILE, TVLQR_EXPORT) if not p.exists()]
if _missing:
  print(
    f"[WARNING]: skipping G1-Robust-Tracking -- missing {_missing}. Rebuild from mj-nlp:\n"
    "  python examples/g1_mimic_periodic_v0/export_tvlqr.py\n"
    "  python examples/g1_mimic_periodic_v0/export_mjlab_motion.py\n"
    "then run mjlab's csv_to_npz on the CSV that prints."
  )
else:
  # The zero-init runner comes along from G1-Tracking-Control, but for a different reason: there
  # it seats a zero action on the gait's mean posture, here on the STANDING idle. Either way an
  # actor whose output layer starts at zero starts at a posture the robot can hold, rather than
  # at a random one it has to recover from.
  register_mjlab_task(
    task_id="G1-Robust-Tracking",
    env_cfg=unitree_g1_robust_tracking_env_cfg(),
    play_env_cfg=unitree_g1_robust_tracking_env_cfg(play=True),
    rl_cfg=unitree_g1_robust_tracking_ppo_runner_cfg(),
    runner_cls=ControlTrackingOnPolicyRunner,
  )
