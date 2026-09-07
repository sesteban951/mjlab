"""Registration for the CLF-RL tracking tasks.

REGISTRATION IS GUARDED. Both env cfgs are built at import time (the CLF arm reads the gain
schedule's u_bar to re-centre its action offset), and ``mjlab.tasks`` imports every task package
eagerly -- so a missing data file used to raise straight out of ``import mjlab.tasks`` and take
the WHOLE registry down with it, not just these two tasks. On a fresh clone without the
trajectories, that bricked every unrelated task too. Now the data is checked first and the pair is
skipped with a warning, so the failure stays local and says how to fix itself.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  MOTION_FILE,
  TVLQR_EXPORT,
  unitree_g1_control_flat_tracking_env_cfg,
)
from .rl_cfg import (
  unitree_g1_control_baseline_tracking_ppo_runner_cfg,
  unitree_g1_control_tracking_ppo_runner_cfg,
)
from .runner import ControlTrackingOnPolicyRunner

_missing = [str(p) for p in (MOTION_FILE, TVLQR_EXPORT) if not p.exists()]
if _missing:
  print(
    "[WARNING]: skipping G1-Tracking-Control and G1-Tracking-Control-Baseline -- missing "
    f"{_missing}. Rebuild from mj-nlp:\n"
    "  python examples/g1_mimic_periodic_v0/export_tvlqr.py\n"
    "  python examples/g1_mimic_periodic_v0/export_mjlab_motion.py\n"
    "then run mjlab's csv_to_npz on the CSV that prints."
  )
else:
  register_mjlab_task(
    task_id="G1-Tracking-Control",
    env_cfg=unitree_g1_control_flat_tracking_env_cfg(has_state_estimation=False),
    play_env_cfg=unitree_g1_control_flat_tracking_env_cfg(
      has_state_estimation=False, play=True
    ),
    rl_cfg=unitree_g1_control_tracking_ppo_runner_cfg(),
    runner_cls=ControlTrackingOnPolicyRunner,
  )

  # THE BASELINE. Same trajectory, sole, DR and PPO; the two CLF-RL rewards removed, the stock
  # action term and the stock HOME offset restored, and the STOCK runner -- so the actor's output
  # layer is NOT zero-initialized, since that init exists only to seat a zero action on the gait
  # for the imitation reward. Both env cfgs come from one function (see its docstring) so the
  # arms cannot drift apart.
  register_mjlab_task(
    task_id="G1-Tracking-Control-Baseline",
    env_cfg=unitree_g1_control_flat_tracking_env_cfg(
      has_state_estimation=False, clf=False
    ),
    play_env_cfg=unitree_g1_control_flat_tracking_env_cfg(
      has_state_estimation=False, play=True, clf=False
    ),
    rl_cfg=unitree_g1_control_baseline_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
  )
