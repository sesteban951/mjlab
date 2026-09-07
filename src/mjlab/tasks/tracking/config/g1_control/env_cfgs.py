"""Unitree G1 CLF-RL tracking environment configuration.

Tracks a DYNAMICALLY FEASIBLE trajectory -- a periodic running gait from a DSMS solve, not a mocap
clip -- and adds the two rewards built on the TVLQR designed around it: the Lyapunov decrease
condition and the distance from the controller's command. See ``mdp/tvlqr.py`` for the controller
and ``export/README.md`` beside the data for the trajectory's provenance and conventions.

Everything else is ``G1-Tracking-Custom``'s: the HOME initial pose, the shared custom DR base, and
the randomized actuator delay.
"""

from dataclasses import replace
from pathlib import Path
from typing import cast

import mjlab
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  SPHERE_FEET_COLLISION,
  get_sphere_feet_spec,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import RewardTermCfg
from mjlab.tasks.tracking.config.g1_custom.env_cfgs import (
  unitree_g1_custom_flat_tracking_env_cfg,
)
from mjlab.tasks.tracking.mdp import tvlqr
from mjlab.tasks.tracking.mdp.rewards import clf_decrease_rbf, qdes_imitation_rbf

# The feasible trajectory and the law designed around it, produced by mj-nlp's
# examples/g1_mimic_periodic_v0 and its export_tvlqr.py / export_mjlab_motion.py.
_TRAJ_DIR = (
  Path(mjlab.MJLAB_SRC_PATH).parent.parent
  / "trajectories"
  / "control"
  / "neutral_jog_ff_180_R_001__A534_M"
)
MOTION_FILE = _TRAJ_DIR / "motion.npz"
TVLQR_EXPORT = _TRAJ_DIR / "g1_mimic_periodic_v0_run_fwd_20260907T002630_tvlqr.npz"


def unitree_g1_control_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
  clf: bool = True,
) -> ManagerBasedRlEnvCfg:
  """G1 tracking on a feasible trajectory, guided by its TVLQR's Lyapunov function.

  ``clf=False`` is the BASELINE: plain trajectory tracking with NOTHING from the controller.
  The two CLF-RL rewards are gone, the action term is the stock one, and the offset is back to
  the HOME keyframe -- the re-centred offset is derived from the LQR's feedforward ``u_bar``, so
  leaving it in would be control help smuggled into the action space. What the baseline keeps is
  the trajectory, the sole, the DR and the PPO config, because those are the plant and the task,
  not the controller.

  Both arms come from THIS function on purpose. A hand-copied baseline drifts the moment either
  side is tuned, and then the comparison silently stops being a comparison.

  Note the TVLQR action term is observationally inert in guide-only mode -- it applies the
  policy's target exactly as the stock term does, and only computes qdes_ctrl and V alongside.
  So dropping it changes no physics; it just removes the npz load and the per-substep gain
  evaluation, which is also the cheaper thing to run.
  """
  cfg = unitree_g1_custom_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation, play=play
  )

  # the baseline needs only the motion; the CLF arm also needs the schedule
  needed = [(MOTION_FILE, "motion")] + ([(TVLQR_EXPORT, "TVLQR export")] if clf else [])
  for path, what in needed:
    if not path.exists():
      raise FileNotFoundError(
        f"{what} not found at {path}. Build it from mj-nlp:\n"
        f"  python examples/g1_mimic_periodic_v0/export_tvlqr.py\n"
        f"  python examples/g1_mimic_periodic_v0/export_mjlab_motion.py\n"
        f"then run mjlab's csv_to_npz on the CSV that prints (it does the forward kinematics)."
      )

  # ---- THE SOLE MUST MATCH THE ONE THE LAW WAS DESIGNED ON. mjlab's stock G1 has 7 capsules per
  # foot at mu=0.6; the trajectory and the gain schedule were built against mj-nlp's 4 point
  # spheres at mu=1.0. Everything else about the two robots already agreed -- kp, kd, tau_max,
  # armature, damping, frictionloss, jnt_range, timestep, integrator, cone -- and the feet alone
  # were enough to break the TVLQR's own Lyapunov certificate here (V climbing 2.6 -> 125 over
  # 1.2 s with the law in the loop). Only the spec and the collision cfg change; init_state (the
  # HOME keyframe from g1_custom) and the articulation are left as they are.
  robot = cfg.scene.entities["robot"]
  robot.spec_fn = get_sphere_feet_spec
  robot.collisions = (SPHERE_FEET_COLLISION,)

  # ---- track the FEASIBLE trajectory instead of the mocap clip
  cfg.commands["motion"] = replace(cfg.commands["motion"], motion_file=str(MOTION_FILE))

  # ---- the action space. Offset re-centred on the trajectory's own posture rather than the HOME
  # keyframe -- see tvlqr.reference_offset. IDENTICAL IN BOTH ARMS: the offset is a property of
  # the trajectory, not of the CLF rewards, and letting it differ would confound the ablation
  # with a change of action space.
  base_action = cast(JointPositionActionCfg, cfg.actions["joint_pos"])
  if not clf:
    # the stock action term, untouched: HOME offset, mjlab's own per-joint scales. The baseline
    # gets its posture prior from the tracking rewards alone, like G1-Tracking-Custom does.
    return cfg
  cfg.actions["joint_pos"] = tvlqr.TvlqrGuidedJointPositionActionCfg(
    entity_name=base_action.entity_name,
    actuator_names=tuple(base_action.actuator_names),
    scale=base_action.scale,
    # re-centred on the trajectory's own posture -- see tvlqr.reference_offset. CLF ARM ONLY:
    # it comes from u_bar, so it is part of the controller's contribution, not of the task.
    offset=tvlqr.reference_offset(str(TVLQR_EXPORT), mode="mean"),
    use_default_offset=False,
    export_path=str(TVLQR_EXPORT),
    motion_command_name="motion",
  )

  # ---- the two CLF-RL rewards. Both sigmas are MEASURED, not guessed: on a 32-env x 200-step
  # zero-action rollout the relative CLF violation lands at p50 0.08 / p95 0.24, and the imitation
  # error at p50 6.7 rad / p95 11.3 rad. Those set the scales below to give ~0.7 and ~0.5 mean
  # reward on that (deliberately pessimistic) rollout, so there is gradient in both directions
  # rather than a term pinned at 0 or 1. Weights are a starting point, not a tuned result.
  cfg.rewards["clf_decrease"] = RewardTermCfg(
    func=clf_decrease_rbf,
    weight=1.0,
    params={"action_name": "joint_pos", "sigma": 0.5},
  )
  # sigma is large because the error is a 29-dim NORM and this export's max|K| is 405 (see the
  # export README's gain-magnitude note): qdes_ctrl swings +/-2.8 rad while any fixed offset sits
  # still, so ~6.7 rad of disagreement is the honest starting point, not a bug. If this term stays
  # pinned near 0 in training, the lever is the action `scale` or this sigma -- not the gains.
  cfg.rewards["qdes_imitation"] = RewardTermCfg(
    func=qdes_imitation_rbf,
    weight=1.0,
    params={"action_name": "joint_pos", "sigma": 3.0},
  )

  return cfg
