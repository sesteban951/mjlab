"""Unitree G1 CLF-guided tracking, tuned for sim2real transfer.

Built on ``G1-Tracking-Custom`` (mode-11 actuators, the shared custom DR base, the randomized
actuator delay and the limb/waist action-rate split) and adds ``G1-Tracking-Control``'s two
guide-only TVLQR rewards. Unlike that task this one has no ablation switches: the blend weight,
the ``lam`` curriculum and the five ``MJLAB_PRIOR_*`` environment variables are all gone, and
the constants below are the only knobs.

What the TVLQR does here is unchanged from ``tracking.mdp.tvlqr``: the schedule is evaluated
every physics substep alongside the policy to produce ``qdes_ctrl`` and ``V``, and the policy's
own target is what reaches the sim. The controller shapes reward; it never acts.
"""

from dataclasses import replace
from pathlib import Path
from typing import cast

import mjlab
from mjlab.asset_zoo.robots.unitree_g1.custom_dr import add_custom_g1_actuator_delay
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import RewardTermCfg
from mjlab.tasks.crawling_common.library import LIBRARY_SPECS, load_idle_qpos
from mjlab.tasks.tracking.config.g1_custom.env_cfgs import (
  unitree_g1_custom_flat_tracking_env_cfg,
)
from mjlab.tasks.tracking.mdp import tvlqr
from mjlab.tasks.tracking.mdp.rewards import clf_decrease_rbf, qdes_imitation_rbf

# The tracked trajectory and the LQR designed around it: mj-nlp's sideroll solve
# (examples/full_run/g1_mimic_DR/g1_mimic_svd_alpha_sideroll_v3.npz), a stand -> side roll ->
# recover-to-stand over 5.44 s that starts AND ends upright, drifting about 2 m sideways.
#
# Both files here are DERIVED FROM THAT ONE SOLVE, which is what keeps them in lockstep:
#
#   sideroll_tvlqr.npz  scripts/mjnlp_solve_to_tvlqr.py, mapping the solve's
#                       {state, input, gains, cost_to_go} onto {x_bar, u_bar, K, P}.
#   motion.npz          scripts/csv_to_npz.py on the solve's own `state` qpos at 100 -> 50 Hz,
#                       NOT on trajectories/single/to_sideroll (a different, longer take). The
#                       schedule is indexed off the motion command's frame counter, so a motion
#                       built from any other source would desync the two.
_TRAJ_DIR = (
  Path(mjlab.MJLAB_SRC_PATH).parent.parent
  / "trajectories"
  / "control"
  / "sideroll_mimic_svd_alpha_v3"
)
MOTION_FILE = _TRAJ_DIR / "motion.npz"
TVLQR_EXPORT = _TRAJ_DIR / "sideroll_tvlqr.npz"

# THE SOLVE'S RATE IS THE SIM'S RATE. The gain schedule was designed at 0.01 s and the loader
# refuses any other physics timestep -- applying K at a rate it was not designed for is a
# different controller. So this env runs physics at the solve's 100 Hz rather than mjlab's
# default 200 Hz, and halves the decimation to hold the policy at the usual 50 Hz.
SOLVE_DT = 0.01
DECIMATION = 2

# Actuator command delay, re-applied after the timestep change (see the env fn). Same seconds as
# G1-Tracking-Custom; only the step count behind it moves.
ACTUATOR_DELAY_RANGE_S = (0.0, 0.04)

# The standing idle pose, shared with G1-Standing-DiffDrive: one qpos row that is both that
# task's zero-twist clip and its initial state. Reused here purely as a start/offset posture --
# this task tracks the jog, not the walk library, so nothing else from that spec is touched.
IDLE_QPOS_CSV = LIBRARY_SPECS["standing_diffdrive"].idle_csv

# CLF decrease RBF width. MEASURED on a 32-env x 200-step zero-action rollout of this env.
#
# The decay rate the condition asks for is a CONSTANT 0.005 per step, written into the export by
# scripts/mjnlp_solve_to_tvlqr. The sideroll solve carries no CLF rate of its own -- its `alpha`
# array is a per-SVD-mode scaling on the gain matrix (6 columns == svd_modes, already baked into
# `gains`), not a decay rate -- so this is 0.5 1/s converted at this env's dt, which is also
# inside the jog export's measured per-step band once that is rescaled from 200 to 100 Hz. It is
# a rate a real TVLQR design delivers, not a hand-picked number.
#
# On that rollout the relative violation lands at p50 0.013 / p95 0.407 / p99 0.772, against
# p50 0.08 / p95 0.24 for the jog. The median is lower and the tail heavier -- the sideroll
# spends its middle on the ground, where V is large and easy to hold, and pays at the transitions.
# sigma = 0.3 gives a mean reward of 0.70 there, matching what G1-Tracking-Control aimed for,
# while still separating that tail from the median; 0.5 would pin the term at 0.82.
CLF_SIGMA = 0.3

# qdes imitation RBF width. RAISED from G1-Tracking-Control's 3.0, because the action offset is
# no longer the trajectory's mean posture. That task sets offset = mean(u_bar), so a zero action
# sits ON the gait; here a zero action is the STAND. The sideroll begins and ends standing and
# spends its middle on the ground, so the offset is nearly free at both ends and maximally wrong
# through the roll -- which is why the spread below is wide rather than the jog's near-constant
# 3 rad gap.
#
# On the same rollout: p50 7.442 rad, p95 12.344, against the control task's p50 6.7 / p95 11.3.
# sigma = 3.3 gives a mean reward of 0.50 there, the ~0.5 G1-Tracking-Control aimed for with 3.0
# (which would give 0.43 here). If the term sits pinned near 0 in training, this constant and the
# action scale are the levers, not the gains.
QDES_SIGMA = 3.3

# Reward weights, both from G1-Tracking-Control. A starting point, not a tuned result.
CLF_WEIGHT = 1.0
QDES_WEIGHT = 1.0


def unitree_g1_robust_tracking_env_cfg(
  has_state_estimation: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """G1 tracking on a feasible trajectory with guide-only CLF shaping, hardened for transfer.

  ``has_state_estimation`` defaults to False, unlike the stock tracking cfgs: this task exists
  to transfer, and the anchor position and base linear velocity the actor would otherwise see
  are not measurable on the robot. The critic keeps them either way.
  """
  cfg = unitree_g1_custom_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation, play=play
  )

  for path, what in ((MOTION_FILE, "motion"), (TVLQR_EXPORT, "TVLQR export")):
    if not path.exists():
      raise FileNotFoundError(
        f"{what} not found at {path}. Rebuild both from the mj-nlp sideroll solve:\n"
        f"  uv run python -m mjlab.scripts.mjnlp_solve_to_tvlqr \\\n"
        f"    --solve-file <mj-nlp>/examples/full_run/g1_mimic_DR/"
        f"g1_mimic_svd_alpha_sideroll_v3.npz \\\n"
        f"    --output-file {TVLQR_EXPORT}\n"
        f"  # then dump that solve's state[:, :36] to a csv and\n"
        f"  uv run python -m mjlab.scripts.csv_to_npz --input-file <qpos.csv> \\\n"
        f"    --output-name {MOTION_FILE} --input-fps 100 --output-fps 50"
      )

  # ---- run physics at the schedule's own rate; hold the policy at 50 Hz
  cfg.sim.mujoco.timestep = SOLVE_DT
  cfg.decimation = DECIMATION
  # The delay is configured in STEPS, and g1_custom converted its seconds at the old timestep, so
  # the inherited lag would mean twice the latency at this one. Re-applying recomputes both the
  # lag bounds and delay_update_period against the rate actually in force.
  if not play:
    add_custom_g1_actuator_delay(cfg, *ACTUATOR_DELAY_RANGE_S)

  # ---- THE SOLE IS LEFT ALONE, which is the one place this task deliberately disagrees with
  # G1-Tracking-Control. That task swaps in mj-nlp's 4 point spheres at mu=1.0 because the TVLQR
  # was designed on them, and documents the capsule sole breaking the law's own certificate
  # (V climbing 2.6 -> 125 over 1.2 s with the law in the loop). The law is NOT in the loop here:
  # it only scores the policy, so a plant it was not designed for degrades the shaping signal
  # rather than the dynamics. mjlab's stock 7-capsule sole at mu=0.6 is the one the rest of the
  # sim2real stack is calibrated against, so it wins. Note the mode-11 swap in g1_custom only
  # replaces the ACTUATOR table (G1_MODE11_ARTICULATION); it never touched spec_fn or collisions,
  # so the capsules are already what mode 11 runs on and there is nothing to restore here.

  # ---- track the feasible trajectory instead of the stock mocap clip
  cfg.commands["motion"] = replace(cfg.commands["motion"], motion_file=str(MOTION_FILE))

  # ---- start standing, and centre the action space on the stand
  # Same two lines as G1-Standing-DiffDrive: the joint-position action applies
  # ``target = default_joint_pos + scale * action``, so writing the idle pose into init_state
  # makes a zero action BE the stand (``use_default_offset`` stays True, inherited). This
  # replaces G1-Tracking-Control's tvlqr.reference_offset, which re-centres on the gait's mean
  # posture instead -- see QDES_SIGMA for what that costs the imitation term and how it is paid.
  pos, quat, joint_pos = load_idle_qpos(IDLE_QPOS_CSV)
  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.init_state = replace(
    robot_cfg.init_state, pos=pos, rot=quat, joint_pos=joint_pos
  )

  # ---- the guide-only TVLQR action term
  base_action = cast(JointPositionActionCfg, cfg.actions["joint_pos"])
  cfg.actions["joint_pos"] = tvlqr.TvlqrGuidedJointPositionActionCfg(
    entity_name=base_action.entity_name,
    actuator_names=tuple(base_action.actuator_names),
    scale=base_action.scale,
    # Offset from the standing idle pose via default_joint_pos, NOT from u_bar.
    use_default_offset=True,
    export_path=str(TVLQR_EXPORT),
    motion_command_name="motion",
  )

  # ---- the two CLF-RL shaping rewards, on top of the inherited tracking rewards
  cfg.rewards["clf_decrease"] = RewardTermCfg(
    func=clf_decrease_rbf,
    weight=CLF_WEIGHT,
    params={"action_name": "joint_pos", "sigma": CLF_SIGMA},
  )
  cfg.rewards["qdes_imitation"] = RewardTermCfg(
    func=qdes_imitation_rbf,
    weight=QDES_WEIGHT,
    params={"action_name": "joint_pos", "sigma": QDES_SIGMA},
  )

  # The limb/waist action-rate split (limbs -0.15, waist -0.5) is already applied by the custom
  # tracking builder via custom_rewards.add_custom_g1_action_rate_split, same as
  # G1-Standing-DiffDrive gets it; nothing to re-apply here.

  return cfg
