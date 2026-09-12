"""Config-level checks for the G1-Robust-Tracking env.

All of these guard a wiring mistake that would only show up as a bad training run: a sole that
silently changed, a schedule read at the wrong rate, or a motion whose frames do not line up
with the gain schedule the rewards index into.
"""

import numpy as np
import pytest

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  get_spec,
  get_sphere_feet_spec,
)
from mjlab.tasks.crawling_common.library import LIBRARY_SPECS, load_idle_qpos
from mjlab.tasks.robust_tracking.config.g1.env_cfgs import (
  MOTION_FILE,
  SOLVE_DT,
  TVLQR_EXPORT,
  unitree_g1_robust_tracking_env_cfg,
)
from mjlab.tasks.tracking.mdp.tvlqr import TvlqrGuidedJointPositionActionCfg

pytestmark = pytest.mark.skipif(
  not (MOTION_FILE.exists() and TVLQR_EXPORT.exists()),
  reason="sideroll trajectory not built; see env_cfgs for the rebuild commands",
)


@pytest.fixture
def cfg():
  return unitree_g1_robust_tracking_env_cfg()


def test_keeps_the_capsule_sole(cfg):
  """The sim2real plant wins over the sole the LQR was designed on -- the whole point of
  guide-only. g1_control swaps in spheres; this task must not."""
  robot = cfg.scene.entities["robot"]
  assert robot.spec_fn is get_spec
  assert robot.spec_fn is not get_sphere_feet_spec


def test_uses_mode11_hip_pitch_actuators(cfg):
  """Mode 11 moves hip pitch to the 22.5:1 gearbox, which is an ACTUATOR change only."""
  groups = [
    a.target_names_expr for a in cfg.scene.entities["robot"].articulation.actuators
  ]
  assert ("left_hip_pitch_joint",) in groups
  assert ("right_hip_pitch_joint",) in groups


def test_starts_on_the_standing_idle_pose(cfg):
  """Init state is G1-Standing-DiffDrive's idle qpos, and `use_default_offset` carries it into
  the action offset, so a zero action is the stand."""
  pos, quat, joint_pos = load_idle_qpos(LIBRARY_SPECS["standing_diffdrive"].idle_csv)
  init = cfg.scene.entities["robot"].init_state
  assert init.pos == pos
  assert init.rot == quat
  assert init.joint_pos == joint_pos
  assert cfg.actions["joint_pos"].use_default_offset


def test_physics_runs_at_the_schedule_rate(cfg):
  """The loader refuses any timestep but the export's, and the policy must stay at 50 Hz."""
  assert cfg.sim.mujoco.timestep == SOLVE_DT
  assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(0.02)


def test_actuator_delay_survives_the_timestep_change(cfg):
  """The delay is configured in STEPS; re-applying after the timestep change is what keeps it
  at 0.04 s instead of silently doubling."""
  act = cfg.scene.entities["robot"].articulation.actuators[0]
  assert act.delay_max_lag * cfg.sim.mujoco.timestep == pytest.approx(0.04)
  assert act.delay_update_period == cfg.decimation


def test_schedule_and_motion_frames_line_up(cfg):
  """k0 = time_steps * decimation, so the schedule must be `decimation` times the motion."""
  n_sched = int(np.load(TVLQR_EXPORT)["u_bar"].shape[0])
  n_motion = int(np.load(MOTION_FILE)["joint_pos"].shape[0])
  assert n_sched // cfg.decimation == n_motion


def test_has_the_guide_only_action_and_both_clf_rewards(cfg):
  action = cfg.actions["joint_pos"]
  assert isinstance(action, TvlqrGuidedJointPositionActionCfg)
  assert not action.residual, (
    "the controller must guide by reward, never drive the robot"
  )
  assert {"clf_decrease", "qdes_imitation"} <= set(cfg.rewards)
  assert {"action_rate_limbs", "action_rate_waist"} <= set(cfg.rewards)
