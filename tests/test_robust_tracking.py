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
from mjlab.tasks.crawling_common.library import load_idle_qpos
from mjlab.tasks.robust_tracking.config.g1.env_cfgs import (
  MOTION_FILE,
  PHYSICS_DT,
  STAND_HOLD_S,
  STAND_QPOS_CSV,
  TVLQR_EXPORT,
  unitree_g1_clf_tracking_env_cfg,
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


def test_action_offset_is_the_standing_idle(cfg):
  """A zero action is the stand the start pad opens on; `use_default_offset` carries it."""
  _, _, stand = load_idle_qpos(STAND_QPOS_CSV)
  init = cfg.scene.entities["robot"].init_state
  assert cfg.actions["joint_pos"].use_default_offset
  assert init.joint_pos == stand
  export = np.load(TVLQR_EXPORT)
  names = [str(n).split("/")[-1] for n in export["dof_names"][6 : int(export["nv"])]]
  np.testing.assert_allclose(
    np.load(MOTION_FILE)["joint_pos"][0], [stand[n] for n in names], atol=1e-6
  )


def test_physics_runs_at_the_schedule_rate(cfg):
  """Physics integrates finer than the schedule; the policy must stay at 50 Hz."""
  dt_export = float(np.load(TVLQR_EXPORT)["dt"])
  stride = round(dt_export / cfg.sim.mujoco.timestep)
  assert cfg.sim.mujoco.timestep == PHYSICS_DT
  # The export rate must be an integer multiple of the physics rate, and the env step must
  # contain a whole number of schedule entries, or the loader rejects the pair.
  assert dt_export == pytest.approx(stride * cfg.sim.mujoco.timestep)
  assert cfg.decimation % stride == 0
  assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(0.02)


def test_actuator_delay_survives_the_timestep_change(cfg):
  """The delay is configured in STEPS; re-applying after the timestep change is what keeps it
  at 0.04 s instead of silently doubling."""
  act = cfg.scene.entities["robot"].articulation.actuators[0]
  assert act.delay_max_lag * cfg.sim.mujoco.timestep == pytest.approx(0.04)
  assert act.delay_update_period == cfg.decimation


def test_schedule_and_motion_frames_line_up(cfg):
  """k0 advances `decimation // stride` per env step, so that must divide the schedule."""
  export = np.load(TVLQR_EXPORT)
  n_sched = int(export["u_bar"].shape[0])
  motion = np.load(MOTION_FILE)
  n_motion = int(motion["joint_pos"].shape[0] - motion["pad_frames"].sum())
  stride = round(float(export["dt"]) / cfg.sim.mujoco.timestep)
  per_env_step = cfg.decimation // stride
  assert n_sched // per_env_step == n_motion
  # The last env step must land on the final entry, not run off the end and clamp.
  assert (n_motion - 1) * per_env_step + (per_env_step - 1) == n_sched - 1


def test_has_the_guide_only_action_and_both_clf_rewards(cfg):
  action = cfg.actions["joint_pos"]
  assert isinstance(action, TvlqrGuidedJointPositionActionCfg)
  assert not action.residual, (
    "the controller must guide by reward, never drive the robot"
  )
  assert {"clf_decrease", "qdes_imitation"} <= set(cfg.rewards)
  assert {"action_rate_limbs", "action_rate_waist"} <= set(cfg.rewards)


def test_terrain_patch_holds_the_whole_clip(cfg):
  """The clip is anchored to the patch origin, so every body must stay on that patch."""
  body_pos = np.load(MOTION_FILE)["body_pos_w"]
  spawn = body_pos[0, 0, :2]
  reach = np.linalg.norm(body_pos[..., :2] - spawn, axis=-1).max()
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  # env_origins land at the patch center, so half the smaller side is the budget.
  assert reach < min(generator.size) / 2


def test_contact_solref_stays_solver_stable(cfg):
  """MuJoCo clamps timeconst at 2*dt, and the clamped contact rebounds off hard hits."""
  ranges = cfg.events["contact_solref"].params["ranges"]
  assert ranges[0][0] > 2 * cfg.sim.mujoco.timestep
  assert ranges[1][0] > 0.0


def test_clip_is_upright_at_both_ends(cfg):
  """Guards the quaternion-order bug: a wxyz dump reordered as xyzw still loads fine."""
  del cfg
  motion = np.load(MOTION_FILE)
  q = motion["body_quat_w"][:, 0]
  tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)))
  assert tilt[0] < 15 and tilt[-1] < 15, "the clip must start and end upright"
  assert tilt.max() > 45, "it is a sideroll; it should lie down in the middle"
  assert motion["body_pos_w"][:, :, 2].min() > -0.05, "reference goes through the floor"


def test_clip_and_schedule_share_one_source(cfg):
  """The motion must be x_bar itself, strided, not a second dump of the same solve."""
  motion, export = np.load(MOTION_FILE), np.load(TVLQR_EXPORT)
  stride = round(float(export["dt"]) / cfg.sim.mujoco.timestep)
  per_env_step = cfg.decimation // stride
  start, end = (int(n) for n in motion["pad_frames"])
  clip = slice(start, motion["body_pos_w"].shape[0] - end)
  idx = np.arange(clip.stop - clip.start) * per_env_step
  x_bar = export["x_bar"]
  np.testing.assert_allclose(motion["body_pos_w"][clip, 0], x_bar[idx, :3], atol=1e-5)
  np.testing.assert_allclose(
    motion["joint_pos"][clip], x_bar[idx, 7 : int(export["nq"])], atol=1e-4
  )


def test_clip_is_padded_with_standing_holds(cfg):
  motion = np.load(MOTION_FILE)
  start, end = (int(n) for n in motion["pad_frames"])
  assert cfg.actions["joint_pos"].motion_pad_start == start
  assert start == round(STAND_HOLD_S * float(motion["fps"][0]))
  for pad in (slice(0, start), slice(-end - 1, None)):
    assert np.ptp(motion["joint_pos"][pad], axis=0).max() == 0.0
  assert not motion["joint_vel"][:start].any() and not motion["joint_vel"][-end:].any()


def test_one_clip_is_one_episode(cfg):
  """The clip is one-shot, so the episode must end with it rather than wrap and teleport."""
  term = cfg.terminations["motion_complete"]
  assert term.time_out, "finishing the clip is success; the value must bootstrap"
  n_frames = int(np.load(MOTION_FILE)["joint_pos"].shape[0])
  clip_s = n_frames * cfg.sim.mujoco.timestep * cfg.decimation
  assert cfg.episode_length_s >= clip_s, "the ceiling must not cut the clip short"
  assert cfg.episode_length_s < 2 * clip_s, "the ceiling must not allow a second pass"


def test_clf_env_adds_v_tracking_and_leaves_robust_untouched(cfg):
  clf = unitree_g1_clf_tracking_env_cfg()
  action = clf.actions["joint_pos"]
  assert isinstance(action, TvlqrGuidedJointPositionActionCfg)
  assert action.v_ref_path is not None and action.v_floor_scale > 0
  assert clf.rewards["clf_decrease"].params["sigma"] == 0.3
  assert "clf_tracking" in clf.rewards and "clf_tracking" in clf.reward_groups["clf"]
  assert cfg.actions["joint_pos"].v_ref_path is None
  assert "clf_tracking" not in cfg.rewards
  assert "clf_tracking" not in cfg.reward_groups["clf"]
