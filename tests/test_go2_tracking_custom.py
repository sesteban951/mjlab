"""Tests for the SuperMimic-style Unitree Go2 custom tracking task."""

import mujoco
import numpy as np
import pytest
import torch

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.asset_zoo.robots.unitree_go2.go2_constants_supermimic import (
  get_go2_supermimic_robot_cfg,
)
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.config.go2_custom.env_cfgs import (
  GO2_TRACKED_BODIES,
  unitree_go2_custom_flat_tracking_env_cfg,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg


def test_go2_supermimic_actuators():
  cfg = get_go2_supermimic_robot_cfg()
  assert cfg.articulation is not None
  for act in cfg.articulation.actuators:
    assert isinstance(act, DcMotorActuatorCfg)
    assert (act.stiffness, act.damping) == (25.0, 0.5)
    assert act.effort_limit == act.saturation_effort == 23.5
    assert act.velocity_limit == 30.0
  model = Entity(cfg).spec.compile()
  assert model.nu == 12


def test_go2_supermimic_legs_self_collide():
  model = Entity(get_go2_supermimic_robot_cfg()).spec.compile()
  data = mujoco.MjData(model)

  def self_contacts(fl_hip: float, fr_hip: float) -> int:
    data.qpos[:] = [0, 0, 1.3, 1, 0, 0, 0] + [0.0, 0.9, -1.8] * 4  # In the air.
    data.qpos[7], data.qpos[10] = fl_hip, fr_hip
    mujoco.mj_forward(model, data)
    return data.ncon

  assert self_contacts(0.0, 0.0) == 0
  # Swinging both front legs inward makes them collide.
  assert self_contacts(-0.8, 0.8) > 0


def test_go2_tracking_task_is_registered():
  cfg = load_env_cfg("Go2-Tracking-Custom")
  expected = unitree_go2_custom_flat_tracking_env_cfg()
  assert cfg.rewards == expected.rewards
  assert cfg.terminations == expected.terminations
  assert load_rl_cfg("Go2-Tracking-Custom").experiment_name == "go2_tracking_custom"


def test_go2_tracking_cfg_matches_supermimic():
  cfg = unitree_go2_custom_flat_tracking_env_cfg()
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  assert motion.anchor_body_name == "base_link"
  assert len(motion.body_names) == 13

  stds = {k: t.params.get("std") for k, t in cfg.rewards.items()}
  assert stds["motion_global_root_pos"] == 0.2
  assert stds["motion_body_pos"] == 0.15
  assert stds["motion_body_lin_vel"] == 0.5
  assert cfg.rewards["joint_limit"].weight == -2.0
  assert cfg.rewards["self_collisions"].weight == -1.0
  assert [s.name for s in cfg.scene.sensors or ()] == ["self_collision"]

  assert cfg.terminations["anchor_pos"].params["threshold"] == 0.16
  assert "ee_body_pos" not in cfg.terminations

  actor_terms = cfg.observations["actor"].terms
  assert "motion_anchor_pos_b" not in actor_terms
  assert "base_lin_vel" not in actor_terms


def test_go2_tracking_custom_dr():
  cfg = unitree_go2_custom_flat_tracking_env_cfg()
  ev = cfg.events
  # SuperMimic's Go2 COM offset, and no PD-gain randomization.
  assert "pd_gains" not in ev
  assert ev["base_com"].params["ranges"][0] == (-0.01, 0.01)
  assert ev["joint_armature"].params["ranges"] == (0.95, 1.05)
  assert ev["base_mass"].params["asset_cfg"].body_names == ("base_link",)
  assert ev["base_com"].params["asset_cfg"].body_names == ("base_link",)
  assert ev["push_robot"].params["velocity_range"]["x"] == (-0.3, 0.3)
  assert ev["actuator_delay"].params == {"min_lag": 0, "max_lag": 8}
  robot = cfg.scene.entities["robot"]
  assert robot.articulation is not None
  for act in robot.articulation.actuators:
    assert (act.delay_max_lag, act.delay_hold_prob) == (8, 1.0)

  # Play keeps the startup DR but drops pushes and command delay.
  play = unitree_go2_custom_flat_tracking_env_cfg(play=True)
  assert "base_mass" in play.events
  assert "push_robot" not in play.events
  assert "actuator_delay" not in play.events


@pytest.fixture
def go2_motion_file(tmp_path):
  """A short standing reference built from the Go2's default stance."""
  robot = Entity(get_go2_supermimic_robot_cfg())
  model = robot.spec.compile()
  data = mujoco.MjData(model)
  data.qpos[2] = 0.30
  data.qpos[3] = 1.0
  data.qpos[7:] = [0.0, 0.9, -1.8] * 4
  mujoco.mj_kinematics(model, data)
  frames = 20
  nbody = model.nbody - 1  # Skip the world body.
  path = tmp_path / "stand.npz"
  np.savez(
    path,
    fps=np.array([50]),
    joint_pos=np.tile(data.qpos[7:], (frames, 1)),
    joint_vel=np.zeros((frames, 12)),
    body_pos_w=np.tile(data.xpos[1:], (frames, 1, 1)),
    body_quat_w=np.tile(data.xquat[1:], (frames, 1, 1)),
    body_lin_vel_w=np.zeros((frames, nbody, 3)),
    body_ang_vel_w=np.zeros((frames, nbody, 3)),
  )
  return str(path)


def test_go2_tracking_env_steps(go2_motion_file):
  cfg = unitree_go2_custom_flat_tracking_env_cfg()
  cfg.scene.num_envs = 2
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.motion_file = go2_motion_file
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    robot = env.scene["robot"]
    assert all(name in robot.body_names for name in GO2_TRACKED_BODIES)
    env.reset()
    actions = torch.zeros(2, env.action_manager.total_action_dim)
    for _ in range(5):
      _, reward, *_ = env.step(actions)
    assert torch.isfinite(reward).all()

    # Gains are not randomized: every env runs SuperMimic's kp 25.
    for act in robot.actuators:
      assert act.stiffness is not None
      assert (act.stiffness == 25.0).all()

    # DC motor gains live in the actuator config, not in MuJoCo's gainprm.
    metadata = get_base_metadata(env, "test")
    assert metadata["joint_stiffness"] == [25.0] * 12
    assert metadata["joint_damping"] == [0.5] * 12
  finally:
    env.close()
