"""Tests for the vendored Unitree Go2 custom velocity envs."""

import mujoco
import pytest

from mjlab.asset_zoo.robots.unitree_go2.go2_constants import get_go2_robot_cfg
from mjlab.entity import Entity
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity_custom.config.go2_custom.env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_rough_env_cfg,
)
from mjlab.tasks.velocity_custom.mdp import UniformVelocityCommandCfg


def test_go2_asset_compiles_with_meshes():
  model = Entity(get_go2_robot_cfg()).spec.compile()
  assert model.nmesh > 0
  assert model.nu == 12
  for name in ("FR_foot_collision", "FL_foot_collision"):
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0


@pytest.mark.parametrize(
  "task_id, make_cfg",
  [
    ("Go2-Velocity-Custom", unitree_go2_flat_env_cfg),
    ("Go2-Velocity-Rough-Custom", unitree_go2_rough_env_cfg),
  ],
)
def test_go2_tasks_are_registered(task_id, make_cfg):
  cfg = load_env_cfg(task_id)
  expected = make_cfg()
  assert cfg.viewer == expected.viewer
  assert cfg.rewards == expected.rewards
  assert cfg.scene.terrain is not None and expected.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == expected.scene.terrain.terrain_type
  assert load_rl_cfg(task_id).experiment_name == "go2_velocity_custom"


def test_go2_flat_cfg_targets_go2_bodies():
  cfg = unitree_go2_flat_env_cfg()
  assert cfg.viewer.body_name == "base_link"
  assert cfg.events["base_com"].params["asset_cfg"].body_names == ("base_link",)
  assert cfg.rewards["action_rate_l2"].weight == -0.05
  assert "height_scan" not in cfg.observations["actor"].terms
  assert not any(s.name == "terrain_scan" for s in cfg.scene.sensors or ())

  twist = unitree_go2_flat_env_cfg(play=True).commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  assert twist.ranges.lin_vel_x == (-0.5, 1.0)
