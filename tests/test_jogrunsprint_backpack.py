"""G1-JogRunSprint-Custom-Backpack: the parent task on the rear-backpack robot."""

import numpy as np
import pytest

import mjlab.tasks  # noqa: F401  Populate the task registry.
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants_backpack as bp
from mjlab.entity import Entity
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK, PARENT = "G1-JogRunSprint-Custom-Backpack", "G1-JogRunSprint-Custom"


def test_robot_carries_the_rear_backpack() -> None:
  robot = load_env_cfg(TASK).scene.entities["robot"]
  assert robot.spec_fn is bp.get_rear_backpack_spec
  model = Entity(robot).spec.compile()
  body = model.body(bp.BACKPACK_BODY_NAME)
  assert model.body(body.parentid[0]).name == "torso_link"
  # 5 % heavier than the stock backpack and 5 cm further back, nothing else moved.
  assert body.mass[0] == pytest.approx(1.05 * bp.BACKPACK_MASS) == pytest.approx(1.47)
  np.testing.assert_allclose(body.pos - np.asarray(bp.BACKPACK_POS), (-0.05, 0, 0))
  np.testing.assert_allclose(
    model.geom("backpack_collision").size, np.asarray(bp.BACKPACK_SIZE) / 2
  )


def test_everything_else_is_the_parent() -> None:
  for play in (False, True):
    cfg, base = load_env_cfg(TASK, play=play), load_env_cfg(PARENT, play=play)
    robot, base_robot = cfg.scene.entities["robot"], base.scene.entities["robot"]
    assert robot.init_state == base_robot.init_state
    assert robot.articulation == base_robot.articulation
    assert robot.collisions == base_robot.collisions
    for part in ("observations", "actions", "commands", "rewards", "terminations"):
      assert getattr(cfg, part) == getattr(base, part), part
    assert cfg.events == base.events
  rl, base_rl = load_rl_cfg(TASK), load_rl_cfg(PARENT)
  assert rl.experiment_name == base_rl.experiment_name + "_backpack"
  rl.experiment_name = base_rl.experiment_name
  assert rl == base_rl
  assert load_runner_cls(TASK) is load_runner_cls(PARENT)
