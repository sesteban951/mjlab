"""G1-{Walk,WalkJog}-Unicycle-Backpack: the parent tasks on the backpack robot."""

import numpy as np
import pytest

import mjlab.tasks  # noqa: F401  Populate the task registry.
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants_backpack as bp
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
from mjlab.entity import Entity
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

PAIRS = [
  ("G1-Walk-Unicycle-Backpack", "G1-Walk-Unicycle"),
  ("G1-WalkJog-Unicycle-Backpack", "G1-WalkJog-Unicycle"),
]


@pytest.mark.parametrize(("task", "parent"), PAIRS)
def test_robot_carries_the_backpack(task: str, parent: str) -> None:
  # The parent must be on the stock sole, the one get_backpack_spec builds on.
  assert load_env_cfg(parent).scene.entities["robot"].spec_fn is get_spec
  robot = load_env_cfg(task).scene.entities["robot"]
  assert robot.spec_fn is bp.get_backpack_spec
  model = Entity(robot).spec.compile()
  body = model.body(bp.BACKPACK_BODY_NAME)
  assert model.body(body.parentid[0]).name == "torso_link"
  assert body.mass[0] == pytest.approx(bp.BACKPACK_MASS)
  np.testing.assert_allclose(body.pos, bp.BACKPACK_POS)


@pytest.mark.parametrize(("task", "parent"), PAIRS)
def test_everything_else_is_the_parent(task: str, parent: str) -> None:
  for play in (False, True):
    cfg, base = load_env_cfg(task, play=play), load_env_cfg(parent, play=play)
    robot, base_robot = cfg.scene.entities["robot"], base.scene.entities["robot"]
    assert robot.init_state == base_robot.init_state
    assert robot.articulation == base_robot.articulation
    assert robot.collisions == base_robot.collisions
    for part in ("observations", "actions", "commands", "rewards", "terminations"):
      assert getattr(cfg, part) == getattr(base, part), part
    assert cfg.events == base.events
  rl, base_rl = load_rl_cfg(task), load_rl_cfg(parent)
  assert rl.experiment_name == base_rl.experiment_name + "_backpack"
  rl.experiment_name = base_rl.experiment_name
  assert rl == base_rl
  assert load_runner_cls(task) is load_runner_cls(parent)
