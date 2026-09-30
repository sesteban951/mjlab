"""G1-{Walk,Jog}-Unicycle-LSTM-Backpack: the LSTM tasks on the backpack robot, nothing else."""

import pytest

import mjlab.tasks  # noqa: F401  Populate the task registry.
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants_backpack
from mjlab.entity import Entity
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

PAIRS = [
  ("G1-Walk-Unicycle-LSTM-Backpack", "G1-Walk-Unicycle-LSTM"),
  ("G1-Jog-Unicycle-LSTM-Backpack", "G1-Jog-Unicycle-LSTM"),
]


@pytest.mark.parametrize(("task", "parent"), PAIRS)
def test_robot_is_the_backpack_model(task: str, parent: str) -> None:
  for play in (False, True):
    robot = load_env_cfg(task, play=play).scene.entities["robot"]
    assert robot.spec_fn is g1_constants_backpack.get_backpack_spec
    model = Entity(robot).spec.compile()
    body = model.body(g1_constants_backpack.BACKPACK_BODY_NAME)
    assert body.mass[0] == pytest.approx(g1_constants_backpack.BACKPACK_MASS)
    assert model.body(body.parentid[0]).name == "torso_link"
    # Still the mode-11 robot underneath.
    assert model.actuator("left_hip_pitch_joint").forcerange[1] == 139.0


@pytest.mark.parametrize(("task", "parent"), PAIRS)
def test_everything_else_is_the_lstm_parent(task: str, parent: str) -> None:
  for play in (False, True):
    cfg, base = load_env_cfg(task, play=play), load_env_cfg(parent, play=play)
    robot, base_robot = cfg.scene.entities["robot"], base.scene.entities["robot"]
    assert robot.init_state == base_robot.init_state
    assert robot.articulation == base_robot.articulation
    assert robot.collisions == base_robot.collisions
    assert cfg.observations == base.observations
    assert cfg.actions == base.actions
    assert cfg.commands == base.commands
    assert cfg.rewards == base.rewards
    assert cfg.terminations == base.terminations
    assert cfg.events == base.events
    assert cfg.episode_length_s == base.episode_length_s


@pytest.mark.parametrize(("task", "parent"), PAIRS)
def test_rl_cfg_is_the_lstm_parent_under_its_own_name(task: str, parent: str) -> None:
  cfg, base = load_rl_cfg(task), load_rl_cfg(parent)
  assert cfg.experiment_name == base.experiment_name + "_backpack"
  cfg.experiment_name = base.experiment_name
  assert cfg == base
  assert load_runner_cls(task) is load_runner_cls(parent)
