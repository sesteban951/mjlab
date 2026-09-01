"""Tests for the prior-blended joint position action and its lam curriculum."""

from typing import Any
from unittest.mock import Mock

import pytest
import torch
from conftest import get_test_device

from mjlab.envs.mdp.actions.action_with_prior import (
  JointPositionActionWithPrior,
  JointPositionActionWithPriorCfg,
)
from mjlab.envs.mdp.curriculums import action_curriculum
from mjlab.managers.curriculum_manager import CurriculumTermCfg

NUM_ENVS = 4
NUM_JOINTS = 3
DEFAULT_POS = 0.5
PRIOR_POS = 2.0


@pytest.fixture
def device():
  return get_test_device()


@pytest.fixture
def mock_env(device):
  """Env with a single robot whose joints are all at ``DEFAULT_POS``."""
  entity = Mock()
  entity.find_joints_by_actuator_names.return_value = (
    list(range(NUM_JOINTS)),
    [f"joint_{i}" for i in range(NUM_JOINTS)],
  )
  entity.data.default_joint_pos = torch.full(
    (NUM_ENVS, NUM_JOINTS), DEFAULT_POS, device=device
  )
  entity.data.encoder_bias = torch.zeros((NUM_ENVS, NUM_JOINTS), device=device)
  entity.set_joint_position_target = Mock()

  env = Mock()
  env.num_envs = NUM_ENVS
  env.device = device
  env.physics_dt = 0.005  # 200 Hz.
  env.cfg.decimation = 4  # 50 Hz control.
  env.common_step_counter = 0
  env.scene = {"robot": entity}
  return env


def constant_prior(env, value: float = PRIOR_POS) -> torch.Tensor:
  del env
  return torch.full((NUM_ENVS, NUM_JOINTS), value, device=get_test_device())


def make_term(mock_env, **kwargs) -> JointPositionActionWithPrior:
  # Most of these tests exercise the lam blend, which only exists under "convex"; the
  # cfg default is "nominal".
  kwargs.setdefault("blend", "convex")
  cfg = JointPositionActionWithPriorCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=1.0,
    prior=constant_prior,
    **kwargs,
  )
  return cfg.build(mock_env)


@pytest.mark.parametrize(
  "lam,expected",
  [
    (0.0, 1.5),  # Pure policy: default 0.5 + action 1.0.
    (0.5, 1.75),  # Half and half with the prior at 2.0.
    (1.0, 2.0),  # Pure prior.
  ],
)
def test_blend_weights(mock_env, device, lam, expected):
  term = make_term(mock_env, lam=lam)
  term.process_actions(torch.ones((NUM_ENVS, NUM_JOINTS), device=device))
  term.apply_actions()

  applied = mock_env.scene["robot"].set_joint_position_target.call_args[0][0]
  torch.testing.assert_close(
    applied, torch.full_like(applied, expected), rtol=0, atol=1e-5
  )


def test_prior_evaluated_at_configured_rate(mock_env, device):
  """100 Hz prior under 200 Hz physics -> every other substep."""
  calls = []

  def counting_prior(env):
    del env
    calls.append(len(calls))
    return torch.full((NUM_ENVS, NUM_JOINTS), PRIOR_POS, device=device)

  cfg = JointPositionActionWithPriorCfg(
    entity_name="robot",
    actuator_names=(".*",),
    prior=counting_prior,
    prior_frequency_hz=100.0,
  )
  term = cfg.build(mock_env)

  # Two control steps of 4 substeps each.
  for _ in range(2):
    term.process_actions(torch.zeros((NUM_ENVS, NUM_JOINTS), device=device))
    for _ in range(mock_env.cfg.decimation):
      term.apply_actions()

  assert len(calls) == 4  # 2 per control step.


def test_rejects_rate_not_dividing_physics(mock_env):
  with pytest.raises(ValueError, match="does not evenly divide"):
    make_term(mock_env, prior_frequency_hz=75.0)


def test_rejects_rate_slower_than_control(mock_env):
  with pytest.raises(ValueError, match="slower than the control rate"):
    make_term(mock_env, prior_frequency_hz=25.0)


def test_rejects_prior_with_wrong_shape(mock_env, device):
  cfg = JointPositionActionWithPriorCfg(
    entity_name="robot",
    actuator_names=(".*",),
    prior=lambda env: torch.zeros((NUM_ENVS, NUM_JOINTS + 1), device=device),
  )
  term = cfg.build(mock_env)
  term.process_actions(torch.zeros((NUM_ENVS, NUM_JOINTS), device=device))
  with pytest.raises(ValueError, match="Prior returned shape"):
    term.apply_actions()


def test_lam_curriculum_interpolates(mock_env):
  term = make_term(mock_env, lam=1.0)
  mock_env.action_manager.get_term.return_value = term

  params: dict[str, Any] = {
    "action_name": "joint_pos",
    "stages": [{"step": 0, "value": 1.0}, {"step": 100, "value": 0.0}],
  }
  curriculum = action_curriculum(
    CurriculumTermCfg(func=Mock(), params=params), mock_env
  )

  env_ids = torch.tensor([0])
  for step, expected in [(0, 1.0), (50, 0.5), (100, 0.0), (200, 0.0)]:
    mock_env.common_step_counter = step
    out = curriculum(mock_env, env_ids, **params)
    assert term.lam == pytest.approx(expected)
    assert out["lam"].item() == pytest.approx(expected)


def test_lam_curriculum_piecewise_constant(mock_env):
  term = make_term(mock_env, lam=1.0)
  mock_env.action_manager.get_term.return_value = term

  params: dict[str, Any] = {
    "action_name": "joint_pos",
    "stages": [{"step": 0, "value": 1.0}, {"step": 100, "value": 0.0}],
    "interpolate": False,
  }
  curriculum = action_curriculum(
    CurriculumTermCfg(func=Mock(), params=params), mock_env
  )

  mock_env.common_step_counter = 50
  curriculum(mock_env, torch.tensor([0]), **params)
  assert term.lam == pytest.approx(1.0)


def test_prior_defaults_to_control_rate(mock_env, device):
  """prior_frequency_hz=None -> once per control step, in lockstep with pi."""
  calls = []

  def counting_prior(env):
    del env
    calls.append(len(calls))
    return torch.full((NUM_ENVS, NUM_JOINTS), PRIOR_POS, device=device)

  cfg = JointPositionActionWithPriorCfg(
    entity_name="robot",
    actuator_names=(".*",),
    prior=counting_prior,
  )
  term = cfg.build(mock_env)
  assert cfg.prior_frequency_hz is None

  for _ in range(3):
    term.process_actions(torch.zeros((NUM_ENVS, NUM_JOINTS), device=device))
    for _ in range(mock_env.cfg.decimation):
      term.apply_actions()

  assert len(calls) == 3


def test_targets_survive_reset(mock_env, device):
  """The cached targets are not clobbered by a reset (they are pure caches)."""
  term = make_term(mock_env, lam=1.0)
  term.process_actions(torch.ones((NUM_ENVS, NUM_JOINTS), device=device))
  term.apply_actions()
  before = term.blended_target.clone()

  term.reset(torch.arange(NUM_ENVS, device=device))
  torch.testing.assert_close(term.blended_target, before)


@pytest.mark.parametrize(
  "blend,expected",
  [
    ("convex", 1.75),  # lam = 0.5: half the policy's 1.5, half the prior's 2.0.
    ("residual", 3.5),  # u_prior + pi(o) = 2.0 + 1.5.
    ("nominal", 1.5),  # Pure policy: the prior is evaluated but never applied.
  ],
)
def test_blend_modes(mock_env, device, blend, expected):
  term = make_term(mock_env, lam=0.5, blend=blend)
  term.process_actions(torch.ones((NUM_ENVS, NUM_JOINTS), device=device))
  term.apply_actions()

  applied = mock_env.scene["robot"].set_joint_position_target.call_args[0][0]
  torch.testing.assert_close(
    applied, torch.full_like(applied, expected), rtol=0, atol=1e-5
  )


def test_nominal_still_publishes_the_prior(mock_env, device):
  """The prior is evaluated under "nominal" so rewards can read it."""
  calls = []

  def counting_prior(env):
    del env
    calls.append(len(calls))
    return torch.full((NUM_ENVS, NUM_JOINTS), PRIOR_POS, device=device)

  cfg = JointPositionActionWithPriorCfg(
    entity_name="robot",
    actuator_names=(".*",),
    prior=counting_prior,
    blend="nominal",
  )
  term = cfg.build(mock_env)
  term.process_actions(torch.ones((NUM_ENVS, NUM_JOINTS), device=device))
  term.apply_actions()

  assert len(calls) == 1
  torch.testing.assert_close(
    term.prior_target, torch.full_like(term.prior_target, PRIOR_POS)
  )
  # lam is irrelevant under "nominal": the applied target does not move with it.
  before = term.blended_target.clone()
  term.lam = 1.0
  term.apply_actions()
  torch.testing.assert_close(term.blended_target, before)
