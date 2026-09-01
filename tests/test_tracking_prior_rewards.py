"""Tests for the prior-relative rewards: L1 behavior cloning and LQR shaping."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.tracking_prior.mdp.rewards import (
  action_prior_deviation_exp,
  lqr_lyapunov_shaping,
)

NUM_ENVS = 2
NUM_JOINTS = 3
NDX = 2 * (6 + NUM_JOINTS)


@pytest.fixture
def device():
  return get_test_device()


class FakeScene:
  """Scene stub: indexable by entity name and carrying ``env_origins``."""

  def __init__(self, entity, env_origins):
    self._entity = entity
    self.env_origins = env_origins

  def __getitem__(self, name):
    del name
    return self._entity


def make_exp_term(device, policy, prior, lam):
  term = Mock()
  term.processed_actions = policy
  term.prior_target = prior
  term.prior_weight = lam
  # isinstance() against the real class is what the reward checks, so patch it out.
  env = Mock(num_envs=NUM_ENVS, device=device)
  env.action_manager.get_term.return_value = term
  return env, term


def test_deviation_exp(device, monkeypatch):
  import mjlab.tasks.tracking_prior.mdp.rewards as rewards_mod

  policy = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], device=device)
  prior = torch.tensor([[1.5, 2.0, 1.0], [0.0, 0.0, 0.0]], device=device)
  env, _ = make_exp_term(device, policy, prior, lam=0.5)
  monkeypatch.setattr(rewards_mod, "JointPositionActionWithPrior", Mock)

  std = 0.5
  params = {"std": std}
  reward = action_prior_deviation_exp(
    RewardTermCfg(func=Mock(), weight=1.0, params=params), env
  )

  # mean(0.5^2, 0, 2^2) = 4.25/3 for env 0; env 1 matches the prior exactly, so 1.0.
  expected = torch.tensor([float(np.exp(-(4.25 / 3) / std**2)), 1.0], device=device)
  out = reward(env, std=std, fade_with_prior=False)
  torch.testing.assert_close(out, expected)
  # Bounded in (0, 1]: a bonus, not a penalty.
  assert bool(((out > 0.0) & (out <= 1.0)).all())
  # Faded, it picks up the prior weight; it does not by default.
  out = reward(env, std=std, fade_with_prior=True)
  torch.testing.assert_close(out, 0.5 * expected)


def test_deviation_exp_rejects_bad_std(device, monkeypatch):
  import mjlab.tasks.tracking_prior.mdp.rewards as rewards_mod

  zeros = torch.zeros((NUM_ENVS, NUM_JOINTS), device=device)
  env, _ = make_exp_term(device, zeros, zeros, lam=0.0)
  monkeypatch.setattr(rewards_mod, "JointPositionActionWithPrior", Mock)
  with pytest.raises(ValueError, match="std must be positive"):
    action_prior_deviation_exp(
      RewardTermCfg(func=Mock(), weight=1.0, params={"std": 0.0}), env
    )


def write_tape(path, frames=4, p=None):
  ref_qpos = np.zeros((frames, 7 + NUM_JOINTS), dtype=np.float32)
  ref_qpos[:, 3] = 1.0  # Identity quaternion, wxyz.
  ref_qvel = np.zeros((frames, 6 + NUM_JOINTS), dtype=np.float32)
  if p is None:
    np.savez(path, ref_qpos=ref_qpos, ref_qvel=ref_qvel)
  else:
    np.savez(path, ref_qpos=ref_qpos, ref_qvel=ref_qvel, P=p)
  return str(path)


def make_shaping_env(device, error_scale):
  """Env whose robot sits at ``error_scale`` on every tangent coordinate."""
  entity = Mock()
  entity.data.joint_pos = torch.full((NUM_ENVS, NUM_JOINTS), error_scale, device=device)
  entity.data.joint_vel = torch.zeros((NUM_ENVS, NUM_JOINTS), device=device)
  entity.data.root_link_pos_w = torch.zeros((NUM_ENVS, 3), device=device)
  entity.data.root_link_quat_w = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0]] * NUM_ENVS, device=device
  )
  entity.data.root_link_lin_vel_w = torch.zeros((NUM_ENVS, 3), device=device)
  entity.data.root_link_ang_vel_w = torch.zeros((NUM_ENVS, 3), device=device)

  env = Mock(num_envs=NUM_ENVS, device=device)
  env.scene = FakeScene(entity, torch.zeros((NUM_ENVS, 3), device=device))
  command = Mock()
  command.time_steps = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
  env.command_manager.get_term.return_value = command
  return env, entity


def test_lyapunov_shaping_pays_for_descent(device, tmp_path):
  tape = write_tape(tmp_path / "tape.npz", p=np.eye(NDX, dtype=np.float32))
  env, entity = make_shaping_env(device, error_scale=1.0)
  params = {"tape_file": tape, "gamma": 1.0, "kappa": 1.0, "clip_to_rho_sat": False}
  reward = lqr_lyapunov_shaping(
    RewardTermCfg(func=Mock(), weight=1.0, params=params), env
  )

  # First step of an episode: nothing to have descended from, so exactly zero.
  out = reward(env, gamma=1.0, tape_file=tape, kappa=1.0, clip_to_rho_sat=False)
  torch.testing.assert_close(out, torch.zeros(NUM_ENVS, device=device))

  # s^T I s with 3 joint-position coordinates at 1.0 -> 3.0. Halving the error to 0.5
  # takes the quadratic to 0.75, so the agent is paid 3.0 - 0.75 = 2.25 for descending.
  entity.data.joint_pos = torch.full((NUM_ENVS, NUM_JOINTS), 0.5, device=device)
  out = reward(env, gamma=1.0, tape_file=tape, kappa=1.0, clip_to_rho_sat=False)
  torch.testing.assert_close(out, torch.full((NUM_ENVS,), 2.25, device=device))

  # Climbing back out is penalized by exactly what descending paid.
  entity.data.joint_pos = torch.full((NUM_ENVS, NUM_JOINTS), 1.0, device=device)
  out = reward(env, gamma=1.0, tape_file=tape, kappa=1.0, clip_to_rho_sat=False)
  torch.testing.assert_close(out, torch.full((NUM_ENVS,), -2.25, device=device))


def test_lyapunov_shaping_resets_per_episode(device, tmp_path):
  tape = write_tape(tmp_path / "tape.npz", p=np.eye(NDX, dtype=np.float32))
  env, _ = make_shaping_env(device, error_scale=1.0)
  params = {"tape_file": tape, "gamma": 0.99, "clip_to_rho_sat": False}
  reward = lqr_lyapunov_shaping(
    RewardTermCfg(func=Mock(), weight=1.0, params=params), env
  )

  reward(env, gamma=0.99, tape_file=tape, clip_to_rho_sat=False)
  reward.reset(env_ids=torch.tensor([0], device=device))
  out = reward(env, gamma=0.99, tape_file=tape, clip_to_rho_sat=False)
  # Env 0 restarted, so it is scored as a first step; env 1 carries its potential over.
  assert out[0].item() == 0.0
  assert out[1].item() == pytest.approx(3.0 - 0.99 * 3.0, abs=1e-5)


def test_lyapunov_shaping_requires_P(device, tmp_path):
  tape = write_tape(tmp_path / "tape.npz", p=None)
  env, _ = make_shaping_env(device, error_scale=1.0)
  params = {"tape_file": tape, "gamma": 0.99, "clip_to_rho_sat": False}
  with pytest.raises(KeyError, match="no 'P' array"):
    lqr_lyapunov_shaping(RewardTermCfg(func=Mock(), weight=1.0, params=params), env)


def test_lyapunov_shaping_rejects_wrong_P_shape(device, tmp_path):
  tape = write_tape(tmp_path / "tape.npz", p=np.eye(NDX + 1, dtype=np.float32))
  env, _ = make_shaping_env(device, error_scale=1.0)
  params = {"tape_file": tape, "gamma": 0.99, "clip_to_rho_sat": False}
  with pytest.raises(ValueError, match="expected"):
    lqr_lyapunov_shaping(RewardTermCfg(func=Mock(), weight=1.0, params=params), env)


def test_lyapunov_shaping_clips_at_rho_sat(device, tmp_path):
  """The clip flattens the potential, which is what bounds the (1-gamma) leak."""
  path = tmp_path / "tape.npz"
  ref_qpos = np.zeros((4, 7 + NUM_JOINTS), dtype=np.float32)
  ref_qpos[:, 3] = 1.0
  np.savez(
    path,
    ref_qpos=ref_qpos,
    ref_qvel=np.zeros((4, 6 + NUM_JOINTS), dtype=np.float32),
    P=np.eye(NDX, dtype=np.float32),
    rho_sat=np.full(4, 1.0, dtype=np.float32),
  )
  tape = str(path)
  env, entity = make_shaping_env(device, error_scale=1.0)
  params = {"tape_file": tape, "gamma": 1.0, "clip_to_rho_sat": True}
  reward = lqr_lyapunov_shaping(
    RewardTermCfg(func=Mock(), weight=1.0, params=params), env
  )

  # Unclipped this state is worth 3.0 (three joints at 1.0); rho_sat caps it at 1.0.
  reward(env, gamma=1.0, tape_file=tape)
  # Moving further out cannot change a clipped potential, so the shaping is exactly 0.
  entity.data.joint_pos = torch.full((NUM_ENVS, NUM_JOINTS), 5.0, device=device)
  out = reward(env, gamma=1.0, tape_file=tape)
  torch.testing.assert_close(out, torch.zeros(NUM_ENVS, device=device))

  # Coming back inside the radius still pays: 1.0 (clipped) -> 0.03 (3 * 0.1^2).
  entity.data.joint_pos = torch.full((NUM_ENVS, NUM_JOINTS), 0.1, device=device)
  out = reward(env, gamma=1.0, tape_file=tape)
  torch.testing.assert_close(out, torch.full((NUM_ENVS,), 0.97, device=device))


def test_lyapunov_shaping_requires_rho_sat_when_clipping(device, tmp_path):
  tape = write_tape(tmp_path / "tape.npz", p=np.eye(NDX, dtype=np.float32))
  env, _ = make_shaping_env(device, error_scale=1.0)
  params = {"tape_file": tape, "gamma": 0.99, "clip_to_rho_sat": True}
  with pytest.raises(KeyError, match="rho_sat"):
    lqr_lyapunov_shaping(RewardTermCfg(func=Mock(), weight=1.0, params=params), env)


def test_lyapunov_shaping_saturating_potential(device, tmp_path):
  """v_half keeps a gradient far outside rho_sat, where the hard clip goes flat."""
  tape = write_tape(tmp_path / "tape.npz", p=np.eye(NDX, dtype=np.float32))
  env, entity = make_shaping_env(device, error_scale=1.0)
  v_half = 3.0
  params = {"tape_file": tape, "gamma": 1.0, "kappa": 1.0, "v_half": v_half}
  reward = lqr_lyapunov_shaping(
    RewardTermCfg(func=Mock(), weight=1.0, params=params), env
  )

  def h(v):
    return v / (v + v_half)

  # Three joints at 1.0 -> V = 3.0; the potential is h(3) = 0.5, not the raw quadratic.
  reward(env, gamma=1.0, tape_file=tape, kappa=1.0, v_half=v_half)
  # Moving far out still changes the potential: V = 3*25 = 75 -> h(75) != h(3).
  entity.data.joint_pos = torch.full((NUM_ENVS, NUM_JOINTS), 5.0, device=device)
  out = reward(env, gamma=1.0, tape_file=tape, kappa=1.0, v_half=v_half)
  expected = h(3.0) - h(75.0)
  torch.testing.assert_close(out, torch.full((NUM_ENVS,), expected, device=device))
  assert abs(expected) > 1e-3, "a hard clip would give exactly 0 here"

  # Bounded: the potential lives in (-kappa, 0], so one step can never pay more than kappa.
  assert bool((out.abs() <= 1.0).all())


def test_lyapunov_shaping_v_half_overrides_clip(device, tmp_path):
  """v_half makes the rho_sat clip redundant, so a tape without rho_sat is fine."""
  tape = write_tape(tmp_path / "tape.npz", p=np.eye(NDX, dtype=np.float32))
  env, _ = make_shaping_env(device, error_scale=1.0)
  params = {"tape_file": tape, "gamma": 0.99, "v_half": 50.0}
  reward = lqr_lyapunov_shaping(
    RewardTermCfg(func=Mock(), weight=1.0, params=params), env
  )
  assert reward._rho is None


def test_lyapunov_shaping_rejects_bad_v_half(device, tmp_path):
  tape = write_tape(tmp_path / "tape.npz", p=np.eye(NDX, dtype=np.float32))
  env, _ = make_shaping_env(device, error_scale=1.0)
  params = {"tape_file": tape, "gamma": 0.99, "v_half": 0.0}
  with pytest.raises(ValueError, match="v_half must be positive"):
    lqr_lyapunov_shaping(RewardTermCfg(func=Mock(), weight=1.0, params=params), env)
