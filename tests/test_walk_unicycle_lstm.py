"""G1-Walk-Unicycle-LSTM: recurrent actor cfg, sizing, play-time reset, ONNX export."""

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

import mjlab.tasks  # noqa: F401  Populate the task registry.
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.rl.recurrent import ResetOnEpisodeStart, wrap_for_inference
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.walk_unicycle.config.g1_lstm.rl_cfg import (
  HEAD_HIDDEN_DIMS,
  LSTM_HIDDEN_DIM,
)

TASK = "G1-Walk-Unicycle-LSTM"
BASE = "G1-Walk-Unicycle"


def test_lstm_actor_mlp_critic_same_ppo() -> None:
  cfg, base = load_rl_cfg(TASK), load_rl_cfg(BASE)
  assert isinstance(cfg, RslRlOnPolicyRunnerCfg)
  assert isinstance(base, RslRlOnPolicyRunnerCfg)
  assert cfg.actor.class_name == "RNNModel" and cfg.actor.rnn_type == "lstm"
  assert cfg.actor.rnn_hidden_dim == LSTM_HIDDEN_DIM
  assert cfg.actor.hidden_dims == HEAD_HIDDEN_DIMS
  # The A/B isolates memory: critic and PPO settings are the baseline's, verbatim.
  assert cfg.critic == base.critic and cfg.critic.class_name == "MLPModel"
  assert cfg.algorithm == base.algorithm
  assert cfg.num_steps_per_env == base.num_steps_per_env
  assert cfg.experiment_name == "g1_walk_unicycle_lstm"


def test_lstm_env_is_the_base_env_without_observation_history() -> None:
  cfg = load_env_cfg(TASK)
  for name, term in cfg.observations["actor"].terms.items():
    assert term.history_length <= 1, f"{name} stacks history; the LSTM is the memory"
  assert cfg.commands["motion"] == load_env_cfg(BASE).commands["motion"]


class _FakeRecurrent:
  is_recurrent = True

  def __init__(self) -> None:
    self.resets: list[torch.Tensor | None] = []

  def reset(self, dones: torch.Tensor | None = None) -> None:
    self.resets.append(None if dones is None else dones.clone())

  def __call__(self, obs: torch.Tensor) -> torch.Tensor:
    return obs


class _FakeEnv:
  def __init__(self, lengths: list[int]) -> None:
    self.episode_length_buf = torch.tensor(lengths)


def test_wrapper_resets_exactly_the_fresh_envs() -> None:
  policy, env = _FakeRecurrent(), _FakeEnv([0, 5, 0, 12])
  wrapped = wrap_for_inference(policy, env)
  assert isinstance(wrapped, ResetOnEpisodeStart)
  wrapped(torch.zeros(4, 3))
  assert len(policy.resets) == 1
  first = policy.resets[0]
  assert first is not None and first.tolist() == [True, False, True, False]
  env.episode_length_buf[:] = 3
  wrapped(torch.zeros(4, 3))
  assert len(policy.resets) == 1  # nothing fresh -> no reset call
  wrapped.reset()  # the viewer's manual full reset forwards as a full reset
  assert policy.resets[-1] is None
  assert wrapped.is_recurrent  # attributes forward to the policy


def test_wrapper_leaves_feedforward_policies_alone() -> None:
  class _Mlp:
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
      return obs

  policy = _Mlp()
  assert wrap_for_inference(policy, _FakeEnv([0])) is policy


@pytest.mark.slow
def test_lstm_policy_exports_with_state_io(tmp_path: Path) -> None:
  import onnx

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper

  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = 2
  rl_cfg = load_rl_cfg(TASK)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device="cpu"), clip_actions=rl_cfg.clip_actions
  )
  runner_cls = load_runner_cls(TASK)
  assert runner_cls is not None
  runner = runner_cls(env, asdict(rl_cfg), device="cpu")
  policy = runner.alg.get_policy()
  assert policy.is_recurrent
  runner.export_policy_to_onnx(str(tmp_path), "policy.onnx")
  model = onnx.load(str(tmp_path / "policy.onnx"))
  assert [i.name for i in model.graph.input] == ["obs", "h_in", "c_in"]
  assert [o.name for o in model.graph.output] == ["actions", "h_out", "c_out"]
  env.close()
