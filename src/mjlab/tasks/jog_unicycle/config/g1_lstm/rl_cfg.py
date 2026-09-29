"""RL config for the Unitree G1 unicycle jogging task with an LSTM actor."""

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.jog_unicycle.config.g1.rl_cfg import (
  unitree_g1_jog_unicycle_ppo_runner_cfg,
)
from mjlab.tasks.walk_unicycle.config.g1_lstm.rl_cfg import (
  HEAD_HIDDEN_DIMS,
  LSTM_HIDDEN_DIM,
  LSTM_NUM_LAYERS,
)

# The walk LSTM's sizing, shared rather than copied: the jog actor sees the same 98-dim
# observation as the walk one, and the walk LSTM is the variant proven on hardware, so the
# two stay in lock-step. See walk_unicycle/config/g1_lstm/rl_cfg.py for the rationale.


def unitree_g1_jog_unicycle_lstm_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_jog_unicycle_ppo_runner_cfg()
  cfg.actor = replace(
    cfg.actor,
    class_name="RNNModel",
    rnn_type="lstm",
    rnn_hidden_dim=LSTM_HIDDEN_DIM,
    rnn_num_layers=LSTM_NUM_LAYERS,
    hidden_dims=HEAD_HIDDEN_DIMS,
  )
  cfg.experiment_name = "g1_jog_unicycle_lstm"
  return cfg
