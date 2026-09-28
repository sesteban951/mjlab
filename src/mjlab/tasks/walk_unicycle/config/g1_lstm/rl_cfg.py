"""RL config for the Unitree G1 unicycle walking task with an LSTM actor."""

from dataclasses import replace

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.walk_unicycle.config.g1.rl_cfg import (
  unitree_g1_walk_unicycle_ppo_runner_cfg,
)

# SIZING. The LSTM replaces the baseline actor's 512-wide input layer and the rest of
# the head is the baseline's, so the A/B against G1-Walk-Unicycle isolates memory, not
# capacity:
#
#   baseline  98 -> 512 -> 256 -> 128 -> 29            ~219k params
#   this      98 -> LSTM(128) -> 256 -> 128 -> 29      ~186k params
#
# 128 sits between Unitree's shipped LSTM-64 (unitree_rl_gym's G1: a 47-dim, 12-DoF
# velocity policy) and rsl_rl's 256 default. What the state has to hold is small -- a
# body-velocity estimate (the actor sees no base_lin_vel), the contact state, and the
# per-episode DR draws (actuator delay, PD gains, COM, mass, friction) -- so 128 is
# comfortable and trains faster and more stably than 256. The critic stays the baseline
# MLP: it already sees privileged state, so it has nothing to remember, and changing it
# would confound the comparison. PPO settings are the baseline's too; the 24-step
# rollout is the BPTT chunk, and rsl_rl carries hidden state across chunks, so the
# memory horizon is not limited to it.
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 1
HEAD_HIDDEN_DIMS = (256, 128)


def unitree_g1_walk_unicycle_lstm_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_walk_unicycle_ppo_runner_cfg()
  cfg.actor = replace(
    cfg.actor,
    class_name="RNNModel",
    rnn_type="lstm",
    rnn_hidden_dim=LSTM_HIDDEN_DIM,
    rnn_num_layers=LSTM_NUM_LAYERS,
    hidden_dims=HEAD_HIDDEN_DIMS,
  )
  cfg.experiment_name = "g1_walk_unicycle_lstm"
  return cfg
