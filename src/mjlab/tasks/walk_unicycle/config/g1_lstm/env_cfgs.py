"""Unitree G1 unicycle walking with an LSTM actor: the base env, verbatim.

``G1-Walk-Unicycle`` with memory in the network instead of the observation: the LSTM
in ``rl_cfg`` gets ONE frame per step and keeps its own state across the episode, so
the observation cfg must NOT stack history. Everything else (library, command,
rewards, DR, critic) is the base task's. See ``rl_cfg.py`` for the network sizing.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.walk_unicycle.config.g1.env_cfgs import (
  unitree_g1_walk_unicycle_env_cfg,
)


def unitree_g1_walk_unicycle_lstm_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """``G1-Walk-Unicycle`` unchanged; the LSTM lives in the RL config."""
  return unitree_g1_walk_unicycle_env_cfg(play=play)
