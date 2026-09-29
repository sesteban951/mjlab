"""Unitree G1 unicycle jogging with an LSTM actor: the base env, verbatim.

``G1-Jog-Unicycle`` with memory in the network rather than the observation, exactly as
``G1-Walk-Unicycle-LSTM`` does for the walk. The LSTM in ``rl_cfg`` gets ONE frame per
step and keeps its own state across the episode, so the observation cfg must NOT stack
history, or the two would be confounded. Everything else (library, command, rewards,
DR, critic) is the base task's.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.jog_unicycle.config.g1.env_cfgs import (
  unitree_g1_jog_unicycle_env_cfg,
)


def unitree_g1_jog_unicycle_lstm_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """``G1-Jog-Unicycle`` unchanged; the LSTM lives in the RL config."""
  return unitree_g1_jog_unicycle_env_cfg(play=play)
