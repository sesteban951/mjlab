"""Unitree G1 unicycle walking with an LSTM actor: the base env, verbatim.

``G1-Walk-Unicycle`` with memory moved from the observation into the network. Where
``G1-Walk-Unicycle-History`` stacks the actor's dynamics terms over 5 frames (a fixed
100 ms window into an MLP), this task feeds the LSTM in ``rl_cfg`` ONE frame per step
and lets it keep its own state across the episode -- so the observation cfg must NOT
stack history, or the two would be confounded. Everything else (library, command,
rewards, DR, critic) is the base task's. See ``rl_cfg.py`` for the network sizing and
its rationale.

"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.walk_unicycle.config.g1.env_cfgs import (
  unitree_g1_walk_unicycle_env_cfg,
)


def unitree_g1_walk_unicycle_lstm_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """``G1-Walk-Unicycle`` unchanged; the LSTM lives in the RL config."""
  return unitree_g1_walk_unicycle_env_cfg(play=play)
