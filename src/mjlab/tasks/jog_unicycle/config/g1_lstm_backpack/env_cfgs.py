"""Unitree G1 unicycle jogging with an LSTM actor, on the backpack robot.

``G1-Jog-Unicycle-LSTM`` with one change: the robot carries the compute backpack
(``g1_constants_backpack``: a 1.4 kg box on the back of the torso), exactly as
``G1-Walk-Unicycle-LSTM-Backpack`` does for the walk. Only the entity's spec function
moves; everything else comes through from the LSTM parent unchanged. The backpack's mass
is NOT randomized: the shared DR's pseudo-inertia term is scoped to ``torso_link``.
"""

from mjlab.asset_zoo.robots.unitree_g1.g1_constants_backpack import get_backpack_spec
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.jog_unicycle.config.g1_lstm.env_cfgs import (
  unitree_g1_jog_unicycle_lstm_env_cfg,
)


def unitree_g1_jog_unicycle_lstm_backpack_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """``G1-Jog-Unicycle-LSTM`` on the backpack robot."""
  cfg = unitree_g1_jog_unicycle_lstm_env_cfg(play=play)
  cfg.scene.entities["robot"].spec_fn = get_backpack_spec
  return cfg
