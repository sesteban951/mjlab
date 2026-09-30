"""Unitree G1 unicycle walking with an LSTM actor, on the backpack robot.

``G1-Walk-Unicycle-LSTM`` with one change: the robot carries the compute backpack
(``g1_constants_backpack``: a 1.4 kg box on the back of the torso). Only the entity's spec
function moves; the mode-11 actuator table, action scale, HOME keyframe, collision policy,
shared DR and every task term come through from the LSTM parent unchanged. The backpack's
collision geom matches the ``.*_collision`` policy already on the entity, so it collides
like any other non-foot link. Its mass is NOT randomized: the shared DR's pseudo-inertia
term is scoped to ``torso_link``.
"""

from mjlab.asset_zoo.robots.unitree_g1.g1_constants_backpack import get_backpack_spec
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.walk_unicycle.config.g1_lstm.env_cfgs import (
  unitree_g1_walk_unicycle_lstm_env_cfg,
)


def unitree_g1_walk_unicycle_lstm_backpack_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """``G1-Walk-Unicycle-LSTM`` on the backpack robot."""
  cfg = unitree_g1_walk_unicycle_lstm_env_cfg(play=play)
  cfg.scene.entities["robot"].spec_fn = get_backpack_spec
  return cfg
