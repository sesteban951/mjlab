"""G1 jog + run + sprint on the robot carrying the compute backpack.

``G1-JogRunSprint-Unicycle`` with one change: the robot's spec adds the compute backpack
(``g1_constants_backpack``: a 1.5 kg box on the back of the torso). Everything else
comes through from the parent unchanged, and the backpack's mass is not randomized
(the shared DR is scoped to ``torso_link``).
"""

from mjlab.asset_zoo.robots.unitree_g1.g1_constants_backpack import get_backpack_spec
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.jogrunsprint_custom.config.g1.env_cfgs import (
  unitree_g1_jogrunsprint_env_cfg,
)


def unitree_g1_jogrunsprint_backpack_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """``G1-JogRunSprint-Unicycle`` on the backpack robot."""
  cfg = unitree_g1_jogrunsprint_env_cfg(play=play)
  cfg.scene.entities["robot"].spec_fn = get_backpack_spec
  return cfg
