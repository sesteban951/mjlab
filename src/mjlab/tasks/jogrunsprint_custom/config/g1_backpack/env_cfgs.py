"""G1 jog + run + sprint on the robot carrying the rear backpack.

``G1-JogRunSprint-Custom`` with one change: the robot's spec adds the compute backpack
in its rear mounting (``g1_constants_backpack.get_rear_backpack_spec``: 1.47 kg, 5 cm
behind the torso shell). Everything else comes through from the parent unchanged, and
the backpack's mass is not randomized (the shared DR is scoped to ``torso_link``).
"""

from mjlab.asset_zoo.robots.unitree_g1.g1_constants_backpack import (
  get_rear_backpack_spec,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.jogrunsprint_custom.config.g1.env_cfgs import (
  unitree_g1_jogrunsprint_env_cfg,
)


def unitree_g1_jogrunsprint_backpack_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """``G1-JogRunSprint-Custom`` on the rear-backpack robot."""
  cfg = unitree_g1_jogrunsprint_env_cfg(play=play)
  cfg.scene.entities["robot"].spec_fn = get_rear_backpack_spec
  return cfg
