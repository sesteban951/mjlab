"""Compile a robot entity to a flat XML, written next to its source XML.

Building the entity injects the actuators, collision edits and initial-state
keyframe that the robot's constants file defines on top of the raw XML.

    uv run python -m mjlab.asset_zoo.robots.compile_xml g1
    uv run python -m mjlab.asset_zoo.robots.compile_xml g1_mode11
    uv run python -m mjlab.asset_zoo.robots.compile_xml g1_backpack
    uv run python -m mjlab.asset_zoo.robots.compile_xml go2
"""

from pathlib import Path
from typing import Callable, Literal

import mujoco
import tyro

import mjlab
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML, get_g1_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants_backpack import (
  get_g1_backpack_robot_cfg,
)
from mjlab.asset_zoo.robots.unitree_g1.g1_constants_mode11 import (
  get_g1_mode11_robot_cfg,
)
from mjlab.asset_zoo.robots.unitree_go2.go2_constants import (
  GO2_XML,
  get_go2_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.entity.entity import Entity

Robot = Literal["g1", "g1_mode11", "g1_backpack", "go2"]

# robot -> (cfg factory, output path); output sits by the source so meshdir resolves
ROBOTS: dict[str, tuple[Callable[[], EntityCfg], Path]] = {
  "g1": (get_g1_robot_cfg, G1_XML.parent / "g1_compiled.xml"),
  "g1_mode11": (get_g1_mode11_robot_cfg, G1_XML.parent / "g1_mode11_compiled.xml"),
  "g1_backpack": (
    get_g1_backpack_robot_cfg,
    G1_XML.parent / "g1_backpack_compiled.xml",
  ),
  "go2": (get_go2_robot_cfg, GO2_XML.parent / "go2_compiled.xml"),
}


def main(robot: tyro.conf.Positional[Robot]) -> None:
  """Write the compiled XML for `robot` next to its source assets."""
  cfg_fn, output_xml = ROBOTS[robot]
  Entity(cfg_fn()).write_xml(output_xml)

  # Verify the serialized XML recompiles from disk (resolves meshes/keyframe).
  model = mujoco.MjModel.from_xml_path(str(output_xml))
  print(
    f"Wrote {output_xml} "
    f"(nbody={model.nbody}, njnt={model.njnt}, nu={model.nu}, "
    f"nq={model.nq}, nkey={model.nkey})"
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
