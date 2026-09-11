"""Compile the Unitree G1 entity to a flat XML.

Builds the G1 robot entity (which injects the actuators, collision edits, and
initial-state keyframe that ``g1_constants.py`` defines in Python on top of the
raw ``g1.xml``) and serializes the result to ``xmls/g1_compiled.xml``, next to
the source assets so the relative ``meshdir`` keeps resolving.

``--mode11 True`` compiles the mode_machine 11 variant instead
(``g1_constants_mode11.py``: hip pitch on the 22.5:1 gearbox) to
``xmls/g1_mode11_compiled.xml``.

Run with::

    uv run python -m mjlab.asset_zoo.robots.unitree_g1.compile_xml
    uv run python -m mjlab.asset_zoo.robots.unitree_g1.compile_xml --mode11 True
"""

import mujoco
import tyro

import mjlab
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML, get_g1_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants_mode11 import (
  get_g1_mode11_robot_cfg,
)
from mjlab.entity.entity import Entity


def main(mode11: bool = False) -> None:
  """Write the compiled G1 XML next to the source assets.

  Args:
    mode11: compile the mode_machine 11 actuator table (hip pitch = 7520_22) instead of
      the base rev_1_0 one.
  """
  # Always write the compiled model here, regardless of the current directory.
  output_xml = G1_XML.parent / (
    "g1_mode11_compiled.xml" if mode11 else "g1_compiled.xml"
  )
  robot = Entity(get_g1_mode11_robot_cfg() if mode11 else get_g1_robot_cfg())
  robot.write_xml(output_xml)

  # Verify the serialized XML recompiles from disk (resolves meshes/keyframe).
  model = mujoco.MjModel.from_xml_path(str(output_xml))
  print(
    f"Wrote {output_xml} "
    f"(nbody={model.nbody}, njnt={model.njnt}, nu={model.nu}, "
    f"nq={model.nq}, nkey={model.nkey})"
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
