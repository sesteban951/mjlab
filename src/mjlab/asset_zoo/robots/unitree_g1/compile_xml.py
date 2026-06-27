"""Compile the Unitree G1 entity to a flat XML.

Builds the G1 robot entity (which injects the actuators, collision edits, and
initial-state keyframe that ``g1_constants.py`` defines in Python on top of the
raw ``g1.xml``) and serializes the result to ``xmls/g1_compiled.xml``, next to
the source assets so the relative ``meshdir`` keeps resolving.

Run with::

    uv run python -m mjlab.asset_zoo.robots.unitree_g1.compile_xml
"""

import mujoco

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML, get_g1_robot_cfg
from mjlab.entity.entity import Entity

# Always write the compiled model here, regardless of the current directory.
OUTPUT_XML = G1_XML.parent / "g1_compiled.xml"


def main() -> None:
  robot = Entity(get_g1_robot_cfg())
  robot.write_xml(OUTPUT_XML)

  # Verify the serialized XML recompiles from disk (resolves meshes/keyframe).
  model = mujoco.MjModel.from_xml_path(str(OUTPUT_XML))
  print(
    f"Wrote {OUTPUT_XML} "
    f"(nbody={model.nbody}, njnt={model.njnt}, nu={model.nu}, "
    f"nq={model.nq}, nkey={model.nkey})"
  )


if __name__ == "__main__":
  main()
