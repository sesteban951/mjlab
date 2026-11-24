"""Unitree G1 constants for the mode_machine 11 robot carrying a compute backpack.

Child of :mod:`g1_constants_mode11`: same ``g1.xml``, actuator table and action scale, plus
one body on ``torso_link``: a 55 x 115 x 120 mm box of 1.5 kg, centred on the back of the
torso and 23.5 mm behind its logo plate. The mass sits on the collision geom and MuJoCo
derives the uniform-density box inertia from it. That geom is named ``backpack_collision``
so every ``.*_collision`` CollisionCfg in this package picks it up.
"""

import mujoco

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  FULL_COLLISION,
  KNEES_BENT_KEYFRAME,
  SPHERE_FEET_COLLISION,
  get_spec,
  get_sphere_feet_spec,
)
from mjlab.asset_zoo.robots.unitree_g1.g1_constants_mode11 import (
  G1_MODE11_ACTION_SCALE,
  G1_MODE11_ARTICULATION,
)
from mjlab.entity import EntityCfg

##
# Backpack: the one change against g1_constants_mode11.
##

BACKPACK_BODY_NAME = "backpack_link"
BACKPACK_PARENT_BODY_NAME = "torso_link"
BACKPACK_MASS = 1.5
"""Total mass of the box in kg, spread at uniform density."""
BACKPACK_SIZE = (0.055, 0.115, 0.12)
"""Full extents (x depth, y width, z height) in metres."""
BACKPACK_POS = (-0.12, 0.0, 0.1775)
"""Box centre in the ``torso_link`` frame.

The torso's logo plate sits at x = -0.069 over z in [0.14, 0.26], so the box's front face
at x = -0.0925 leaves a 23.5 mm gap to it. The 120 mm height spans z in [0.1175, 0.2375].
"""
BACKPACK_MATERIAL = "black"
"""g1.xml's material for the logo and head, rgba 0.2 0.2 0.2."""


def add_backpack(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Attach the backpack body to ``torso_link`` in ``spec`` and return ``spec``."""
  half = tuple(s / 2 for s in BACKPACK_SIZE)
  torso = spec.body(BACKPACK_PARENT_BODY_NAME)
  body = torso.add_body(name=BACKPACK_BODY_NAME, pos=BACKPACK_POS)
  body.add_geom(
    spec.find_default("visual"),
    name="backpack_visual",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=half,
    material=BACKPACK_MATERIAL,
  )
  # Carries the mass: MuJoCo computes the uniform-density box inertia from it.
  body.add_geom(
    spec.find_default("collision"),
    name="backpack_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=half,
    mass=BACKPACK_MASS,
  )
  return spec


def get_backpack_spec() -> mujoco.MjSpec:
  return add_backpack(get_spec())


def get_backpack_sphere_feet_spec() -> mujoco.MjSpec:
  return add_backpack(get_sphere_feet_spec())


##
# Final config.
##


def get_g1_backpack_robot_cfg() -> EntityCfg:
  """Mode-11 G1 with the backpack and mjlab's stock 7-capsule sole."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_backpack_spec,
    articulation=G1_MODE11_ARTICULATION,
  )


def get_g1_backpack_sphere_feet_robot_cfg() -> EntityCfg:
  """Mode-11 G1 with the backpack and the 4-sphere sole of mj-nlp's ``g1_29dof_feet.xml``."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(SPHERE_FEET_COLLISION,),
    spec_fn=get_backpack_sphere_feet_spec,
    articulation=G1_MODE11_ARTICULATION,
  )


# The backpack adds no actuator, so the per-joint action scale is mode 11's unchanged.
G1_BACKPACK_ACTION_SCALE: dict[str, float] = dict(G1_MODE11_ACTION_SCALE)


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_backpack_robot_cfg())

  viewer.launch(robot.spec.compile())
