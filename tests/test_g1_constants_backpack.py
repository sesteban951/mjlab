"""Tests for g1_constants_backpack.py: mode 11 plus one 1.4 kg box on the back of the torso."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.unitree_g1 import g1_constants_backpack, g1_constants_mode11
from mjlab.entity import Entity

BODY = g1_constants_backpack.BACKPACK_BODY_NAME
HALF = np.asarray(g1_constants_backpack.BACKPACK_SIZE) / 2


@pytest.fixture(scope="module")
def mode11_model() -> mujoco.MjModel:
  return Entity(g1_constants_mode11.get_g1_mode11_robot_cfg()).spec.compile()


@pytest.fixture(scope="module")
def backpack_model() -> mujoco.MjModel:
  return Entity(g1_constants_backpack.get_g1_backpack_robot_cfg()).spec.compile()


def test_backpack_body(backpack_model) -> None:
  """A fixed body on torso_link with the stated mass and a uniform box inertia."""
  body = backpack_model.body(BODY)
  assert backpack_model.body(body.parentid[0]).name == "torso_link"
  assert body.jntnum[0] == 0
  np.testing.assert_allclose(body.pos, g1_constants_backpack.BACKPACK_POS)
  assert body.mass[0] == pytest.approx(g1_constants_backpack.BACKPACK_MASS)
  # Inertial frame at the box centre, axes aligned with the box.
  np.testing.assert_allclose(body.ipos, 0.0, atol=1e-12)
  np.testing.assert_allclose(body.iquat, (1.0, 0.0, 0.0, 0.0))
  # Uniform-density box: I = m/12 * (sum of the squares of the other two extents).
  m = g1_constants_backpack.BACKPACK_MASS
  x, y, z = g1_constants_backpack.BACKPACK_SIZE
  expected = m / 12 * np.array([y**2 + z**2, x**2 + z**2, x**2 + y**2])
  np.testing.assert_allclose(body.inertia, expected, rtol=1e-6)


def test_backpack_geoms(backpack_model) -> None:
  """A black visual box plus a collision box that follows the package-wide policy."""
  visual = backpack_model.geom("backpack_visual")
  collision = backpack_model.geom("backpack_collision")
  for geom in (visual, collision):
    assert backpack_model.body(geom.bodyid[0]).name == BODY
    assert geom.type[0] == mujoco.mjtGeom.mjGEOM_BOX
    np.testing.assert_allclose(geom.size, HALF)
  assert backpack_model.mat(visual.matid[0]).name == "black"
  assert visual.contype[0] == visual.conaffinity[0] == 0
  # FULL_COLLISION: a non-foot ``*_collision`` geom is condim 1, priority 0.
  assert collision.contype[0] == collision.conaffinity[0] == 1
  assert collision.condim[0] == 1
  assert collision.priority[0] == 0


def test_everything_else_matches_mode11(mode11_model, backpack_model) -> None:
  """The only difference from mode 11 is the one extra body and its mass."""
  assert backpack_model.nbody == mode11_model.nbody + 1
  assert backpack_model.njnt == mode11_model.njnt
  assert backpack_model.nu == mode11_model.nu == 29
  np.testing.assert_array_equal(backpack_model.jnt_range, mode11_model.jnt_range)
  for i in range(mode11_model.nbody):
    a = mode11_model.body(i)
    b = backpack_model.body(a.name)
    assert a.mass[0] == b.mass[0]
    np.testing.assert_array_equal(a.inertia, b.inertia)
  assert backpack_model.body_subtreemass[0] == pytest.approx(
    mode11_model.body_subtreemass[0] + g1_constants_backpack.BACKPACK_MASS
  )
  for i in range(mode11_model.nu):
    a, b = mode11_model.actuator(i), backpack_model.actuator(i)
    assert a.name == b.name
    np.testing.assert_array_equal(a.gainprm, b.gainprm)
    np.testing.assert_array_equal(a.biasprm, b.biasprm)
    np.testing.assert_array_equal(a.forcerange, b.forcerange)
    np.testing.assert_array_equal(
      mode11_model.joint(a.trnid[0]).armature,
      backpack_model.joint(b.trnid[0]).armature,
    )


def test_action_scale() -> None:
  assert (
    g1_constants_backpack.G1_BACKPACK_ACTION_SCALE
    == g1_constants_mode11.G1_MODE11_ACTION_SCALE
  )


def test_sphere_feet_variant_has_backpack_and_sole() -> None:
  model = Entity(
    g1_constants_backpack.get_g1_backpack_sphere_feet_robot_cfg()
  ).spec.compile()
  assert model.body(BODY).mass[0] == pytest.approx(g1_constants_backpack.BACKPACK_MASS)
  feet = [
    model.geom(i)
    for i in range(model.ngeom)
    if re.match(r"^(left|right)_foot[1-4]_collision$", model.geom(i).name)
  ]
  assert len(feet) == 8
  assert all(g.friction[0] == 1.0 for g in feet)
  assert model.actuator("left_hip_pitch_joint").forcerange[1] == 139.0
