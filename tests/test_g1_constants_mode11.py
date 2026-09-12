"""Tests for g1_constants_mode11.py: only the hip pitch actuator differs from the base G1."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.unitree_g1 import g1_constants, g1_constants_mode11
from mjlab.entity import Entity

HIP_PITCH = re.compile(r".*_hip_pitch_joint$")


@pytest.fixture(scope="module")
def base_model() -> mujoco.MjModel:
  return Entity(g1_constants.get_g1_robot_cfg()).spec.compile()


@pytest.fixture(scope="module")
def mode11_model() -> mujoco.MjModel:
  return Entity(g1_constants_mode11.get_g1_mode11_robot_cfg()).spec.compile()


def test_hip_pitch_is_a_7520_22(mode11_model) -> None:
  cfg = g1_constants.G1_ACTUATOR_7520_22
  seen = 0
  for i in range(mode11_model.nu):
    act = mode11_model.actuator(i)
    if not HIP_PITCH.match(act.name):
      continue
    seen += 1
    assert act.gainprm[0] == pytest.approx(g1_constants.STIFFNESS_7520_22)
    assert act.biasprm[1] == pytest.approx(-g1_constants.STIFFNESS_7520_22)
    assert act.biasprm[2] == pytest.approx(-g1_constants.DAMPING_7520_22)
    assert act.forcerange[1] == cfg.effort_limit == 139.0
    joint = mode11_model.joint(act.trnid[0])
    assert joint.armature[0] == pytest.approx(g1_constants.ARMATURE_7520_22)
  assert seen == 2


def test_everything_else_matches_base(base_model, mode11_model) -> None:
  """Every non-hip-pitch actuator and joint, and all body masses, equal the base model."""
  assert base_model.nu == mode11_model.nu == 29
  assert base_model.nbody == mode11_model.nbody
  np.testing.assert_array_equal(base_model.body_mass, mode11_model.body_mass)
  np.testing.assert_array_equal(base_model.jnt_range, mode11_model.jnt_range)
  for i in range(base_model.nu):
    a, b = base_model.actuator(i), mode11_model.actuator(i)
    assert a.name == b.name
    if HIP_PITCH.match(a.name):
      continue
    np.testing.assert_array_equal(a.gainprm, b.gainprm)
    np.testing.assert_array_equal(a.biasprm, b.biasprm)
    np.testing.assert_array_equal(a.forcerange, b.forcerange)
    np.testing.assert_array_equal(
      base_model.joint(a.trnid[0]).armature, mode11_model.joint(b.trnid[0]).armature
    )


def test_action_scale() -> None:
  base = g1_constants.G1_ACTION_SCALE
  m11 = g1_constants_mode11.G1_MODE11_ACTION_SCALE
  assert set(base) == set(m11)
  assert m11[".*_hip_pitch_joint"] == pytest.approx(
    0.25 * 139.0 / g1_constants.STIFFNESS_7520_22
  )
  assert m11[".*_hip_pitch_joint"] == pytest.approx(m11[".*_hip_roll_joint"])
  for k in base:
    if k != ".*_hip_pitch_joint":
      assert m11[k] == pytest.approx(base[k])


def test_sphere_feet_variant_keeps_the_sole() -> None:
  model = Entity(
    g1_constants_mode11.get_g1_mode11_sphere_feet_robot_cfg()
  ).spec.compile()
  feet = [
    model.geom(i)
    for i in range(model.ngeom)
    if re.match(r"^(left|right)_foot[1-4]_collision$", model.geom(i).name)
  ]
  assert len(feet) == 8
  assert all(g.friction[0] == 1.0 for g in feet)
  assert model.actuator("left_hip_pitch_joint").forcerange[1] == 139.0
