"""Unitree G1 constants for the mode_machine 11 hardware variant.

mjlab's ``g1.xml`` is Unitree's ``g1_29dof_rev_1_0``, which is mode_machine 5. The robot in
this lab reports mode_machine 11 (Unitree app: Device -> Data -> Robot -> Machine Type). Per
unitree_ros/robots/g1_description, mode 11 differs from rev_1_0 in exactly one place: the hip
PITCH gearbox is 22.5:1 instead of 14.3:1, so the hip pitch motor is the same 7520_22 unit that
already drives hip roll and knee -- 139 N.m, 20 rad/s and 0.0251 kg.m2 of reflected inertia
instead of 88 N.m, 32 rad/s and 0.0102 kg.m2. Kinematics, masses, inertias, joint ranges, the
4010 wrists and the unlocked waist are identical, so the XML is reused unchanged and only the
actuator table moves ``.*_hip_pitch_joint`` from the 7520_14 group to the 7520_22 group. The
derived PD gains follow (hip pitch Kp 40.18 -> 99.10, Kd 2.56 -> 6.31) and so does the action
scale (0.548 -> 0.351).

Use :func:`get_g1_mode11_robot_cfg` / :func:`get_g1_mode11_sphere_feet_robot_cfg` in place of
the ``g1_constants`` getters and :data:`G1_MODE11_ACTION_SCALE` in place of ``G1_ACTION_SCALE``.
Every other constant is the base one; import it from ``g1_constants``.
"""

from dataclasses import replace

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  FULL_COLLISION,
  G1_ACTION_SCALE,
  G1_ACTUATOR_7520_14,
  G1_ACTUATOR_7520_22,
  G1_ARTICULATION,
  KNEES_BENT_KEYFRAME,
  SPHERE_FEET_COLLISION,
  get_spec,
  get_sphere_feet_spec,
)
from mjlab.entity import EntityCfg

MODE_MACHINE = 11

##
# Actuator config: the one change against g1_constants.
##

# The hip pitch joints take the 7520_22's gains, limit and armature.
G1_MODE11_ACTUATOR_HIP_PITCH = replace(
  G1_ACTUATOR_7520_22, target_names_expr=(".*_hip_pitch_joint",)
)


def _single(cfg: BuiltinPositionActuatorCfg, joint: str) -> BuiltinPositionActuatorCfg:
  """``cfg`` restricted to exactly one joint."""
  return replace(cfg, target_names_expr=(joint,))


# ORDER MATTERS. mjlab creates actuators group by group, joints in definition order within a
# group, and that sequence is the compiled model's ``ctrl`` order. mjlab's own action term and
# ONNX export re-map by joint name, but mj-nlp consumes the compiled XML directly and its saved
# ``input`` arrays are in ``ctrl`` order -- so the mode-11 model must keep the base order:
#   ..., L hip pitch, L hip yaw, R hip pitch, R hip yaw, waist yaw, L hip roll, L knee, ...
# Splitting the old 7520_14 group into single-joint groups in that sequence reproduces it exactly
# while giving the two hip pitch entries the 7520_22 parameters.
_MODE11_7520_GROUPS: tuple[BuiltinPositionActuatorCfg, ...] = (
  _single(G1_MODE11_ACTUATOR_HIP_PITCH, "left_hip_pitch_joint"),
  _single(G1_ACTUATOR_7520_14, "left_hip_yaw_joint"),
  _single(G1_MODE11_ACTUATOR_HIP_PITCH, "right_hip_pitch_joint"),
  _single(G1_ACTUATOR_7520_14, "right_hip_yaw_joint"),
  _single(G1_ACTUATOR_7520_14, "waist_yaw_joint"),
)

# Same articulation as the base with the 7520_14 entry replaced by the groups above; the
# 5020, 7520_22 (hip roll + knee), 4010, waist and ankle actuators are untouched.
_actuators: list[BuiltinPositionActuatorCfg] = []
for _a in G1_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _actuators.extend(_MODE11_7520_GROUPS if _a is G1_ACTUATOR_7520_14 else (_a,))
G1_MODE11_ARTICULATION = replace(G1_ARTICULATION, actuators=tuple(_actuators))

##
# Final config.
##


def get_g1_mode11_robot_cfg() -> EntityCfg:
  """Mode-11 G1 with mjlab's stock 7-capsule sole (see :func:`g1_constants.get_g1_robot_cfg`)."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_MODE11_ARTICULATION,
  )


def get_g1_mode11_sphere_feet_robot_cfg() -> EntityCfg:
  """Mode-11 G1 with the 4-sphere sole of mj-nlp's ``g1_29dof_feet.xml``."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(SPHERE_FEET_COLLISION,),
    spec_fn=get_sphere_feet_spec,
    articulation=G1_MODE11_ARTICULATION,
  )


# Per-joint action scale, 0.25 * effort / stiffness, keyed like G1_ACTION_SCALE. Only hip pitch
# changes, 0.548 -> 0.351 rad, because both its limit and its stiffness are now the 7520_22's.
G1_MODE11_ACTION_SCALE: dict[str, float] = dict(G1_ACTION_SCALE)
assert G1_MODE11_ACTUATOR_HIP_PITCH.effort_limit is not None
G1_MODE11_ACTION_SCALE[".*_hip_pitch_joint"] = (
  0.25
  * G1_MODE11_ACTUATOR_HIP_PITCH.effort_limit
  / G1_MODE11_ACTUATOR_HIP_PITCH.stiffness
)


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_mode11_robot_cfg())

  viewer.launch(robot.spec.compile())
