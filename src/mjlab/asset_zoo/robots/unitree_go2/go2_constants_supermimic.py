"""Unitree Go2 configured like TrackerLab SuperMimic's Go2 tracking task.

SuperMimic (https://github.com/Renforce-Dynamics/trackerlab-supermimic) drives the
Go2 with Isaac Lab v2.3.2's Go2 DC motor model: kp 25, kd 0.5, 23.5 Nm continuous
and peak torque on every joint (calves included), and a 30 rad/s no-load speed.
It inits and offsets actions from a hips-at-zero stance with a uniform 0.25
action scale.

The MJCF and meshes are shared with ``go2_constants``. Armature keeps the vendored
Go2 rotor values, since SuperMimic's USD does not state one.

Unlike SuperMimic and the vendored ``FULL_COLLISION`` (``conaffinity=0``, so robot
geoms never touch each other), self-collisions are enabled so that legs cannot
pass through each other or the base. MuJoCo still skips parent-child pairs, and
the default stance and the Kine2Go trot have no self-contacts.
"""

import dataclasses

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.asset_zoo.robots.unitree_go2.go2_constants import FULL_COLLISION, get_spec
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# Collision config.
##

# FULL_COLLISION's foot settings and floor contacts, plus robot-robot contacts.
GO2_SELF_COLLISION = dataclasses.replace(FULL_COLLISION, conaffinity=1)

##
# Actuator config.
##

STIFFNESS = 25.0
DAMPING = 0.5
EFFORT_LIMIT = 23.5
VELOCITY_LIMIT = 30.0

GO2_SUPERMIMIC_ACTUATOR_HIP_THIGH = DcMotorActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_LIMIT,
  saturation_effort=EFFORT_LIMIT,
  velocity_limit=VELOCITY_LIMIT,
  armature=0.01,
)
GO2_SUPERMIMIC_ACTUATOR_CALF = DcMotorActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_LIMIT,
  saturation_effort=EFFORT_LIMIT,
  velocity_limit=VELOCITY_LIMIT,
  armature=0.02,
)

GO2_SUPERMIMIC_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_SUPERMIMIC_ACTUATOR_HIP_THIGH,
    GO2_SUPERMIMIC_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
)

GO2_SUPERMIMIC_ACTION_SCALE = 0.25

##
# Keyframe.
##

GO2_SUPERMIMIC_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.30),
  joint_pos={
    ".*_hip_joint": 0.0,
    ".*_thigh_joint": 0.9,
    ".*_calf_joint": -1.8,
  },
  joint_vel={".*": 0.0},
)


def get_go2_supermimic_robot_cfg() -> EntityCfg:
  """Get a fresh Go2 configuration with SuperMimic's actuators and stance."""
  return EntityCfg(
    init_state=GO2_SUPERMIMIC_INIT_STATE,
    collisions=(GO2_SELF_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_SUPERMIMIC_ARTICULATION,
  )
