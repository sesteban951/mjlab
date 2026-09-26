"""Unitree G1 unicycle walking with a short observation history on the actor.

``G1-Walk-Unicycle`` with the actor's proprioceptive terms stacked over the last
``HISTORY_LENGTH`` control steps (100 ms at 50 Hz). The actor is reference-free and sees neither
base linear velocity nor position, yet the twist reward and the egocentric root target ask it to
hold a speed. From one frame it can only read speed off the stance leg's joint velocities, and
only if it knows which foot is planted. A short window exposes the contact state and averages the
joint-velocity noise (implicit leg odometry), spans the 0-40 ms actuator delay of the custom DR
so the delay and the mass/friction draws can be identified per episode, and shows touchdown
impacts as joint-velocity jumps.

Only the terms that carry dynamics are stacked: joint positions, joint velocities, base angular
velocity, projected gravity and the last actions. The commanded twist (piecewise constant) and
the motion phase (a known ramp) stay at one frame, and the critic is unchanged (it already sees
privileged state). The stacked layout is term-major and flattened; the history buffer restarts,
backfilled with the first frame, at every episode reset.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.walk_unicycle.config.g1.env_cfgs import (
  unitree_g1_walk_unicycle_env_cfg,
)

HISTORY_LENGTH = 5  # control steps at 50 Hz -> 100 ms, including the current frame
HISTORY_TERMS = (
  "joint_pos",
  "joint_vel",
  "base_ang_vel",
  "projected_gravity",
  "actions",
)


def unitree_g1_walk_unicycle_history_env_cfg(
  play: bool = False, history_length: int = HISTORY_LENGTH
) -> ManagerBasedRlEnvCfg:
  """``G1-Walk-Unicycle`` with ``history_length`` frames on the actor's dynamics terms."""
  cfg = unitree_g1_walk_unicycle_env_cfg(play=play)
  actor_terms = cfg.observations["actor"].terms
  for name in HISTORY_TERMS:
    actor_terms[name].history_length = history_length
  return cfg
