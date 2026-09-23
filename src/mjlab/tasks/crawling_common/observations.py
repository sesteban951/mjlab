"""Shared observation surgery for the twist-conditioned gait-library envs.

Every library controller (crawl, walk, jog) turns the inherited tracking observations into a
REFERENCE-FREE actor the same way, so the recipe lives here once rather than in each env cfg.
See :func:`make_actor_reference_free`.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.observations import projected_gravity
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.crawling_fwd.mdp.observations import commanded_twist, motion_phase
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# Reference terms the actor must not see. ``command`` is the full reference command vector and
# ``motion_anchor_ori_b`` the reference anchor's orientation -- both require a motion clip, which
# does not exist at deployment (there is a joystick twist instead).
_ACTOR_REFERENCE_TERMS = ("command", "motion_anchor_ori_b")

# Tilt-sensing noise on the projected-gravity observation (rad-ish on a unit vector). Matches the
# value mjlab's velocity task and unitree_rl_mjlab both use for this term.
GRAVITY_NOISE = 0.05


def make_actor_reference_free(
  cfg: ManagerBasedRlEnvCfg,
  command_name: str = "motion",
  gravity_noise: float = GRAVITY_NOISE,
) -> None:
  """Turn an inherited tracking observation set into a reference-free actor, in place.

  Three steps, applied to an already-built tracking env cfg whose ``motion`` command is a
  gait-library command:

  1. Add the commanded twist and a phase clock to BOTH groups, un-noised -- they are the task
     input, known exactly at deployment.
  2. Drop the reference terms (:data:`_ACTOR_REFERENCE_TERMS`) from the ACTOR only. The critic
     keeps the full reference, which is what makes this asymmetric; the library still drives
     learning through the imitation rewards.
  3. Restore a reference-free orientation signal: projected gravity is the robot's own IMU tilt,
     replacing the roll/pitch lost with ``motion_anchor_ori_b``. It carries no yaw (gravity is
     symmetric about the vertical), so heading is left to the twist reward.

  Note ``has_state_estimation=False`` on the tracking base has already dropped
  ``motion_anchor_pos_b`` and ``base_lin_vel``, so after this the actor sees only proprioception,
  the phase clock and the commanded twist.

  Args:
    cfg: An already-built tracking env cfg carrying a gait-library ``motion`` command.
    command_name: Name of that command term.
    gravity_noise: Symmetric uniform noise on the projected-gravity observation.
  """
  twist_obs = ObservationTermCfg(
    func=commanded_twist, params={"command_name": command_name}
  )
  phase_obs = ObservationTermCfg(
    func=motion_phase, params={"command_name": command_name}
  )
  for group in ("actor", "critic"):
    cfg.observations[group].terms["commanded_twist"] = replace(twist_obs)
    cfg.observations[group].terms["motion_phase"] = replace(phase_obs)

  for ref_term in _ACTOR_REFERENCE_TERMS:
    cfg.observations["actor"].terms.pop(ref_term, None)

  cfg.observations["actor"].terms["projected_gravity"] = ObservationTermCfg(
    func=projected_gravity, noise=Unoise(n_min=-gravity_noise, n_max=gravity_noise)
  )
