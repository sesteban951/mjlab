"""Config surgery shared by the twist-conditioned gait-library envs.

Every library task starts from a built tracking env cfg and applies the same steps, so
each lives here once: carry the single-clip command over to a library command
(:func:`library_command_fields`), make the actor reference-free
(:func:`make_actor_reference_free`) and track the unrolled root
(:func:`use_egocentric_root_rewards`).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.observations import projected_gravity
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.gait_library.mdp.observations import commanded_twist, motion_phase
from mjlab.tasks.gait_library.mdp.rewards import (
  egocentric_anchor_orientation_error_exp,
  egocentric_anchor_position_error_exp,
)
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg
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


def library_command_fields(old: MotionCommandCfg, motion_dir: str) -> dict[str, Any]:
  """Keyword arguments carrying a built ``MotionCommandCfg`` over to a library command.

  ``motion_file`` is unused by the library loader, but train and play require an
  existing path on every tracking task, so it points at the motion dir.
  """
  return {
    "entity_name": old.entity_name,
    "debug_vis": old.debug_vis,
    "pose_range": old.pose_range,
    "velocity_range": old.velocity_range,
    "joint_position_range": old.joint_position_range,
    "adaptive_kernel_size": old.adaptive_kernel_size,
    "adaptive_lambda": old.adaptive_lambda,
    "adaptive_uniform_ratio": old.adaptive_uniform_ratio,
    "adaptive_alpha": old.adaptive_alpha,
    "sampling_mode": old.sampling_mode,
    "anchor_body_name": old.anchor_body_name,
    "body_names": old.body_names,
    "viz": old.viz,
    "motion_dir": motion_dir,
    "motion_file": motion_dir,
  }


def use_egocentric_root_rewards(
  cfg: ManagerBasedRlEnvCfg, command_name: str = "motion"
) -> None:
  """Swap the absolute root position and orientation rewards for egocentric ones.

  A looping clip's absolute root pose is a sawtooth, so tracking it punishes net
  progress and net turning; the egocentric targets follow the unrolled reference.
  The base terms' weights and stds are kept.
  """
  cfg.rewards["motion_global_root_pos"] = RewardTermCfg(
    func=egocentric_anchor_position_error_exp,
    weight=0.5,
    params={"command_name": command_name, "std": 0.3},
  )
  cfg.rewards["motion_global_root_ori"] = RewardTermCfg(
    func=egocentric_anchor_orientation_error_exp,
    weight=0.5,
    params={"command_name": command_name, "std": 0.4},
  )
