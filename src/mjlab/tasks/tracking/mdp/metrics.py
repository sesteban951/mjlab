from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.tasks.tracking.mdp.rewards import _tvlqr
from mjlab.utils.lab_api.math import quat_error_magnitude

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.tracking.mdp.commands import MotionCommand


def compute_mpkpe(command: MotionCommand) -> torch.Tensor:
  """Compute Mean Per-Keybody Position Error (MPKPE).

  MPKPE measures the average Euclidean distance between the reference and
  actual key body positions in the global world frame. It captures all
  tracking error, including global translation and heading drift.
  """
  pos_error = command.body_pos_w - command.robot_body_pos_w
  per_body_error = torch.norm(pos_error, dim=-1)  # (num_envs, num_bodies)
  return per_body_error.mean(dim=-1)  # (num_envs,)


def compute_root_relative_mpkpe(command: MotionCommand) -> torch.Tensor:
  """Compute Root-relative Mean Per-Keybody Position Error (R-MPKPE).

  R-MPKPE measures intrinsic pose error independent of global drift. It
  uses ``body_pos_relative_w``, the reference re-anchored to the robot's
  current root position and heading each step (the same quantity the
  tracking reward optimizes), so both global translation and yaw drift are
  removed and only the local body pose error remains.
  """
  pos_error = command.body_pos_relative_w - command.robot_body_pos_w
  per_body_error = torch.norm(pos_error, dim=-1)  # (num_envs, num_bodies)
  return per_body_error.mean(dim=-1)  # (num_envs,)


def compute_joint_velocity_error(command: MotionCommand) -> torch.Tensor:
  """Compute root-mean-square joint velocity error.

  Uses an RMS over joints (rather than a raw L2 norm) so the value is a
  per-joint quantity, comparable across robots with different DOF counts.
  """
  vel_error = command.joint_vel - command.robot_joint_vel
  return torch.sqrt(torch.mean(vel_error**2, dim=-1))  # (num_envs,)


def compute_ee_position_error(
  command: MotionCommand,
  ee_body_names: tuple[str, ...],
) -> torch.Tensor:
  """Compute end effector position error."""
  ee_indices = _get_body_indices(command, ee_body_names)
  if len(ee_indices) == 0:
    return torch.zeros(command.num_envs, device=command.device)

  ref_ee_pos = command.body_pos_relative_w[:, ee_indices]
  robot_ee_pos = command.robot_body_pos_w[:, ee_indices]

  pos_error = ref_ee_pos - robot_ee_pos
  per_ee_error = torch.norm(pos_error, dim=-1)  # (num_envs, num_ee)
  return per_ee_error.mean(dim=-1)  # (num_envs,)


def compute_ee_orientation_error(
  command: MotionCommand,
  ee_body_names: tuple[str, ...],
) -> torch.Tensor:
  """Compute end effector orientation error."""
  ee_indices = _get_body_indices(command, ee_body_names)
  if len(ee_indices) == 0:
    return torch.zeros(command.num_envs, device=command.device)

  ref_ee_quat = command.body_quat_relative_w[:, ee_indices]
  robot_ee_quat = command.robot_body_quat_w[:, ee_indices]

  per_ee_error = quat_error_magnitude(ref_ee_quat, robot_ee_quat)  # (num_envs, num_ee)
  return per_ee_error.mean(dim=-1)  # (num_envs,)


def _get_body_indices(
  command: MotionCommand,
  body_names: tuple[str, ...],
) -> list[int]:
  """Get indices of specified bodies within the command's body list.

  Args:
    command: The motion command.
    body_names: Names of bodies to find.

  Returns:
    List of indices into command.cfg.body_names, in the order requested.

  Raises:
    ValueError: If any requested body name is not tracked by the command.
      Silently dropping unknown names would otherwise report a spurious
      zero error for misconfigured end-effector lists.
  """
  name_to_index = {name: i for i, name in enumerate(command.cfg.body_names)}
  missing = [name for name in body_names if name not in name_to_index]
  if missing:
    raise ValueError(
      f"Body names {missing} are not tracked by the command. "
      f"Available bodies: {tuple(command.cfg.body_names)}."
    )
  return [name_to_index[name] for name in body_names]


# TVLQR diagnostics: the raw quantities the CLF rewards pass through their RBF kernels,
# measured rather than optimized. The sigmas are picked against these distributions, so
# logging them is how you tell a badly-sized sigma from a policy that is actually failing.


def clf_violation_rel(
  env: ManagerBasedRlEnv, action_name: str = "joint_pos"
) -> torch.Tensor:
  """The CLF decrease violation as a fraction of V -- what ``clf_decrease_rbf`` scores."""
  return _tvlqr(env, action_name).violation_rel


def clf_value(env: ManagerBasedRlEnv, action_name: str = "joint_pos") -> torch.Tensor:
  """``V = e^T P e`` at the latest substep, in the export's own units."""
  return _tvlqr(env, action_name).v


def clf_value_ratio(
  env: ManagerBasedRlEnv, action_name: str = "joint_pos"
) -> torch.Tensor:
  """``V / lambda_max(P_k)`` -- what ``clf_value_exp`` scales by sigma^2."""
  term = _tvlqr(env, action_name)
  return term.v / term.lam_max


def qdes_distance(
  env: ManagerBasedRlEnv, action_name: str = "joint_pos"
) -> torch.Tensor:
  """``||qdes_policy - qdes_ctrl||`` in radians -- what ``qdes_imitation_rbf`` scores."""
  term = _tvlqr(env, action_name)
  return torch.linalg.norm(term.qdes_policy - term.qdes_ctrl, dim=-1)
