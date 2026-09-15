from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import MotionCommand
from .tvlqr import TvlqrGuidedJointPositionAction

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


##
# CLF-RL: rewards built on the TVLQR that was designed around this trajectory.
# See mdp/tvlqr.py -- the controller guides by reward and is never applied.
##


def _tvlqr(env: ManagerBasedRlEnv, action_name: str) -> TvlqrGuidedJointPositionAction:
  term = env.action_manager.get_term(action_name)
  if not isinstance(term, TvlqrGuidedJointPositionAction):
    raise TypeError(
      f"action term {action_name!r} is a {type(term).__name__}; the CLF rewards need a "
      f"TvlqrGuidedJointPositionAction, which is what carries K, P and alpha"
    )
  return term


def clf_decrease_rbf(
  env: ManagerBasedRlEnv,
  action_name: str,
  sigma: float,
  squared: bool = False,
  normalize: bool = True,
) -> torch.Tensor:
  """Radial basis on the CLF decrease violation: ``exp(-viol / sigma^2)``, in [0, 1].

  ``viol = max(V_{k+1} - V_k + alpha_k V_k, 0)`` is robot_rl's ``clf_decreasing_condition`` in the
  discrete form the backward pass actually prescribes -- see ``tvlqr.violation``. Maximal when the
  policy holds the decay rate the LQR guarantees, and one-sided, so exceeding it earns nothing.

  ``alpha_k`` is the EXPORTED per-step rate, not a hand-set constant. That matters: robot_rl's
  alpha is continuous (0.5-1.0 1/s) and the discrete equivalent at a 5 ms step is
  ``1 - exp(-alpha*dt) ~ 0.0025``, so porting the number directly asks for ~150x the decay. The
  measured schedule here sits in [2e-4, 6.8e-3] per step, i.e. [0.04, 1.4] 1/s -- which brackets
  robot_rl's hand-picked value, and is the reason to read it off the design instead of guessing.

  ``normalize=True`` (the default) uses the violation as a FRACTION of V -- see
  ``tvlqr.violation_rel``. Keep it on: the raw violation spans four orders of magnitude on a
  zero-action rollout (p5 0.3, p99 833) because V itself spans three, so no fixed ``sigma`` covers
  it and the reward is either dead or saturated everywhere. The ratio sits at p50 0.08, p95 0.24.
  ``normalize=False`` gives the unscaled quantity for diagnostics.

  ``squared=True`` gives the textbook ``exp(-x^2 / sigma^2)`` instead of the ``exp(-x / sigma^2)``
  form used here.
  """
  term = _tvlqr(env, action_name)
  viol = term.violation_rel if normalize else term.violation
  return torch.exp(-(viol**2 if squared else viol) / sigma**2) * term.in_clip


def clf_value_kernel(
  env: ManagerBasedRlEnv, action_name: str, beta: float
) -> torch.Tensor:
  """``(1 + V / V_ref,k)^-beta`` in (0, 1]: bounded-elasticity tracking on the CLF value."""
  term = _tvlqr(env, action_name)
  return (1.0 + term.v / term.v_ref) ** -beta * term.in_clip


def qdes_imitation_rbf(
  env: ManagerBasedRlEnv, action_name: str, sigma: float, squared: bool = False
) -> torch.Tensor:
  """Radial basis on how far the policy's command sits from the controller's: in [0, 1].

  ``exp(-||qdes_policy - qdes_ctrl|| / sigma^2)`` with ``qdes_ctrl = u_bar_k + K_k dx`` and
  ``qdes_policy = action * scale + offset``.

  BOTH SIDES ARE IN RADIANS. ``qdes_policy`` is ``_processed_actions``, i.e. the policy output
  AFTER scale and offset; differencing the raw action against ``qdes_ctrl`` would subtract a
  dimensionless number from a joint angle. Both are also in this action term's joint order, which
  is not the export's actuator order -- the term permutes K's rows on load.
  """
  term = _tvlqr(env, action_name)
  err = torch.linalg.norm(term.qdes_policy - term.qdes_ctrl, dim=-1)
  return torch.exp(-(err**2 if squared else err) / sigma**2) * term.in_clip
