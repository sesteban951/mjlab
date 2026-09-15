"""Control priors for the tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.envs.mdp.actions import JointPriorAction, JointPriorActionCfg


def motion_reference_joint_pos(
  env: ManagerBasedRlEnv,
  joint_ids: torch.Tensor,
  command_name: str = "motion",
) -> torch.Tensor:
  """Prior that commands the reference motion's joint angles.

  This is privileged: the actor never observes the reference joint angles
  directly, only the anchor-relative body poses. ``joint_ids`` is injected by
  the action term and must not be set in ``prior_params``.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.joint_pos[:, joint_ids]


def tangent_state_error(
  env: ManagerBasedRlEnv,
  entity,
  qpos_ref: torch.Tensor,
  qvel_ref: torch.Tensor,
) -> torch.Tensor:
  """``x - xbar`` in the tangent basis an LQR schedule is designed in, ``(N, 2*nv)``.

  The layout matches the columns of a tape's ``gain`` (and so of any ``P`` solved
  alongside it)::

      0:3    base position      2*nv//2 + 0:3   base linear velocity
      3:6    base rotation      ...      + 3:6  base angular velocity
      6:nv   joint positions    ...      + 6:   joint velocities

  Base position is taken relative to the env origin, base rotation as the log map of
  the error quaternion expressed in the *reference's* frame, and base angular velocity
  in the body frame -- MuJoCo's own free-joint conventions, which is what the gains
  were designed against. ``qpos_ref``/``qvel_ref`` are ``(N, 7 + njoint)`` and
  ``(N, 6 + njoint)``, already gathered at each env's current frame.
  """
  data = entity.data
  quat = data.root_link_quat_w
  return torch.cat(
    [
      # Positions: world offset, base rotation logged in the reference's frame.
      data.root_link_pos_w - env.scene.env_origins - qpos_ref[:, :3],
      axis_angle_from_quat(quat_mul(quat_inv(qpos_ref[:, 3:7]), quat)),
      data.joint_pos - qpos_ref[:, 7:],
      # Velocities: free-joint qvel is linear in world, angular in the body frame.
      data.root_link_lin_vel_w - qvel_ref[:, :3],
      quat_apply_inverse(quat, data.root_link_ang_vel_w) - qvel_ref[:, 3:6],
      data.joint_vel - qvel_ref[:, 6:],
    ],
    dim=-1,
  )


class motion_tape_prior:
  """Prior that replays a solved control tape, optionally closing its LQR loop.

  The tape is the ``*_prior.npz`` written by mj-nlp's ``to_mjlab_motion.py --prior``:
  joint-position targets ``u`` on the motion's own frame grid, so frame ``k`` of the
  tape and frame ``k`` of the motion are the same instant and the tape is indexed on
  the motion command's ``time_steps`` (correct under RSI's random start frame).

  With ``feedback=True`` and a tape that carries a gain schedule, the prior is
  ``u = uff[k] + alpha[k] * K[k] @ (x - xbar[k])`` with the error taken in MuJoCo's
  own conventions (base rotation in the reference's frame, base angular velocity in
  the body frame), which is what the gains were designed against. Otherwise the
  recorded ``u`` is replayed open loop.

  ``clip_to_joint_limits`` defaults on, which is right wherever the prior is a
  behavior-cloning target: the policy should be pulled toward a reachable pose. It is
  wrong wherever the prior is the applied control, because a position setpoint outside
  the joint range is how a servo asks for its full torque -- mjlab builds these
  actuators ``ctrllimited=False`` for exactly that reason, and clamping a solved tape
  to 90% of the range deletes up to 1.5 rad of the target it asked for. ``play_prior``
  therefore passes it off; leave it alone in a training config.
  """

  def __init__(
    self,
    cfg: JointPriorActionCfg,
    env: ManagerBasedRlEnv,
    term: JointPriorAction,
  ) -> None:
    tape = np.load(cfg.prior_params["tape_file"])
    device = env.device

    mode = str(tape["actuator_mode"]) if "actuator_mode" in tape else "position"
    if mode != "position":
      raise ValueError(f"Tape actuator_mode is {mode!r}, not a joint-position target.")

    # Reorder the tape's columns into the action term's joint order.
    names = [str(n).split("/")[-1] for n in tape["joint_names"]]
    missing = [n for n in term.target_names if n not in names]
    if missing:
      raise ValueError(f"Tape is missing joints the action term drives: {missing}")
    cols = torch.tensor(
      [names.index(n) for n in term.target_names], dtype=torch.long, device=device
    )

    def _to(key: str) -> torch.Tensor:
      return torch.tensor(tape[key], dtype=torch.float32, device=device)

    self._u = _to("u")[:, cols]
    self._entity = env.scene[cfg.entity_name]
    self._num_frames = self._u.shape[0]
    # The last evaluation's pieces, unclipped, for diagnostics of the policy's input.
    nu = self._u.shape[1]
    self.feedforward_target = torch.zeros(env.num_envs, nu, device=device)
    self.feedback_correction = torch.zeros(env.num_envs, nu, device=device)
    self.closed_loop_target = torch.zeros(env.num_envs, nu, device=device)

    self._gain: torch.Tensor | None = None
    if "gain" in tape:
      njoint = self._entity.data.joint_pos.shape[1]
      qpos_ref, qvel_ref = _to("ref_qpos"), _to("ref_qvel")
      if qpos_ref.shape[1] != 7 + njoint or qvel_ref.shape[1] != 6 + njoint:
        raise ValueError(
          f"Tape reference is {qpos_ref.shape[1]}/{qvel_ref.shape[1]} wide, expected "
          f"{7 + njoint}/{6 + njoint} for a floating base with {njoint} joints."
        )
      self._feedforward = _to("feedforward")[:, cols]
      self._gain = _to("gain")[:, cols, :]
      self._alpha = _to("alpha")
      self._qpos_ref, self._qvel_ref = qpos_ref, qvel_ref

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str = "motion",
    feedback: bool = True,
    alpha_scale: float = 1.0,
    clip_to_joint_limits: bool = True,
    joint_ids: torch.Tensor | None = None,
    **kwargs,
  ) -> torch.Tensor:
    del kwargs
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    k = command.time_steps.clamp(max=self._num_frames - 1)

    if self._gain is not None:
      uff = self._feedforward[k]
      ufb = (alpha_scale * self._alpha[k]).unsqueeze(-1) * (
        torch.bmm(self._gain[k], self._state_error(env, k).unsqueeze(-1)).squeeze(-1)
      )
    else:
      uff, ufb = self._u[k], torch.zeros_like(self._u[k])
    self.feedforward_target[:] = uff
    self.feedback_correction[:] = ufb
    self.closed_loop_target[:] = uff + ufb
    target = self.closed_loop_target if feedback and self._gain is not None else uff

    if clip_to_joint_limits and joint_ids is not None:
      limits = self._entity.data.soft_joint_pos_limits[:, joint_ids]
      target = target.clamp(limits[..., 0], limits[..., 1])
    return target

  def _state_error(self, env: ManagerBasedRlEnv, k: torch.Tensor) -> torch.Tensor:
    return tangent_state_error(env, self._entity, self._qpos_ref[k], self._qvel_ref[k])
