"""Body pose command (base height, roll, pitch) layered on a twist command.

The policy always sees all three values. Feasibility lives in the sampler:
roll and pitch are active only while the twist command is zero, low heights
are allowed only at low walking speeds, and the attitude is scaled down when
the commanded height leaves the legs too little travel to reach it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_from_euler_xyz

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def effective_body_pose(
  raw: torch.Tensor, twist: torch.Tensor, cfg: BodyPoseCommandCfg
) -> torch.Tensor:
  """Apply the feasibility rules to raw ``[height, roll, pitch]`` targets.

  Pure function of the raw targets and the twist command, so a deploy-side
  command interface can apply the exact rules the policy was trained under.
  """
  height, roll, pitch = raw[:, 0], raw[:, 1], raw[:, 2]
  moving = torch.norm(twist, dim=1) > cfg.standing_threshold

  # While moving: clamp to the walking band, and raise the floor with speed.
  speed = torch.norm(twist[:, :2], dim=1)
  v0, v1 = cfg.crouch_speed_range
  frac = torch.clamp((speed - v0) / (v1 - v0), 0.0, 1.0)
  walk_lo = cfg.walking_height_range[0] + frac * (
    cfg.fast_min_height - cfg.walking_height_range[0]
  )
  walk_height = torch.minimum(
    torch.maximum(height, walk_lo),
    torch.full_like(height, cfg.walking_height_range[1]),
  )
  height = torch.where(moving, walk_height, height)

  # Attitude only while standing.
  standing = (~moving).float()
  roll = roll * standing
  pitch = pitch * standing

  # Scale the attitude so every foot stays within the leg's height travel.
  lo, hi = cfg.leg_height_limits
  margin = torch.clamp(torch.minimum(height - lo, hi - height), min=0.0)
  need = cfg.foot_half_length * torch.abs(torch.sin(pitch)) + (
    cfg.foot_half_width * torch.abs(torch.sin(roll))
  )
  scale = torch.where(
    need > margin, margin / need.clamp(min=1e-6), torch.ones_like(need)
  )
  # Scale the sines, so the reach check above holds exactly after scaling.
  roll = torch.asin(scale * torch.sin(roll))
  pitch = torch.asin(scale * torch.sin(pitch))
  return torch.stack((height, roll, pitch), dim=1)


class BodyPoseCommand(CommandTerm):
  cfg: BodyPoseCommandCfg

  def __init__(self, cfg: BodyPoseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.raw = torch.zeros(self.num_envs, 3, device=self.device)
    self.raw[:, 0] = cfg.nominal_height
    self.pose_command = self.raw.clone()

    self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_attitude"] = torch.zeros(self.num_envs, device=self.device)

    self._gui_enabled: viser.GuiCheckboxHandle | None = None
    self._gui_sliders: list[viser.GuiSliderHandle] = []
    self._gui_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.pose_command

  def _update_metrics(self) -> None:
    max_steps = self.cfg.resampling_time_range[1] / self._env.step_dt
    height = self.robot.data.root_link_pos_w[:, 2]
    self.metrics["error_height"] += (
      torch.abs(height - self.pose_command[:, 0]) / max_steps
    )
    target_g = target_projected_gravity(self.pose_command)
    self.metrics["error_attitude"] += (
      torch.norm(self.robot.data.projected_gravity_b - target_g, dim=1) / max_steps
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.raw[env_ids, 0] = r.uniform_(*self.cfg.height_range)
    self.raw[env_ids, 1] = r.uniform_(*self.cfg.roll_range)
    self.raw[env_ids, 2] = r.uniform_(*self.cfg.pitch_range)
    nominal = r.uniform_(0.0, 1.0) < self.cfg.rel_nominal_envs
    nominal_ids = env_ids[nominal]
    self.raw[nominal_ids, 0] = self.cfg.nominal_height
    self.raw[nominal_ids, 1:] = 0.0

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    # Pure function of the raw targets and the current twist command.
    del env_ids
    twist = self._env.command_manager.get_command(self.cfg.twist_command_name)
    assert twist is not None
    self.pose_command = effective_body_pose(self.raw, twist, self.cfg)

  # GUI (viser viewer).

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Sliders for the raw height, roll and pitch targets."""
    axes = [
      ("height", self.cfg.height_range, self.cfg.nominal_height),
      ("roll", self.cfg.roll_range, 0.0),
      ("pitch", self.cfg.pitch_range, 0.0),
    ]
    with server.gui.add_folder(name.capitalize()):
      self._gui_enabled = server.gui.add_checkbox("Enable", initial_value=False)
      self._gui_sliders = [
        server.gui.add_slider(label, min=lo, max=hi, step=0.01, initial_value=init)
        for label, (lo, hi), init in axes
      ]
    self._gui_get_env_idx = get_env_idx

  def compute(
    self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    if self._gui_enabled is not None and self._gui_enabled.value:
      assert self._gui_get_env_idx is not None
      idx = self._gui_get_env_idx()
      for i, s in enumerate(self._gui_sliders):
        self.raw[idx, i] = s.value
    super().compute(dt, env_ids)


@dataclass(kw_only=True)
class BodyPoseCommandCfg(CommandTermCfg):
  """Base height (m, base origin above flat ground), roll and pitch (rad)."""

  entity_name: str
  twist_command_name: str = "twist"
  nominal_height: float
  height_range: tuple[float, float]
  roll_range: tuple[float, float]
  pitch_range: tuple[float, float]
  rel_nominal_envs: float = 0.25
  """Fraction of resamples that command the nominal pose."""
  standing_threshold: float = 0.1
  """Twist norm at or below which the robot counts as standing."""
  walking_height_range: tuple[float, float]
  crouch_speed_range: tuple[float, float] = (0.6, 2.0)
  """Below the first speed any walking height is allowed; by the second the
  floor has risen linearly to ``fast_min_height``."""
  fast_min_height: float
  leg_height_limits: tuple[float, float]
  """Lowest and highest base height the legs can hold with feet under hips."""
  foot_half_length: float
  """Half the front-to-rear foot spacing, for the pitch feasibility check."""
  foot_half_width: float
  """Half the left-to-right foot spacing, for the roll feasibility check."""

  def build(self, env: ManagerBasedRlEnv) -> BodyPoseCommand:
    return BodyPoseCommand(self, env)


def target_projected_gravity(pose_command: torch.Tensor) -> torch.Tensor:
  """Gravity in the base frame for the commanded roll and pitch (zero yaw)."""
  roll, pitch = pose_command[:, 1], pose_command[:, 2]
  quat = quat_from_euler_xyz(roll, pitch, torch.zeros_like(roll))
  gravity = torch.tensor((0.0, 0.0, -1.0), device=pose_command.device)
  return quat_apply_inverse(quat, gravity.expand_as(pose_command))


##
# Rewards.
##


def track_base_height(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward base height tracking on flat ground."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = asset.data.root_link_pos_w[:, 2] - command[:, 0]
  return torch.exp(-torch.square(error) / std**2)


def track_body_attitude(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward roll and pitch tracking, compared as base-frame gravity vectors."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = asset.data.projected_gravity_b - target_projected_gravity(command)
  return torch.exp(-torch.sum(torch.square(error), dim=1) / std**2)


def body_orientation_target_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize tilt away from the commanded roll and pitch (flat when walking)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = asset.data.projected_gravity_b - target_projected_gravity(command)
  return torch.sum(torch.square(error[:, :2]), dim=1)


def feet_clearance_height_scaled(
  env: ManagerBasedRlEnv,
  pose_command_name: str,
  height_ratio: float,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """``feet_clearance`` with a swing height proportional to the base height."""
  asset: Entity = env.scene[asset_cfg.name]
  pose = env.command_manager.get_command(pose_command_name)
  twist = env.command_manager.get_command(command_name)
  assert pose is not None and twist is not None
  target = (height_ratio * pose[:, 0]).unsqueeze(1)
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  vel_norm = torch.norm(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
  cost = torch.sum(torch.abs(foot_z - target) * vel_norm, dim=1)
  total = torch.norm(twist[:, :2], dim=1) + torch.abs(twist[:, 2])
  return cost * (total > command_threshold).float()


##
# Observations.
##


def base_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Base origin height in the world frame (flat ground)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]
