from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


#########################################################################
# UNITREE RL GYM TERMINATIONS
#########################################################################

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def bad_orientation_rpy(
  env: ManagerBasedRlEnv,
  roll_limit: float,
  pitch_limit: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when roll or pitch exceeds separate limits."""
  asset = env.scene[asset_cfg.name]
  quat = asset.data.root_link_quat_w  # [B, 4], (w, x, y, z)
  roll, pitch, _ = euler_xyz_from_quat(quat)
  return (torch.abs(roll) > roll_limit) | (torch.abs(pitch) > pitch_limit)
