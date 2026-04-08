from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Per-foot vertical clearance above terrain.

  Returns:
    Tensor of shape [B, F] where F is the number of frames (feet).
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor), (
    f"foot_height requires a TerrainHeightSensor, got {type(sensor).__name__}"
  )
  return sensor.data.heights


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


#########################################################################
# UNITREE RL GYM OBSERVATIONS
#########################################################################


def phase(
  env: ManagerBasedRlEnv,
  period: float = 0.8,
) -> torch.Tensor:
  """Single-leg sinusoidal phase clock matching unitree_rl_gym.

  Returns:
    Tensor of shape [B, 2]: [sin(phase), cos(phase)].
  """
  phase = (env.episode_length_buf * env.step_dt) % period / period
  return torch.stack(
    [torch.sin(2.0 * math.pi * phase), torch.cos(2.0 * math.pi * phase)],
    dim=-1,
  )


#########################################################################
# OTHER CUSTOM OBSERVATIONS
#########################################################################


def gait_phase(
  env: ManagerBasedRlEnv,
  period: float = 1.0,
  offset: float = 0.5,
) -> torch.Tensor:
  """Sinusoidal gait phase clock for left and right legs.

  Returns:
    Tensor of shape [B, 4]: [sin(left), cos(left), sin(right), cos(right)].
  """
  phase = (env.episode_length_buf * env.step_dt) % period / period
  phase_right = (phase + offset) % 1.0
  two_pi = 2.0 * math.pi
  return torch.stack(
    [
      torch.sin(two_pi * phase),
      torch.cos(two_pi * phase),
      torch.sin(two_pi * phase_right),
      torch.cos(two_pi * phase_right),
    ],
    dim=-1,
  )
