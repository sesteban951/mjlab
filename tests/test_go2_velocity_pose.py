"""Tests for the Go2 velocity + body pose (height, roll, pitch) task."""

import math

import torch

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity_custom.config.go2_custom.env_cfgs import (
  unitree_go2_pose_env_cfg,
)
from mjlab.tasks.velocity_custom.mdp import (
  BodyPoseCommandCfg,
  effective_body_pose,
  target_projected_gravity,
)
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_from_euler_xyz


def _pose_cfg() -> BodyPoseCommandCfg:
  cmd = unitree_go2_pose_env_cfg().commands["body_pose"]
  assert isinstance(cmd, BodyPoseCommandCfg)
  return cmd


def test_attitude_only_while_standing():
  cfg = _pose_cfg()
  raw = torch.tensor([[0.29, 0.2, -0.2], [0.29, 0.2, -0.2]])
  twist = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
  out = effective_body_pose(raw, twist, cfg)
  assert out[0, 1] != 0 and out[0, 2] != 0
  assert torch.all(out[1, 1:] == 0)


def test_walking_height_floor_rises_with_speed():
  cfg = _pose_cfg()
  raw = torch.full((3, 3), 0.0)
  raw[:, 0] = cfg.height_range[0]
  twist = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])
  out = effective_body_pose(raw, twist, cfg)
  # Standing and slow walking keep the low height; full speed raises it.
  assert math.isclose(out[0, 0].item(), cfg.height_range[0], abs_tol=1e-6)
  assert math.isclose(out[1, 0].item(), cfg.walking_height_range[0], abs_tol=1e-6)
  assert math.isclose(out[2, 0].item(), cfg.fast_min_height, abs_tol=1e-6)


def test_attitude_scaled_to_leg_travel():
  cfg = _pose_cfg()
  raw = torch.tensor([[cfg.height_range[0], 0.35, 0.4]])
  out = effective_body_pose(raw, torch.zeros(1, 3), cfg)
  margin = cfg.height_range[0] - cfg.leg_height_limits[0]
  need = cfg.foot_half_length * abs(math.sin(out[0, 2].item())) + (
    cfg.foot_half_width * abs(math.sin(out[0, 1].item()))
  )
  assert need <= margin + 1e-6


def test_target_gravity_matches_rotated_base():
  pose = torch.tensor([[0.3, 0.25, -0.3]])
  quat = quat_from_euler_xyz(pose[:, 1], pose[:, 2], torch.tensor([1.2]))
  actual = quat_apply_inverse(quat, torch.tensor([[0.0, 0.0, -1.0]]))
  # Yaw must not change the target.
  assert torch.allclose(target_projected_gravity(pose), actual, atol=1e-5)


def test_task_registered_with_six_commands():
  cfg = load_env_cfg("Go2-Velocity-Pose-Custom")
  actor = list(cfg.observations["actor"].terms)
  assert actor.index("body_pose") == actor.index("command") + 1
  assert "base_height" in cfg.observations["critic"].terms
  assert "stand_still" not in cfg.rewards
  assert {"track_base_height", "track_body_attitude"} <= set(cfg.rewards)
  rl = load_rl_cfg("Go2-Velocity-Pose-Custom")
  assert rl.experiment_name == "go2_velocity_pose_custom"
