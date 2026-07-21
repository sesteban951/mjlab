"""Terrain-level curriculum for crawling.

Mirrors ``velocity.mdp.terrain_levels_vel`` (promote a robot to harder terrain once it has crawled
past half a sub-terrain, demote if it fell far short of its commanded distance), but the crawl
command is a gait-library motion, not a velocity vector -- so the expected distance is derived from
the differential-drive command's forward-speed magnitude (``twist_command[:, 0]``). Turn-in-place
and idle envs (vx approx 0) are never demoted for not translating; they simply hold their level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.crawling_fwd.mdp.commands import LibraryMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


def terrain_levels_crawl(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  """Promote/demote terrain levels by crawl distance vs. commanded forward speed."""
  asset: Entity = env.scene[asset_cfg.name]
  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None
  cmd = cast(LibraryMotionCommand, env.command_manager.get_term(command_name))

  # Distance crawled from the spawn origin (xy).
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Promote if the robot crawled past half a sub-terrain.
  move_up = distance > terrain_generator.size[0] / 2

  # Demote a STRAIGHT-driving env (nonzero commanded vx) that covered less than half the distance
  # its commanded speed should have produced this episode. Turn/idle envs (vx approx 0) are exempt.
  speed = cmd.twist_command[env_ids, 0].abs()
  move_down = (
    (speed > 1e-3) & (distance < speed * env.max_episode_length_s * 0.5) & ~move_up
  )

  terrain.update_env_origins(env_ids, move_up, move_down)

  # Logging: mean/max level overall + per sub-terrain type (one column per type in curriculum mode).
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  if terrain_origins.shape[1] == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result
