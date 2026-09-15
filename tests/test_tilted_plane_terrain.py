"""Tests for BoxTiltedPlaneTerrainCfg, a whole patch tilted about its spawn origin."""

import mujoco
import numpy as np
import pytest

from mjlab.terrains import BoxTiltedPlaneTerrainCfg, TerrainEntity, TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

MAX_TILT_DEG = 5.0
PATCH_SIZE = (8.0, 8.0)


def make_terrain(num_rows=4, num_cols=4, difficulty_range=(0.0, 1.0), **overrides):
  cfg = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      seed=0,
      size=PATCH_SIZE,
      num_rows=num_rows,
      num_cols=num_cols,
      border_width=0.0,
      difficulty_range=difficulty_range,
      sub_terrains={
        "tilted": BoxTiltedPlaneTerrainCfg(max_tilt_deg=MAX_TILT_DEG, **overrides)
      },
    ),
    num_envs=num_rows * num_cols,
  )
  return TerrainEntity(cfg, device="cpu")


def patch_frames(terrain):
  """Return (tilt_deg, top_face_center) for every patch box, in world coordinates."""
  model = terrain.spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  tilts, tops = [], []
  for i in range(model.ngeom):
    if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX:
      continue
    rot = data.geom_xmat[i].reshape(3, 3)
    tilts.append(np.degrees(np.arccos(np.clip(rot[2, 2], -1.0, 1.0))))
    tops.append(data.geom_xpos[i] + rot @ (0.0, 0.0, model.geom_size[i][2]))
  return np.array(tilts), np.array(tops)


def test_tilt_stays_within_max_tilt_deg():
  tilts, _ = patch_frames(make_terrain())
  assert tilts.max() <= MAX_TILT_DEG + 1e-9
  assert tilts.max() > 0.0


def test_tilt_scales_with_difficulty():
  flat_tilts, _ = patch_frames(make_terrain(difficulty_range=(0.0, 0.0)))
  np.testing.assert_allclose(flat_tilts, 0.0, atol=1e-9)
  steep_tilts, _ = patch_frames(make_terrain(difficulty_range=(1.0, 1.0)))
  assert steep_tilts.min() > 0.0


def test_ground_passes_through_the_spawn_origin():
  """The tilt pivots about the origin, so a robot spawns on the surface, not in it."""
  terrain = make_terrain()
  origins = terrain.terrain_origins.reshape(-1, 3).numpy()  # pyright: ignore
  _, tops = patch_frames(terrain)
  nearest = np.linalg.norm(tops[:, None, :2] - origins[None, :, :2], axis=-1).argmin(
    axis=1
  )
  np.testing.assert_allclose(tops, origins[nearest], atol=1e-9)


@pytest.mark.parametrize("thickness", [0.5, 1.0])
def test_patch_is_solid_below_the_surface(thickness):
  terrain = make_terrain(plane_thickness=thickness)
  model = terrain.spec.compile()
  half_z = [
    model.geom_size[i][2]
    for i in range(model.ngeom)
    if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
  ]
  np.testing.assert_allclose(half_z, thickness / 2, atol=1e-9)
