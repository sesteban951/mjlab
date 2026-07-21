"""Unitree G1 differential-drive crawling on ROUGH terrain.

Reuses ``G1-Crawling-DiffDrive``'s env config verbatim, then (Phase 1) swaps the flat plane for a
crawl-tuned generated terrain (mostly flat + small grass/brick roughness + gentle ~10 deg slopes),
adds the out-of-bounds termination, and bumps sim contact headroom. Phases 2-3 (terrain-relative
reference command, orientation relax, friction DR, curriculum) layer on top of this builder.
"""

from dataclasses import fields, replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.crawling_diffdrive.config.g1.env_cfgs import (
  unitree_g1_crawling_diffdrive_env_cfg,
)
from mjlab.tasks.crawling_diffdrive_rough.mdp.commands import (
  RoughDiffDriveMotionCommandCfg,
)
from mjlab.tasks.crawling_diffdrive_rough.mdp.curriculums import terrain_levels_crawl
from mjlab.tasks.velocity.mdp.terminations import out_of_terrain_bounds
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import (
  flat,
  hf_pyramid_slope,
  hf_pyramid_slope_inv,
  random_rough,
  wave_terrain,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

# Crawl-tuned rough terrain: majority flat, low-amplitude high-frequency roughness (grass/brick,
# 1-5 cm), and GENTLE inclines (slope <= 0.17 rad ~= 10 deg -- a belly-crawler, not a stair-climber).
# Curriculum layout: rows = difficulty (interpolated 0->1), one column per sub-terrain type.
CRAWL_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  curriculum=True,
  sub_terrains={
    "flat": flat(proportion=0.35),
    "random_rough": random_rough(
      proportion=0.30, noise_range=(0.01, 0.05), noise_step=0.01
    ),
    "wave": wave_terrain(proportion=0.15, amplitude_range=(0.0, 0.05), num_waves=4),
    "slope": hf_pyramid_slope(proportion=0.10, slope_range=(0.0, 0.17)),
    "slope_inv": hf_pyramid_slope_inv(proportion=0.10, slope_range=(0.0, 0.17)),
  },
  add_lights=True,
)


def unitree_g1_crawling_diffdrive_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """G1 differential-drive crawling on rough terrain (terrain foundation; Phase 1)."""
  cfg = unitree_g1_crawling_diffdrive_env_cfg(play=play)

  # Swap the flat plane (inherited from the contact-rich tracking base) for the crawl rough grid.
  # replace() gives each build its own generator copy so the play-mode mutations below never leak
  # into the shared module-level constant (mirrors the velocity task).
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=replace(CRAWL_ROUGH_TERRAINS_CFG),
    max_init_terrain_level=2,  # start near-flat; the curriculum (Phase 3) promotes upward
  )

  # Truncate episodes when a robot leaves the terrain grid (no-ops on a plane, safe to add here).
  cfg.terminations["out_of_terrain_bounds"] = TerminationTermCfg(
    func=out_of_terrain_bounds, time_out=True
  )

  # Contact-rich crawling on rough ground raises the active-contact/constraint counts; give headroom
  # (base nconmax=35, ccd=50 are tuned for a flat plane). njmax=512 is inherited from crawling_fwd.
  cfg.sim.nconmax = 70
  cfg.sim.mujoco.ccd_iterations = 200

  # --- Phase 2: terrain-relative reference height + orientation relax -----------------------------
  # Privileged sim-only ground-height query under the torso, read ONLY by the command (never by any
  # actor/critic observation -> actor stays blind). anchor_body_name == "torso_link".
  torso_height_scan = TerrainHeightSensorCfg(
    name="torso_height_scan",
    frame=(ObjRef(type="body", name="torso_link", entity="robot"),),
    pattern=RingPatternCfg.single_ring(radius=0.15, num_samples=4, include_center=True),
    ray_alignment="yaw",
    max_distance=2.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # terrain only
    reduction="mean",
    debug_vis=True,
  )
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (torso_height_scan,)

  # Re-wrap the diffdrive command as the terrain-relative variant, carrying over every field.
  old = cfg.commands["motion"]
  kwargs = {f.name: getattr(old, f.name) for f in fields(old)}
  cfg.commands["motion"] = RoughDiffDriveMotionCommandCfg(
    **kwargs, terrain_height_sensor_name="torso_height_scan"
  )

  # Relax ABSOLUTE torso orientation so the body can pitch with inclines (height is handled by the
  # terrain-relative command; we don't rotate the reference to the terrain normal). Downweight +
  # loosen the two terms that track flat-ground roll/pitch, and widen the uprightness termination.
  cfg.rewards["motion_global_root_ori"].weight = 0.25
  cfg.rewards["motion_global_root_ori"].params["std"] = 0.6
  cfg.rewards["motion_body_ori"].weight = 0.5
  cfg.rewards["motion_body_ori"].params["std"] = 0.6
  cfg.terminations["anchor_ori"].params["threshold"] = 1.0

  # --- Phase 3: friction DR + terrain-level curriculum -------------------------------------------
  # Widen tangential-friction randomization to span grass (low/slippery) and brick (high). Overrides
  # the shared contact-rich base range (0.3, 1.6) for THIS task only.
  cfg.events["body_friction"].params["ranges"] = (0.2, 1.8)

  # Promote/demote robots across terrain difficulty rows as they succeed (crawl-aware metric).
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=terrain_levels_crawl, params={"command_name": "motion"}
  )

  # Play-mode terrain overrides (mirror the velocity task): a fixed, non-curriculum grid the robot is
  # randomly dropped onto, with the out-of-bounds truncation removed.
  if play:
    cfg.curriculum = {}  # non-curriculum grid in play; terrain_levels_crawl must not run
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain, mode="reset", params={}
    )
    assert cfg.scene.terrain.terrain_generator is not None
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.scene.terrain.terrain_generator.num_rows = 5
    cfg.scene.terrain.terrain_generator.num_cols = 5
    cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg
