"""Terrain-relative differential-drive crawl command.

Subclasses ``crawling_diffdrive``'s :class:`DiffDriveMotionCommand` (so the twist sampler, blended
transitions, and idle stop are all inherited unchanged) and makes the flat reference's HEIGHT
terrain-relative. A privileged, sim-only terrain-height sensor under the torso gives the local
ground height; per env we add ``_ground_offset = local_ground_z - env_origin_z`` to the reference
anchor/body z. Because every downstream height consumer reads through these two accessors, the one
offset propagates automatically to the egocentric z reward target, the ``motion_body_pos`` z term,
the z-based terminations, and the RSI spawn height -- no other code needs to change.

The actor never sees this sensor (it is read only here), so the policy stays blind and deployable;
only the training reference is terrain-aware. Orientation is NOT made terrain-relative here -- that
is handled by relaxing the absolute-orientation reward weights in the env config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.crawling_diffdrive.mdp.commands import (
  DiffDriveMotionCommand,
  DiffDriveMotionCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class RoughDiffDriveMotionCommand(DiffDriveMotionCommand):
  """Differential-drive crawl command with a terrain-relative reference height (module docstring)."""

  cfg: RoughDiffDriveMotionCommandCfg

  def __init__(self, cfg: RoughDiffDriveMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    # Per-env vertical shift added to the reference height so it sits above the LOCAL ground rather
    # than the sub-terrain's central height. Refreshed every control step from the torso sensor.
    self._ground_offset = torch.zeros(self.num_envs, device=self.device)

  # --- terrain-relative height: the single seam (both z-source accessors) --------------------------

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    p = super().anchor_pos_w.clone()
    p[:, 2] += self._ground_offset
    return p

  @property
  def body_pos_w(self) -> torch.Tensor:
    p = super().body_pos_w.clone()
    p[:, :, 2] += self._ground_offset[:, None]
    return p

  # --- offset refresh + re-pin ---------------------------------------------------------------------

  def _refresh_ground_offset(self) -> None:
    """Local ground z under the torso (privileged) minus the sub-terrain central height."""
    sensor = self._env.scene[self.cfg.terrain_height_sensor_name]
    # heights = clearance (frame_z - hit_z); frame is torso_link == the anchor body, so
    # local ground z = robot torso z - clearance.
    clearance = sensor.data.heights[:, 0]  # (num_envs,), reduced over the ring
    ground_z = self.robot_anchor_pos_w[:, 2] - clearance
    offset = ground_z - self._env.scene.env_origins[:, 2]
    self._ground_offset = torch.clamp(
      offset, -self.cfg.ground_offset_clip, self.cfg.ground_offset_clip
    )

  def _update_command(self) -> None:
    # Refresh BEFORE super(): update_relative_body_poses (in super) reads the height accessors.
    self._refresh_ground_offset()
    super()._update_command()
    # Re-pin the egocentric z target to the terrain-relative reference height. The base integrates z
    # by the FLAT reference vertical velocity (crawling_fwd _update_command), which is terrain-blind
    # and would let the z target drift off the surface between twist resamples.
    self.ref_anchor_pos_w[:, 2] = self.anchor_pos_w[:, 2]

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
    # Spawn with offset 0 so RSI uses the sub-terrain central height (env_origin z): the sensor is
    # stale mid-reset (it updates after RSI), so we cannot query the true local height yet. The next
    # _update_command refreshes it to the real value. One-step staleness, bounded by the clip.
    if env_ids is None:
      self._ground_offset[:] = 0.0
    else:
      self._ground_offset[env_ids] = 0.0
    return super().reset(env_ids)


@dataclass(kw_only=True)
class RoughDiffDriveMotionCommandCfg(DiffDriveMotionCommandCfg):
  """Config for :class:`RoughDiffDriveMotionCommand`: adds the privileged terrain-height source."""

  # Name of the (privileged, sim-only) TerrainHeightSensor under the torso; read only by the command.
  terrain_height_sensor_name: str = "torso_height_scan"
  # Clamp on the per-env height offset [m] -- guards against a stray raycast miss.
  ground_offset_clip: float = 0.6

  def build(self, env: ManagerBasedRlEnv) -> RoughDiffDriveMotionCommand:
    return RoughDiffDriveMotionCommand(self, env)
