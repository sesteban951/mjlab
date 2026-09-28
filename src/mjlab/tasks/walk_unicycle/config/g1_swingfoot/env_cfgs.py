"""Unitree G1 unicycle walking with swing-foot geometry penalties: anti toe-skid.

``G1-Walk-Unicycle`` plus two hinge costs on the SWING foot (see ``walk_unicycle.mdp.rewards``):

  toe_clearance     the toe tip must stay ``TOE_MIN_HEIGHT`` above the plane while the
                    reference has that foot in the air
  swing_foot_pitch  the foot must not point its toe down more than ``MAX_TOE_DOWN``

The base task already penalizes stance-foot slip and landing impact, but both only act once a
foot is planted. The hardware failure is BEFORE that: the reference swings the foot toe-down
with the toe 1-2 cm off the floor, the real floor catches it, the swing ends early behind the
hip and the robot pitches forward (above ~0.6 m/s it runs away). Neither term is active while
standing (twist gate) and both are exactly zero once the swing foot is clear and flat, so the
reference is tracked unchanged wherever it is already safe.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.walk_unicycle import mdp
from mjlab.tasks.walk_unicycle.config.g1.env_cfgs import (
  unitree_g1_walk_unicycle_env_cfg,
)

TOE_MIN_HEIGHT = (
  0.03  # [m] toe-tip margin over the plane during swing; the floor ate ~2 cm
)
MAX_TOE_DOWN = (
  0.35  # [rad] ~20 deg of toe-down foot pitch allowed in swing (clips: ~64 deg)
)
SWING_HEIGHT = (
  0.05  # [m] reference ankle height above which the foot counts as swinging
)
# Both costs are O(1) per step at the worst case (toe on the floor / foot vertical), the same
# order as the tracking terms, so they matter without dominating.
TOE_CLEARANCE_WEIGHT = -1.0
SWING_FOOT_PITCH_WEIGHT = -1.0


def unitree_g1_walk_unicycle_swingfoot_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """``G1-Walk-Unicycle`` with toe-clearance and swing-foot-pitch penalties."""
  cfg = unitree_g1_walk_unicycle_env_cfg(play=play)
  cfg.rewards["toe_clearance"] = RewardTermCfg(
    func=mdp.toe_clearance,
    weight=TOE_CLEARANCE_WEIGHT,
    params={
      "command_name": "motion",
      "min_height": TOE_MIN_HEIGHT,
      "swing_height": SWING_HEIGHT,
      "command_threshold": 0.05,
    },
  )
  cfg.rewards["swing_foot_pitch"] = RewardTermCfg(
    func=mdp.swing_foot_pitch,
    weight=SWING_FOOT_PITCH_WEIGHT,
    params={
      "command_name": "motion",
      "max_toe_down": MAX_TOE_DOWN,
      "swing_height": SWING_HEIGHT,
      "command_threshold": 0.05,
    },
  )
  return cfg
