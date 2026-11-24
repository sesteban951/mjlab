"""Unitree G1 crawling base: contact-rich tracking over a twist-indexed crawl library.

Shared by ``G1-Crawling-DiffDrive`` and ``G1-Crawling-Omni``: the contact-rich custom
tracking env, the action space centred on the idle crawl pose, a blended library
command with an idle-pose stop, a reference-free actor, the prone twist reward and the
egocentric root rewards. Each task then sets its own twist sampler and ranges.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.crawling_common import mdp
from mjlab.tasks.gait_library.env_cfgs import (
  library_command_fields,
  make_actor_reference_free,
  use_egocentric_root_rewards,
)
from mjlab.tasks.gait_library.library import DEFAULT_IDLE_CSV, load_idle_qpos
from mjlab.tasks.gait_library.mdp.blending import BlendingMotionCommandCfg
from mjlab.tasks.tracking.config.g1_custom_contactrich.env_cfgs import (
  unitree_g1_custom_contactrich_flat_tracking_env_cfg,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg

# Idle crawl pose: the zero-twist clip and the robot's default joint angles.
IDLE_QPOS_CSV = DEFAULT_IDLE_CSV
# Timer-driven twist resampling [s]; the reference blends to the new clip.
RESAMPLING_TIME_RANGE = (3.0, 8.0)
BLEND_TIME_S = 0.4
# Fraction of resamples commanding a zero twist, i.e. the static idle clip.
REL_STATIC_ENVS = 0.15


def unitree_g1_crawling_env_cfg(
  motion_dir: str, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Contact-rich G1 crawl over the library in ``motion_dir``, with blended switches."""
  cfg = unitree_g1_custom_contactrich_flat_tracking_env_cfg(
    has_state_estimation=False, play=play
  )

  # Whole-body ground contact overflows mujoco-warp's default constraint cap.
  cfg.sim.njmax = 512

  # Centre the action space on the idle crawl pose instead of the standing default.
  _, _, joint_pos = load_idle_qpos(IDLE_QPOS_CSV)
  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.init_state = replace(robot_cfg.init_state, joint_pos=joint_pos)

  old = cfg.commands["motion"]
  assert isinstance(old, MotionCommandCfg)
  cfg.commands["motion"] = BlendingMotionCommandCfg(
    **library_command_fields(old, motion_dir),
    resampling_time_range=RESAMPLING_TIME_RANGE,
    rel_static_envs=(0.0 if play else REL_STATIC_ENVS),
    blend_time_s=BLEND_TIME_S,
  )

  make_actor_reference_free(cfg)

  # Light direct twist tracking toward the continuous commanded twist.
  cfg.rewards["twist"] = RewardTermCfg(
    func=mdp.twist_tracking,
    weight=0.3,
    params={"command_name": "motion", "std": 0.1},
  )
  use_egocentric_root_rewards(cfg)

  return cfg
