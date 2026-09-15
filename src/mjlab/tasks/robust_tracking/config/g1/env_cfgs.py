"""Unitree G1 CLF-guided sideroll tracking, tuned for sim2real transfer."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np

import mjlab
from mjlab.asset_zoo.robots.unitree_g1.custom_dr import (
  ContactDRCfg,
  add_custom_g1_actuator_delay,
  add_custom_g1_contact_dr,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.crawling_common.library import LIBRARY_SPECS, load_idle_qpos
from mjlab.tasks.tracking.config.g1_custom.env_cfgs import (
  unitree_g1_custom_flat_tracking_env_cfg,
)
from mjlab.tasks.tracking.mdp import metrics as tracking_metrics
from mjlab.tasks.tracking.mdp import tvlqr
from mjlab.tasks.tracking.mdp.rewards import (
  clf_decrease_rbf,
  clf_value_kernel,
  qdes_imitation_rbf,
)
from mjlab.tasks.tracking.mdp.terminations import motion_complete
from mjlab.terrains import (
  BoxFlatTerrainCfg,
  BoxTiltedPlaneTerrainCfg,
  TerrainEntityCfg,
  TerrainGeneratorCfg,
)

# Both files come from one mj-nlp solve, and motion.npz is built from the export's own
# x_bar, which is what keeps the clip and the gain schedule aligned. Rebuild: see below.
_TRAJ_DIR = (
  Path(mjlab.MJLAB_SRC_PATH).parent.parent
  / "trajectories"
  / "control"
  / "sideroll_mimic_svd_alpha_v3"
)
# motion.npz behind a 1 s stand and ahead of a 2 s end hold, from pad_motion.py.
MOTION_FILE = _TRAJ_DIR / "motion_padded.npz"
STAND_QPOS_CSV = LIBRARY_SPECS["standing_diffdrive"].idle_csv
STAND_HOLD_S, END_HOLD_S = 1.0, 2.0
TVLQR_EXPORT = _TRAJ_DIR / "sideroll_tvlqr_lyap_closed.npz"

# Physics at 200 Hz; tvlqr strides the 100 Hz schedule and holds K between entries.
PHYSICS_DT = 0.005
DECIMATION = 4

# Same seconds as G1-Tracking-Custom; only the step count behind it moves.
ACTUATOR_DELAY_RANGE_S = (0.0, 0.04)

# RBF widths, re-measured on a 32-env x 200-step zero-action rollout against the fixed clip
# and the closed-loop P: each gives its term G1-Tracking-Control's target mean reward there
# (0.70 clf on a relative violation of p50 0.11, 0.50 qdes on p50 4.5 rad).
CLF_SIGMA = 0.63
QDES_SIGMA = 2.55

# From G1-Tracking-Control. A starting point, not a tuned result.
CLF_WEIGHT = 1.0
QDES_WEIGHT = 1.0

# A mild band around MuJoCo's stock (0.02, 1.0). MuJoCo clamps timeconst at 2 * PHYSICS_DT,
# and the clamped contact still rebounds off hard impacts, so the floor sits above it.
SOLREF_TIMECONST = (0.012, 0.030)
SOLREF_DAMPRATIO = (0.9, 1.1)

# Ground tilt as DR, not as terrain the policy is meant to perceive. Patches are sized so
# the clip's 2.9 m body reach stays on the one it spawned on.
MAX_TILT_DEG = 3.0
TILTED_PROPORTION = 0.6
TERRAIN_PATCH_SIZE = (8.0, 8.0)
TERRAIN_NUM_ROWS = 10
TERRAIN_NUM_COLS = 10

# 2 m of drift on a 3 deg slope is ~0.10 m of ground-height offset the flat clip cannot know.
SLOPE_Z_ALLOWANCE = 0.10

# episode_length_s is a ceiling only; motion_complete is what ends an episode.
EPISODE_LENGTH_MARGIN_S = 0.5

# Train/r_<group> curves: what the reward sums to per category, on the Episode_Reward
# scale. Log-only -- the manager sums existing terms and changes nothing it optimizes.
REWARD_GROUPS = {
  "mimic_pos": (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
  ),
  "mimic_vel": ("motion_body_lin_vel", "motion_body_ang_vel"),
  "clf": ("clf_decrease", "qdes_imitation"),
  "regularization": (
    "action_rate_limbs",
    "action_rate_waist",
    "joint_limit",
    "self_collisions",
  ),
}


def unitree_g1_robust_tracking_env_cfg(
  has_state_estimation: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """G1 sideroll tracking with guide-only CLF shaping, hardened for transfer.

  ``has_state_estimation`` defaults to False: the anchor position and base linear velocity
  are not measurable on the robot. The critic keeps them either way.
  """
  cfg = unitree_g1_custom_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation, play=play
  )

  for path, what in ((MOTION_FILE, "motion"), (TVLQR_EXPORT, "TVLQR export")):
    if not path.exists():
      raise FileNotFoundError(
        f"{what} not found at {path}. Rebuild both from the mj-nlp sideroll solve:\n"
        f"  uv run python -m mjlab.scripts.mjnlp_solve_to_tvlqr \\\n"
        f"    --solve-file <mj-nlp>/examples/full_run/g1_mimic_DR/"
        f"g1_mimic_svd_alpha_sideroll_v3.npz \\\n"
        f"    --output-file {TVLQR_EXPORT}\n"
        f"  # build the clip from THAT export, so the two cannot drift apart:\n"
        f"  uv run python -m mjlab.scripts.tvlqr_to_motion_csv \\\n"
        f"    --export-file {TVLQR_EXPORT} --output-file /tmp/sideroll_qpos.csv\n"
        f"  uv run python -m mjlab.scripts.csv_to_npz --input-file /tmp/sideroll_qpos.csv \\\n"
        f"    --output-name sideroll --input-fps 100 --output-fps 50\n"
        f"  # csv_to_npz always writes /tmp/motion.npz (--output-name is a wandb\n"
        f"  # collection, not a path), so finally:\n"
        f"  cp /tmp/motion.npz {_TRAJ_DIR / 'motion.npz'}\n"
        f"  uv run python scripts/tools/pad_motion.py \\\n"
        f"    --motion-file {_TRAJ_DIR / 'motion.npz'} --output-file {MOTION_FILE} \\\n"
        f"    --stand-csv {STAND_QPOS_CSV} --start-s {STAND_HOLD_S} "
        f"--end-s {END_HOLD_S}"
      )

  cfg.sim.mujoco.timestep = PHYSICS_DT
  cfg.decimation = DECIMATION
  # The delay is configured in STEPS, so it must be re-applied at the new timestep.
  if not play:
    add_custom_g1_actuator_delay(cfg, *ACTUATOR_DELAY_RANGE_S)

  # The stock capsule sole is kept, unlike G1-Tracking-Control's 4-sphere swap: the law only
  # scores the policy here, so a plant it was not designed for costs shaping, not stability.

  cfg.commands["motion"] = replace(cfg.commands["motion"], motion_file=str(MOTION_FILE))

  # A zero action is the standing idle the start pad holds; the motion command owns the reset.
  pos, quat, joint_pos = load_idle_qpos(STAND_QPOS_CSV)
  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.init_state = replace(
    robot_cfg.init_state, pos=pos, rot=quat, joint_pos=joint_pos
  )

  base_action = cast(JointPositionActionCfg, cfg.actions["joint_pos"])
  cfg.actions["joint_pos"] = tvlqr.TvlqrGuidedJointPositionActionCfg(
    entity_name=base_action.entity_name,
    actuator_names=tuple(base_action.actuator_names),
    scale=base_action.scale,
    use_default_offset=True,  # offset from the stand above, NOT from u_bar
    export_path=str(TVLQR_EXPORT),
    motion_command_name="motion",
    motion_pad_start=int(np.load(MOTION_FILE)["pad_frames"][0]),
  )

  # Both kernels are zeroed in the standing pads; the tracking rewards alone teach the hold.
  cfg.rewards["clf_decrease"] = RewardTermCfg(
    func=clf_decrease_rbf,
    weight=CLF_WEIGHT,
    params={"action_name": "joint_pos", "sigma": CLF_SIGMA},
  )
  cfg.rewards["qdes_imitation"] = RewardTermCfg(
    func=qdes_imitation_rbf,
    weight=QDES_WEIGHT,
    params={"action_name": "joint_pos", "sigma": QDES_SIGMA},
  )

  cfg.reward_groups = dict(REWARD_GROUPS)

  # The raw quantities the two CLF kernels pass through their RBFs. Measured, never
  # optimized: if a term sits pinned, these say whether the sigma or the policy is wrong.
  for name, func in (
    ("clf_viol_rel", tracking_metrics.clf_violation_rel),
    ("clf_V", tracking_metrics.clf_value),
    ("qdes_dist", tracking_metrics.qdes_distance),
  ):
    cfg.metrics[name] = MetricsTermCfg(
      func=func, params={"action_name": "joint_pos"}, log_prefix="Train"
    )

  # Applied after the timestep change so the 2*dt floor check uses the rate in force.
  add_custom_g1_contact_dr(
    cfg,
    ContactDRCfg(solref_timeconst=SOLREF_TIMECONST, solref_dampratio=SOLREF_DAMPRATIO),
  )

  # curriculum=False draws a difficulty per patch, and randomize_terrain re-draws each env's
  # patch on reset, so an env sees a new tilt every episode rather than one fixed at startup.
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      size=TERRAIN_PATCH_SIZE,
      num_rows=TERRAIN_NUM_ROWS,
      num_cols=TERRAIN_NUM_COLS,
      curriculum=False,
      difficulty_range=(0.0, 1.0),
      sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=1.0 - TILTED_PROPORTION),
        "tilted": BoxTiltedPlaneTerrainCfg(
          proportion=TILTED_PROPORTION, max_tilt_deg=MAX_TILT_DEG
        ),
      },
    ),
  )
  # Runs before the motion command's reset, which re-places the robot on the new patch.
  cfg.events["randomize_terrain"] = EventTermCfg(
    func=envs_mdp.randomize_terrain, mode="reset", params={}
  )

  # The clip was solved flat, so the tilt reads as pure z error against the reference.
  for name in ("anchor_pos", "ee_body_pos"):
    cfg.terminations[name].params["threshold"] += SLOPE_Z_ALLOWANCE

  # One clip is one episode: the base wraps and teleports at the end, which suits a periodic
  # gait but not a one-shot motion. time_out=True so finishing bootstraps rather than zeroes.
  cfg.terminations["motion_complete"] = TerminationTermCfg(
    func=motion_complete, params={"command_name": "motion"}, time_out=True
  )
  n_frames = int(np.load(MOTION_FILE)["joint_pos"].shape[0])
  clip_s = n_frames * cfg.sim.mujoco.timestep * cfg.decimation
  cfg.episode_length_s = clip_s + EPISODE_LENGTH_MARGIN_S

  return cfg


# Per-schedule-entry median V of G1-Robust-Tracking model_20000 under training DR.
CLF_V_REF = _TRAJ_DIR / "sideroll_v_ref_k.npy"
CLF_DECREASE_SIGMA = 0.3
CLF_V_FLOOR_SCALE = 0.1
CLF_TRACKING_BETA = 0.5
CLF_TRACKING_WEIGHT = 1.0


def unitree_g1_clf_tracking_env_cfg(
  has_state_estimation: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """G1-Robust-Tracking plus a per-k V tracking kernel and a floored CLF decrease term."""
  if not CLF_V_REF.exists():
    raise FileNotFoundError(f"V reference not found at {CLF_V_REF}.")
  cfg = unitree_g1_robust_tracking_env_cfg(
    has_state_estimation=has_state_estimation, play=play
  )

  action = cast(tvlqr.TvlqrGuidedJointPositionActionCfg, cfg.actions["joint_pos"])
  cfg.actions["joint_pos"] = replace(
    action, v_ref_path=str(CLF_V_REF), v_floor_scale=CLF_V_FLOOR_SCALE
  )

  cfg.rewards["clf_decrease"].params["sigma"] = CLF_DECREASE_SIGMA
  cfg.rewards["clf_tracking"] = RewardTermCfg(
    func=clf_value_kernel,
    weight=CLF_TRACKING_WEIGHT,
    params={"action_name": "joint_pos", "beta": CLF_TRACKING_BETA},
  )
  cfg.reward_groups["clf"] = (*cfg.reward_groups["clf"], "clf_tracking")

  cfg.metrics["clf_V_ratio"] = MetricsTermCfg(
    func=tracking_metrics.clf_value_ratio,
    params={"action_name": "joint_pos"},
    log_prefix="Train",
  )
  return cfg


# The three guidance rewards each ablation arm keeps; every other reward stays on.
CLF_ABLATION_ARMS: dict[str, tuple[str, ...]] = {
  "Traj": (),
  "CLF": ("clf_decrease", "clf_tracking"),
  "Qdes": ("qdes_imitation",),
  "All": ("clf_decrease", "clf_tracking", "qdes_imitation"),
}


def unitree_g1_clf_ablation_env_cfg(
  arm: str, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """G1-CLF-Tracking with only ``CLF_ABLATION_ARMS[arm]`` of the guidance rewards."""
  cfg = unitree_g1_clf_tracking_env_cfg(play=play)
  kept = CLF_ABLATION_ARMS[arm]
  for name in set(CLF_ABLATION_ARMS["All"]) - set(kept):
    del cfg.rewards[name]
  cfg.reward_groups["clf"] = tuple(n for n in cfg.reward_groups["clf"] if n in kept)
  if not cfg.reward_groups["clf"]:
    del cfg.reward_groups["clf"]
  return cfg
