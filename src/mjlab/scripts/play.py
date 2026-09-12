"""Script to play RL agent with RSL-RL."""

import os
import sys
import time as _time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.crawling_fwd.mdp.commands import (
  LibraryMotionCommand,
  LibraryMotionCommandCfg,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago


def _parse_wandb_dt(value: str | datetime) -> datetime:
  """Parse a W&B datetime string (or pass through a datetime object)."""
  if isinstance(value, str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
  return value


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  no_disturbances: bool = False
  """Replay on the NOMINAL plant: drop every startup domain-randomization and interval (push)
  event, and for tracking tasks zero the reference-state-init joint noise. Reset-mode events are
  kept (they implement the reset itself). Play already disables pushes, observation noise and the
  RSI pose/velocity noise; this removes what is left."""
  twist: tuple[float, float, float] | None = None
  """Gait-library tasks only: pin every env to this commanded twist, COMMA-separated ``vx,vy,wz``
  (mjlab's tyro flags take tuples as one token): ``--twist 0,0,1.0`` for a left turn,
  ``--twist 0,0,0`` to stand, ``--twist -0.6,0,0`` to walk backward. In the native viewer the
  twist can also be driven live from the keyboard, see the key map printed at start-up; in viser,
  from the "Twist command" sliders."""
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def strip_disturbances(env_cfg) -> list[str]:
  """Make ``env_cfg`` a nominal-plant replay (see ``PlayConfig.no_disturbances``).

  Removes every event term whose mode is not ``reset`` -- startup DR (mass, COM, gains, friction,
  armature, encoder bias, ...) and interval pushes -- and zeroes the tracking command's RSI joint
  noise. Returns the names of the removed event terms."""
  removed = [k for k, v in env_cfg.events.items() if v.mode != "reset"]
  for k in removed:
    env_cfg.events.pop(k)
  motion_cmd = env_cfg.commands.get("motion")
  if isinstance(motion_cmd, MotionCommandCfg):
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
  return removed


TELEOP_HELP = """[teleop] gait-library twist keys (native viewer; applied to ALL envs):
  Up / Down          forward speed  +/- 0.05 m/s   (zeroes the yaw rate: translate XOR rotate)
  PageUp / PageDown  yaw rate       +/- 0.1 rad/s  (left / right; zeroes the forward speed)
  Delete             stop: zero twist -> the idle clip
  Home               release the pin -> back to the task's own twist sampler
  values clamp to the library's twist extent; the snapped clip is printed after each change
  (letters and [ ] are avoided: MuJoCo's viewer binds them to render toggles and camera cycling)"""


def make_twist_teleop(cmd: LibraryMotionCommand) -> Callable[[int], None]:
  """Keyboard twist teleop for a gait-library command (see TELEOP_HELP).

  Runs on the viewer thread, so it only calls ``set_fixed_twist`` (a mailbox the command drains
  on the sim thread). Speed and yaw keys are mutually exclusive by design, matching the
  differential-drive command space; on an omni library that is merely conservative."""
  from mjlab.viewer.native.keys import (
    KEY_DELETE,
    KEY_DOWN,
    KEY_HOME,
    KEY_PAGE_DOWN,
    KEY_PAGE_UP,
    KEY_UP,
  )

  lo = cmd.motion.lib_twists.min(dim=0).values.tolist()
  hi = cmd.motion.lib_twists.max(dim=0).values.tolist()
  start = cmd.fixed_twist or tuple(float(v) for v in cmd.twist_command[0].tolist())
  twist = [float(v) for v in start]

  def clamp(i: int, v: float) -> float:
    return round(min(max(v, lo[i]), hi[i]), 3)

  def callback(key: int) -> None:
    if key == KEY_UP:
      twist[0], twist[2] = clamp(0, twist[0] + 0.05), 0.0
    elif key == KEY_DOWN:
      twist[0], twist[2] = clamp(0, twist[0] - 0.05), 0.0
    elif key == KEY_PAGE_UP:
      twist[0], twist[2] = 0.0, clamp(2, twist[2] + 0.1)
    elif key == KEY_PAGE_DOWN:
      twist[0], twist[2] = 0.0, clamp(2, twist[2] - 0.1)
    elif key == KEY_DELETE:
      twist[:] = [0.0, 0.0, 0.0]
    elif key == KEY_HOME:
      cmd.set_fixed_twist(None)
      print("[teleop] pin released: back to the task's twist sampler")
      return
    else:
      return
    t = (twist[0], twist[1], twist[2])
    cmd.set_fixed_twist(t)
    snap = cmd.nearest_clip_twist(t)
    print(
      f"[teleop] twist vx {t[0]:+.2f} vy {t[1]:+.2f} wz {t[2]:+.2f}"
      f"  -> clip vx {snap[0]:+.2f} vy {snap[1]:+.2f} wz {snap[2]:+.2f}"
    )

  return callback


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Nominal plant if requested: no domain randomization, no pushes, exact RSI.
  if cfg.no_disturbances:
    removed = strip_disturbances(env_cfg)
    print(
      f"[INFO]: Disturbances disabled (removed events: {removed}; RSI noise zeroed)"
    )

  # Pin the commanded twist (gait-library tasks).
  if cfg.twist is not None:
    lib_cmd_cfg = env_cfg.commands.get("motion")
    if not isinstance(lib_cmd_cfg, LibraryMotionCommandCfg):
      raise ValueError(
        f"--twist needs a gait-library task (LibraryMotionCommandCfg); {task_id} has "
        f"{type(lib_cmd_cfg).__name__}"
      )
    lib_cmd_cfg.fixed_twist = cfg.twist
    print(f"[INFO]: Commanded twist pinned to {cfg.twist} for every env")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Build checkpoint manager for hot-swapping checkpoints in the viewer.
  ckpt_manager: CheckpointManager | None = None
  if TRAINED_MODE and resume_path is not None:
    _ckpt_runner = runner  # pyright: ignore[reportPossiblyUnboundVariable]

    def _reload_policy(path: str):
      _ckpt_runner.load(
        path,
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
      )
      return _ckpt_runner.get_inference_policy(device=device)

    if cfg.wandb_run_path is None:
      ckpt_dir = resume_path.parent

      def fetch_available_local() -> list[tuple[str, str]]:
        now = _time.time()
        entries: list[tuple[str, str, int]] = []
        for f in sorted(ckpt_dir.glob("*.pt")):
          try:
            step = int(f.stem.split("_")[1])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(int(now - f.stat().st_mtime))
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_local,
        load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
      )
    else:
      import wandb

      api = wandb.Api()
      run_path = str(cfg.wandb_run_path)
      wandb_run = api.run(run_path)
      _log_root = log_root_path  # pyright: ignore[reportPossiblyUnboundVariable]

      def fetch_available_wandb() -> list[tuple[str, str]]:
        wandb_run.load()
        now = datetime.now(tz=timezone.utc)
        entries: list[tuple[str, str, int]] = []
        for f in wandb_run.files():
          if not f.name.endswith(".pt"):
            continue
          try:
            step = int(f.name.split("_")[1].split(".")[0])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(
            int((now - _parse_wandb_dt(f.updated_at)).total_seconds())
          )
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_wandb,
        load_checkpoint=lambda name: _reload_policy(
          str(get_wandb_checkpoint_path(_log_root, Path(run_path), name)[0])
        ),
        run_name=_parse_wandb_dt(wandb_run.created_at).strftime("%Y-%m-%d_%H-%M-%S"),
        run_url=wandb_run.url,
        run_status=wandb_run.state,
      )

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    key_callback = None
    lib_cmd = env.unwrapped.command_manager._terms.get("motion")
    if isinstance(lib_cmd, LibraryMotionCommand):
      key_callback = make_twist_teleop(lib_cmd)
      print(TELEOP_HELP)
    NativeMujocoViewer(env, policy, key_callback=key_callback).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy, checkpoint_manager=ckpt_manager).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  maybe_print_top_level_help("play")

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
