"""Watch a gait library's reference (the green ghost) run sequences of commands.

No policy and no physics: the motion command is stepped as in training and the robot
is posed on the reference, so the ghost is what a perfect tracker is asked to follow.
Each example is a fresh episode of a few commanded twists, blended like in training.

  uv run python -m mjlab.scripts.view_reference G1-Walk-Unicycle-LSTM
  uv run python -m mjlab.scripts.view_reference G1-Jog-Unicycle-LSTM --order library
"""

import os
import random
import sys
import threading
from dataclasses import dataclass
from typing import Any, Literal

import torch
import tyro

import mjlab
import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.gait_library.mdp.commands import (
  LibraryMotionCommand,
  LibraryMotionCommandCfg,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

_NEVER = 1.0e9  # [s] command timer that never fires: the player picks the switches.


@dataclass(frozen=True)
class ViewReferenceConfig:
  hold: float = 3.0
  """Seconds each commanded twist is held before the next one blends in."""
  segments: int = 4
  """Commanded twists per example; the episode then resets and a new example starts."""
  examples: int | None = None
  """Stop after this many examples; None runs until the viewer is closed."""
  order: Literal["sampled", "library"] = "sampled"
  """``sampled`` draws twists from the task's training sampler; ``library`` tours every
  clip in file order."""
  motion_dir: str | None = None
  """Library to show instead of the task's own (a ``*_tracking`` directory)."""
  show_robot: bool = False
  """Also draw the robot, posed on the reference (it coincides with the ghost)."""
  seed: int | None = None
  """None draws a new seed on every run; the one used is printed."""
  device: str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"


def _fmt(twist: torch.Tensor) -> str:
  vx, vy, wz = (float(v) for v in twist)
  return f"vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f}"


class ReferencePlayer:
  """Stands in for the env in a viewer: steps the motion command, never the physics."""

  def __init__(self, env: ManagerBasedRlEnv, cfg: ViewReferenceConfig):
    cmd = env.command_manager.get_term("motion")
    assert isinstance(cmd, LibraryMotionCommand)
    self.env, self.cmd, self._cfg = env, cmd, cfg
    self.num_envs = env.num_envs
    self.hold_steps = max(1, round(cfg.hold / env.step_dt))
    self._steps = 0
    self._example = 0
    self._clip = 0  # next clip of the library tour
    self._dones = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  # --- what a viewer reads from an env ---

  @property
  def device(self) -> torch.device | str:
    return self.env.device

  @property
  def cfg(self) -> Any:
    return self.env.cfg

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self.env

  def get_observations(self) -> None:
    return None

  def close(self) -> None:
    self.env.close()

  def reset(self) -> None:
    """Start the next example: a fresh episode, RSI'd onto its first reference."""
    if self._cfg.order == "library":
      # Apply the pin before the reset so the episode starts IN that clip.
      self._pin_next_clip()
      self.cmd.compute(0.0)
    self.env.reset()
    self._pose_robot()
    self._steps = 0
    self._example += 1
    self._announce(1)

  def step(self, actions: Any) -> tuple[None, None, torch.Tensor, dict]:
    del actions  # Unused: nothing is simulated.
    segment, into = divmod(self._steps, self.hold_steps)
    if into == 0 and segment == self._cfg.segments:
      self.reset()
      return None, None, self._dones, {}
    switch = into == 0 and segment > 0
    if switch and self._cfg.order == "library":
      self._pin_next_clip()
    elif switch:
      self.cmd.time_left[:] = 0.0  # fire the task's own timer resample
    # The robot sits on the current reference, as after a perfectly tracked step.
    self.cmd.compute(self.env.step_dt)
    self._pose_robot()
    self._steps += 1
    if switch:
      self._announce(segment + 1)
    return None, None, self._dones, {}

  # --- internals ---

  def _pin_next_clip(self) -> None:
    twist = self.cmd.motion.lib_twists[self._clip]
    self.cmd.set_fixed_twist((float(twist[0]), float(twist[1]), float(twist[2])))
    self._clip = (self._clip + 1) % self.cmd.motion.num_clips

  def _pose_robot(self) -> None:
    """Put the robot's root and joints on the served reference (zero velocity)."""
    cmd = self.cmd
    root = torch.cat([cmd.body_pos_w[:, 0], cmd.body_quat_w[:, 0]], dim=-1)
    root = torch.cat([root, torch.zeros_like(root[:, :6])], dim=-1)
    cmd.robot.write_root_state_to_sim(root)
    cmd.robot.write_joint_state_to_sim(cmd.joint_pos, torch.zeros_like(cmd.joint_pos))
    self.env.sim.forward()

  def _announce(self, segment: int) -> None:
    env_idx = self.env.cfg.viewer.env_idx
    clip = int(self.cmd.clip_idx[env_idx])
    print(
      f"[example {self._example:3d} | {segment}/{self._cfg.segments}]"
      f" command {_fmt(self.cmd.twist_command[env_idx])}"
      f"  ->  clip {clip:3d} {_fmt(self.cmd.motion.lib_twists[clip])}",
      flush=True,
    )


def build_player(task_id: str, cfg: ViewReferenceConfig) -> ReferencePlayer:
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=True)
  motion = load_env_cfg(task_id, play=False).commands.get("motion")
  if not isinstance(motion, LibraryMotionCommandCfg):
    raise ValueError(
      f"{task_id} is not a gait-library task: its motion command is "
      f"{type(motion).__name__}, not a LibraryMotionCommandCfg"
    )
  # Training's twist sampler on play's undisturbed scene, with an exact RSI.
  motion.pose_range, motion.velocity_range = {}, {}
  motion.joint_position_range = (0.0, 0.0)
  motion.sampling_mode = "uniform"
  motion.resampling_time_range = (_NEVER, _NEVER)
  motion.debug_vis = True
  motion.viz.mode = "ghost"
  if cfg.motion_dir is not None:
    motion.motion_dir = cfg.motion_dir
  env_cfg.commands["motion"] = motion
  env_cfg.scene.num_envs = 1
  env_cfg.seed = random.randrange(2**31) if cfg.seed is None else cfg.seed

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  if not cfg.show_robot:
    # Hide the robot's geoms; the ghost model is built later and recoloured.
    geom_ids = env.scene[motion.entity_name].indexing.geom_ids.cpu().numpy()
    env.sim.mj_model.geom_rgba[geom_ids, 3] = 0.0
  env.reset()  # forward the kinematics once before the command is stepped
  player = ReferencePlayer(env, cfg)
  player.reset()
  return player


def _join_viewer_thread(timeout: float = 10.0) -> None:
  """Wait for MuJoCo's viewer thread to destroy its window.

  Otherwise GLFW's exit handler races it and the process hangs or segfaults."""
  for thread in threading.enumerate():
    if thread.daemon and "_launch_internal" in thread.name:
      thread.join(timeout)


def run_view_reference(task_id: str, cfg: ViewReferenceConfig) -> None:
  configure_torch_backends()
  player = build_player(task_id, cfg)

  def policy(obs: torch.Tensor) -> torch.Tensor:
    del obs  # Unused: the player ignores actions.
    return torch.empty(0)

  viewer = cfg.viewer
  if viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    viewer = "native" if has_display else "viser"
  print(
    f"[INFO] {cfg.order} order, {cfg.segments} twists of {cfg.hold:g} s per example,"
    f" seed {player.env.cfg.seed}. Reset the viewer to skip to the next example."
  )
  # An example is its segments plus the one step that resets into the next.
  per_example = cfg.segments * player.hold_steps + 1
  num_steps = None if cfg.examples is None else cfg.examples * per_example - 1
  if viewer == "native":
    NativeMujocoViewer(player, policy).run(num_steps=num_steps)
    _join_viewer_thread()
  else:
    ViserPlayViewer(player, policy).run(num_steps=num_steps)
  player.close()


def main() -> None:
  maybe_print_top_level_help("view_reference")
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    ViewReferenceConfig,
    args=remaining_args,
    default=ViewReferenceConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_view_reference(chosen_task, args)


if __name__ == "__main__":
  main()
