"""Play a solved control tape as the sole controller, with no policy in the loop."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

import mjlab
import mjlab.tasks  # noqa: F401  Populate the task registry.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionWithPriorCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking_prior.mdp import motion_tape_prior
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

# lam the prior runs at. The policy keeps 1 / (1 + lam) of the command, so this leaves it
# a millionth of the authority -- near enough to a pure prior, and finite, unlike inf,
# which would make lam / (1 + lam) a NaN.
PURE_PRIOR_LAM = 1e6


@dataclass(frozen=True)
class PlayPriorConfig:
  motion_file: str
  """Motion .npz for the tracking command (the ghost the tape is scored against)."""
  tape_file: str | None = None
  """Control tape .npz. Default: <motion_file>_prior.npz."""
  task: str = "Mjlab-Tracking-Prior-Flat-Unitree-G1"
  feedback: bool = True
  """Close the tape's LQR loop. Ignored (with a warning) if the tape has no gains."""
  alpha_scale: float = 1.0
  """Extra multiplier on the tape's own alpha schedule."""
  prior_hz: float | None = None
  """Rate the prior is evaluated at. Default: once per control step. 200 runs the
  feedback at the physics rate, which is what the gains were designed for."""
  num_envs: int = 1
  device: str | None = None
  viewer: Literal["none", "auto", "native", "viser"] = "none"
  no_terminations: bool = False
  sim_timestep: float | None = None
  """Physics timestep. Decimation is rescaled to hold the control rate at the tape's fps,
  so this tests the tape against the integrator settings it was solved with."""
  sim_iterations: int | None = None
  integrator: Literal["euler", "implicitfast"] | None = None
  randomize: bool = False
  """Keep the task's startup randomization (COM offset, encoder bias, foot friction).
  Off by default: a tape is scored against the model it was solved on."""
  steps: int | None = None
  """Rollout length. Default: the tape's frame count."""


def _build_env(cfg: PlayPriorConfig, device: str, tape_file: Path, fps: float):
  env_cfg = load_env_cfg(cfg.task, play=True)

  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = cfg.motion_file

  action = env_cfg.actions["joint_pos"]
  if not isinstance(action, JointPositionActionWithPriorCfg):
    raise TypeError(
      f"Task {cfg.task!r} has no prior-blended action term; its joint_pos action is a "
      f"{type(action).__name__}."
    )
  action.prior = motion_tape_prior
  action.prior_params = {
    "tape_file": str(tape_file),
    "command_name": "motion",
    "feedback": cfg.feedback,
    "alpha_scale": cfg.alpha_scale,
  }
  action.prior_frequency_hz = cfg.prior_hz
  action.lam = PURE_PRIOR_LAM

  if cfg.sim_timestep is not None:
    env_cfg.sim.mujoco.timestep = cfg.sim_timestep
    env_cfg.decimation = max(1, round(1.0 / (fps * cfg.sim_timestep)))
    print(
      f"[INFO] physics {1 / cfg.sim_timestep:.0f} Hz, decimation {env_cfg.decimation}"
      f" -> control {1 / (cfg.sim_timestep * env_cfg.decimation):.1f} Hz"
    )
  if cfg.sim_iterations is not None:
    env_cfg.sim.mujoco.iterations = cfg.sim_iterations
  if cfg.integrator is not None:
    env_cfg.sim.mujoco.integrator = cfg.integrator

  env_cfg.scene.num_envs = cfg.num_envs
  # Report where the tape actually loses the robot instead of resetting out from under it.
  env_cfg.auto_reset = False
  if not cfg.randomize:
    env_cfg.events = {}
  if cfg.no_terminations:
    env_cfg.terminations = {}
  return ManagerBasedRlEnv(cfg=env_cfg, device=device)


def _rollout(env: ManagerBasedRlEnv, steps: int) -> None:
  """Step the tape open loop and report how far it gets and how well it tracks."""
  command = env.command_manager.get_term("motion")
  action = torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )

  env.reset()
  history: dict[str, list[float]] = {}
  survived = steps
  fired: list[str] = []
  for step in range(steps):
    _, _, terminated, truncated, _ = env.step(action)
    for key, value in command.metrics.items():
      if key.startswith("error_"):
        history.setdefault(key, []).append(float(value.mean()))
    if bool((terminated | truncated).any()):
      survived = step + 1
      fired = [
        name
        for name in env.termination_manager.active_terms
        if bool(env.termination_manager.get_term(name).any())
      ]
      break

  dt = env.step_dt
  print(
    f"\nsurvived {survived}/{steps} frames ({survived * dt:.2f}/{steps * dt:.2f} s)"
    + (
      ""
      if survived == steps
      else f"  <- terminated on {', '.join(fired) or 'time out'}"
    )
  )
  print(f"{'metric':<22}{'mean':>12}{'max':>12}")
  for key, values in history.items():
    v = np.asarray(values)
    print(f"{key:<22}{v.mean():>12.4f}{v.max():>12.4f}")


def run(cfg: PlayPriorConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  tape_file = Path(
    cfg.tape_file
    if cfg.tape_file is not None
    else str(Path(cfg.motion_file).with_suffix("")) + "_prior.npz"
  )
  if not tape_file.exists():
    raise FileNotFoundError(f"Control tape not found: {tape_file}")
  tape = np.load(tape_file)
  if cfg.feedback and "gain" not in tape:
    print(f"[WARN] {tape_file.name} carries no gain schedule; replaying u open loop.")
  print(f"[INFO] motion: {cfg.motion_file}")
  print(
    f"[INFO] tape:   {tape_file}  ({tape['u'].shape[0]} frames, "
    f"feedback={cfg.feedback and 'gain' in tape})"
  )

  env = _build_env(cfg, device, tape_file, float(tape["fps"][0]))

  if cfg.viewer == "none":
    _rollout(env, cfg.steps or int(tape["u"].shape[0]))
    env.close()
    return

  wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(cfg.task).clip_actions)
  zeros = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=device)

  def policy(obs):
    del obs
    return zeros

  backend = cfg.viewer
  if backend == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    backend = "native" if has_display else "viser"
  if backend == "native":
    NativeMujocoViewer(wrapped, policy).run()
  else:
    ViserPlayViewer(wrapped, policy).run()
  wrapped.close()


def main():
  run(tyro.cli(PlayPriorConfig, prog=sys.argv[0], config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
