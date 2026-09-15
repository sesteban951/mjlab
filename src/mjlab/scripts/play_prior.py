"""Play a solved control tape as the sole controller, with no policy in the loop."""

import json
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
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  get_locked_wrists_spec,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import (
  JointPositionActionCfg,
  JointPositionPriorReplayActionCfg,
)
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking_prior.mdp import motion_tape_prior
from mjlab.utils.spec_config import CollisionCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


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
  """Rate the prior is evaluated at. Default: once per control step, which is the tape's
  own frame rate. Raise it to run the feedback at the physics rate instead."""
  num_envs: int = 20
  device: str | None = None
  loop: bool | None = None
  """Reset and replay when the episode ends. Default: on under a viewer, off headless,
  where stopping at the end is what reports where the tape loses the robot."""
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
  match_solve_plant: bool = False
  """Put the sim on the plant the tape was solved on, read from the tape's own
  provenance (timestep, integrator, cone, contact params, solver budget). Applied before
  --sim-timestep / --sim-iterations / --integrator, so those still override it."""
  lock_wrists: bool = False
  """Swap the robot for the equality-pinned-wrist variant, matching mj-nlp's solve model.
  A tape solved under that lock carries wrist commands the constraint was absorbing."""
  clip_targets: bool = False
  """Clamp the replayed target to the entity's soft joint limits. Off here: mjlab's
  position actuators are ctrllimited=False precisely so a setpoint may leave the joint
  range, and clamping a solved tape to 90% of it deletes the torque the tape asked for.
  The prior keeps clamping by default everywhere it is a behavior-cloning target."""
  steps: int | None = None
  """Rollout length. Default: the tape's frame count."""


def _solve_plant(tape_file: Path) -> dict:
  """The mj-nlp DynamicsConfig the tape was solved under, from its own provenance."""
  tape = np.load(tape_file)
  prov = json.loads(str(tape["provenance"])) if "provenance" in tape else {}
  plant = prov.get("dynamics_config")
  if not plant:
    raise ValueError(
      f"{tape_file.name} carries no dynamics_config, so the plant it was solved on is "
      f"unknown and --match-solve-plant has nothing to apply. Re-export it from an "
      f"mj-nlp file that saved one, or set the sim options by hand."
    )
  return plant


def _apply_solve_plant(env_cfg, plant: dict) -> None:
  """Put the sim on the plant the tape was solved on, reading the numbers off the tape.

  The task's own settings are tuned for training throughput on thousands of envs, not
  for reproducing one solved trajectory: 10 solver iterations, a pyramidal cone, and a
  stiffer contact than the solve used. None of that is wrong for a policy, and all of it
  moves a tape whose gains have no margin.
  """
  mj = env_cfg.sim.mujoco
  mj.timestep = plant["sim_dt"]
  mj.integrator = plant["integrator"]
  mj.cone = plant["friction_cone"]
  mj.iterations, mj.ls_iterations = 100, 50
  # Contacts, on the robot's own geoms. priority 1 makes them win the pair against the
  # terrain, so these are the parameters that actually apply rather than a solmix blend.
  env_cfg.scene.entities["robot"].collisions = (
    CollisionCfg(
      geom_names_expr=(".*_collision",),
      condim=plant["condim"],
      priority=1,
      friction=tuple(plant["friction"]),
      solref=tuple(plant["solref"]),
      solimp=tuple(plant["solimp"]),
    ),
  )
  # The sideroll puts most of the robot on the ground at once; the task's 35 is sized for
  # a walking clip and this one overflows it, silently dropping contacts.
  env_cfg.sim.nconmax, env_cfg.sim.njmax = 256, 1000
  print(
    f"[INFO] plant: {1 / mj.timestep:g} Hz {mj.integrator}, {mj.cone} cone, "
    f"solver {mj.iterations}/{mj.ls_iterations}, condim {plant['condim']}, "
    f"mu {plant['friction'][0]}, solref {plant['solref']}, solimp {plant['solimp']}"
  )


def _build_env(cfg: PlayPriorConfig, device: str, tape_file: Path, fps: float):
  env_cfg = load_env_cfg(cfg.task, play=True)

  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = cfg.motion_file

  # Replace the task's action with one the tape drives on its own: no policy in the loop.
  action = env_cfg.actions["joint_pos"]
  if not isinstance(action, JointPositionActionCfg):
    raise TypeError(
      f"Task {cfg.task!r} has no joint-position action; its joint_pos action is a "
      f"{type(action).__name__}."
    )
  env_cfg.actions["joint_pos"] = (
    JointPositionPriorReplayActionCfg.from_joint_position_cfg(
      action,
      prior=motion_tape_prior,
      prior_params={
        "tape_file": str(tape_file),
        "command_name": "motion",
        "feedback": cfg.feedback,
        "alpha_scale": cfg.alpha_scale,
        "clip_to_joint_limits": cfg.clip_targets,
      },
      prior_frequency_hz=cfg.prior_hz,
    )
  )

  if cfg.match_solve_plant:
    _apply_solve_plant(env_cfg, _solve_plant(tape_file))
  if cfg.sim_timestep is not None:
    env_cfg.sim.mujoco.timestep = cfg.sim_timestep

  # Hold the control rate at the tape's own frame rate, whatever the physics timestep.
  # The task's 50 Hz is right for a policy; replaying a 100 Hz tape there holds every
  # sample for two of its frames, which is a different control signal, not a coarser one.
  dt = env_cfg.sim.mujoco.timestep
  ratio = 1.0 / (fps * dt)
  env_cfg.decimation = max(1, round(ratio))
  if abs(ratio - env_cfg.decimation) > 1e-6:
    raise ValueError(
      f"the tape's {fps:g} Hz does not divide the {1 / dt:g} Hz physics rate "
      f"(ratio {ratio:.4f}). Pass --sim-timestep 1/(k*{fps:g}) for an integer k."
    )
  print(
    f"[INFO] physics {1 / dt:.0f} Hz, decimation {env_cfg.decimation}"
    f" -> control {1 / (dt * env_cfg.decimation):.1f} Hz (tape {fps:g} Hz)"
  )
  if cfg.sim_iterations is not None:
    env_cfg.sim.mujoco.iterations = cfg.sim_iterations
  if cfg.integrator is not None:
    env_cfg.sim.mujoco.integrator = cfg.integrator

  if cfg.lock_wrists:
    robot = env_cfg.scene.entities["robot"]
    robot.spec_fn = get_locked_wrists_spec
    print("[INFO] robot: wrists pinned by equality (mj-nlp's solve model)")

  env_cfg.scene.num_envs = cfg.num_envs
  # Headless, hold the terminal state instead of resetting out from under the rollout:
  # that is what reports where the tape loses the robot. Under a viewer, replay on a loop.
  env_cfg.auto_reset = cfg.loop if cfg.loop is not None else cfg.viewer != "none"
  if not cfg.randomize:
    env_cfg.events = {}
    # Play mode zeroes the command's RSI pose and velocity ranges but leaves
    # joint_position_range at (-0.1, 0.1), so every reset still lands the robot with up to
    # 0.1 rad of independent noise on all 29 joints -- enough that a fixed tape survives
    # some replays and not others. A tape is scored against the state it was solved from.
    motion_cmd.joint_position_range = (0.0, 0.0)
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
  # Which design this actually is: two exports under one filename, differing only in the
  # gain, cost a day of debugging once.
  if "provenance" in tape:
    prov = json.loads(str(tape["provenance"]))
    print(
      f"[INFO] law:    {prov['law']}  (R x {prov['r_scale']:g}, "
      f"|K|max {prov['gain_max']:.1f})"
    )
    print(
      f"[INFO] from:   {prov['export_path']}  sha256 {prov['export_sha256'][:16]}\n"
      f"[INFO]         {1 / prov['sim_dt']:g} Hz {prov['spline_type']} tape on "
      f"{prov['node_dt']:g} s nodes, replayed at {float(tape['fps'][0]):g} Hz"
    )
  else:
    print("[WARN] tape carries no provenance; re-export it with lqr_export_to_tape.py")

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
