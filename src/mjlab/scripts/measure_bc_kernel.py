"""Measure where a policy sits inside ``action_prior_deviation_exp``'s kernel.

The BC term is ``w * exp(-mean((pi(o) - u_prior)^2) / std^2)``, so it has two knobs that
do different jobs. ``std`` is the per-joint RMS deviation in radians at which the kernel
falls to 1/e -- it sets *where* the reward is responsive. ``w`` scales what it is worth
once there. A policy sitting well past ``std`` is on the kernel's flat tail, and raising
``w`` there scales a small gradient rather than restoring a useful one; widening ``std``
is what moves the operating point back onto the slope.

This rolls a checkpoint out, reports the deviation the policy actually holds, and tables
the realized reward share and gradient for a grid of (std, weight).
"""

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

import mjlab
import mjlab.tasks  # noqa: F401  Populate the task registry.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPriorAction
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking_prior.config.g1.env_cfgs import (
  unitree_g1_box_rolldown_prior_env_cfg,
  unitree_g1_sideroll_prior_env_cfg,
  unitree_g1_tracking_prior_env_cfg,
)
from mjlab.utils.torch import configure_torch_backends

# Each clip's wrapper carries its own tuned constants, so the task id picks the factory.
CFG_FACTORIES = {
  "Mjlab-Tracking-Prior-BoxRolldown-Unitree-G1": unitree_g1_box_rolldown_prior_env_cfg,
  "Mjlab-Tracking-Prior-Sideroll-Unitree-G1": unitree_g1_sideroll_prior_env_cfg,
  "Mjlab-Tracking-Prior-Flat-Unitree-G1": unitree_g1_tracking_prior_env_cfg,
}


@dataclass(frozen=True)
class BcConfig:
  checkpoint: str
  motion_file: str
  task: str = "Mjlab-Tracking-Prior-BoxRolldown-Unitree-G1"
  num_envs: int = 256
  steps: int = 400
  device: str = "cuda:0"
  target_share: float = 0.1
  """Share of the per-step task reward the BC term should be worth."""
  stds: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
  out: str | None = None
  """Optional path to write the measured summary as JSON."""


def run(cfg: BcConfig) -> None:
  configure_torch_backends()

  # Measure on the control arm: a policy the BC term has never pulled on.
  os.environ["MJLAB_PRIOR_REWARD"] = "none"
  if cfg.task not in CFG_FACTORIES:
    raise KeyError(f"no env-cfg factory for {cfg.task!r}; add it to CFG_FACTORIES.")
  env_cfg = CFG_FACTORIES[cfg.task]()
  env_cfg.scene.num_envs = cfg.num_envs
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = cfg.motion_file
  env = ManagerBasedRlEnv(env_cfg, device=cfg.device)

  agent_cfg = load_rl_cfg(cfg.task)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=cfg.device)
  runner.load(
    cfg.checkpoint, load_cfg={"actor": True}, strict=True, map_location=cfg.device
  )
  policy = runner.get_inference_policy(device=cfg.device)

  term = env.action_manager.get_term("joint_pos")
  assert isinstance(term, JointPriorAction)
  rm = env.reward_manager

  mses: list[np.ndarray] = []
  totals: list[np.ndarray] = []
  obs = wrapped.get_observations()
  with torch.inference_mode():
    for _ in range(cfg.steps):
      obs, _, _, _ = wrapped.step(policy(obs))
      err = term.processed_actions - term.prior_target
      mses.append(torch.mean(err.square(), dim=1).float().cpu().numpy())
      totals.append(rm._step_reward.sum(dim=1).float().cpu().numpy())

  mse = np.concatenate(mses)
  task = float(np.median(np.abs(np.concatenate(totals))))
  rms = np.sqrt(mse)

  print(f"\nper-joint RMS deviation |pi(o) - u_prior| over {mse.size} samples (rad):")
  for p in (10, 25, 50, 75, 90):
    print(f"  p{p:<3} {float(np.percentile(rms, p)):.4f}")
  print(f"  mean {rms.mean():.4f}   median mse {np.median(mse):.5f}")
  print(f"\nmedian per-step task reward: {task:.4f}")

  m = float(np.median(mse))
  rows = []
  print("\n  std   kernel   w for target   share at w=0.2   dR/d(rms) at w for target")
  print("  " + "-" * 70)
  for std in cfg.stds:
    kernel = float(np.exp(-m / std**2))
    # Solve w * kernel = target * (task - current + w * kernel) for w.
    w = cfg.target_share * task / (kernel * (1.0 - cfg.target_share))
    # d/d(rms) of w*exp(-rms^2/std^2), the pull the term exerts at the operating point.
    grad = 2.0 * w * np.sqrt(m) / std**2 * kernel
    share02 = 0.2 * kernel / task
    rows.append(
      {"std": float(std), "kernel": kernel, "w_for_target": float(w),
       "share_at_w02": float(share02), "grad": float(grad)}
    )  # fmt: skip
    print(
      f"  {std:.2f}  {kernel:6.3f}   {w:9.2f}      {share02:8.1%}        {grad:10.2f}"
    )
  print(
    f"\n  kernel = exp(-{m:.5f}/std^2); 'w for target' puts the term at "
    f"{cfg.target_share:.0%} of the task reward."
  )
  print("  A kernel far below ~0.37 (=1/e) means the policy sits past one std,")
  print("  out where the exp tail is flattening and weight alone buys little shape.")

  if cfg.out:
    summary = {
      "checkpoint": cfg.checkpoint,
      "motion_file": cfg.motion_file,
      "median_mse": m,
      "median_rms_rad": float(np.sqrt(m)),
      "task_reward_per_step_median": float(task),
      "target_share": cfg.target_share,
      "rows": rows,
    }
    Path(cfg.out).write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {cfg.out}")


def main():
  run(tyro.cli(BcConfig, prog=sys.argv[0], config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
