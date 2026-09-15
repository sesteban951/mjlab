"""Measure the LQR Lyapunov value a trained policy actually operates at.

``lqr_lyapunov_shaping``'s ``v_half`` is the half-saturation point of the potential, and
the reward's own docstring is explicit that the V it should sit near is a property of the
*policy*, not of P -- so it has to be re-measured per motion, against a policy trained on
that motion. This rolls a checkpoint out on its clip, records ``V = e^T P[k] e`` with the
same tangent error the reward builds, and reports the percentiles plus the ``kappa`` that
puts the shaping term at a target share of the task rewards.

Run it on the control arm's checkpoint, then feed the numbers back into the env cfg.
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
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking_prior.config.g1.env_cfgs import (
  unitree_g1_box_rolldown_prior_env_cfg,
  unitree_g1_sideroll_prior_env_cfg,
  unitree_g1_tracking_prior_env_cfg,
)
from mjlab.tasks.tracking_prior.config.g1.rl_cfg import (
  unitree_g1_tracking_prior_ppo_runner_cfg,
)
from mjlab.tasks.tracking_prior.mdp.priors import tangent_state_error
from mjlab.utils.torch import configure_torch_backends

# Each clip's wrapper carries its own tuned constants, so the task id picks the factory.
CFG_FACTORIES = {
  "Mjlab-Tracking-Prior-BoxRolldown-Unitree-G1": unitree_g1_box_rolldown_prior_env_cfg,
  "Mjlab-Tracking-Prior-Sideroll-Unitree-G1": unitree_g1_sideroll_prior_env_cfg,
  "Mjlab-Tracking-Prior-Flat-Unitree-G1": unitree_g1_tracking_prior_env_cfg,
}


@dataclass(frozen=True)
class MeasureConfig:
  checkpoint: str
  """Trained policy to measure. Use the control arm's, not one already shaped by V."""
  motion_file: str
  tape_file: str
  """The ``*_prior.npz`` carrying the P the reward would use."""
  task: str = "Mjlab-Tracking-Prior-BoxRolldown-Unitree-G1"
  num_envs: int = 256
  steps: int = 1000
  device: str = "cuda:0"
  gamma: float | None = None
  """Discount for the kappa suggestion. Default: the task's own PPO gamma."""
  target_share: float = 0.1
  """Share of the per-step task reward the shaping term should be worth."""
  alpha: float = 0.01
  """Per-step decay rate to report clf_decrease_rbf's violation against."""
  sigma: float = 0.5
  """RBF width to report clf_decrease_rbf's mean reward at."""
  out: str | None = None
  """Optional path to write the measured summary as JSON."""


def run(cfg: MeasureConfig) -> None:
  configure_torch_backends()

  # The control arm: measure V under a policy the shaping terms never touched.
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

  tape = np.load(cfg.tape_file)
  if "P" not in tape:
    raise KeyError(f"{cfg.tape_file} has no 'P'; there is no cost-to-go to measure.")

  def _to(key: str) -> torch.Tensor:
    return torch.tensor(tape[key], dtype=torch.float32, device=cfg.device)

  P, qpos_ref, qvel_ref = _to("P"), _to("ref_qpos"), _to("ref_qvel")
  num_frames = qpos_ref.shape[0]

  entity = env.scene["robot"]
  command = env.command_manager.get_term("motion")
  assert isinstance(command, MotionCommand)
  rm = env.reward_manager
  term_names = list(rm.active_terms)

  # Keep the (step, env) grid rather than a flat list: F differences one env across two
  # consecutive steps, so the time axis has to survive.
  vals = np.zeros((cfg.steps, env.num_envs), dtype=np.float32)
  valid = np.zeros((cfg.steps, env.num_envs), dtype=bool)
  steps_r = np.zeros((cfg.steps, env.num_envs, len(term_names)), dtype=np.float32)
  # V is only comparable step to step within an episode; drop the step after a reset.
  fresh = torch.ones(env.num_envs, dtype=torch.bool, device=cfg.device)

  obs = wrapped.get_observations()
  with torch.inference_mode():
    for t in range(cfg.steps):
      obs, _, dones, _ = wrapped.step(policy(obs))
      k = command.time_steps.clamp(max=num_frames - 1)
      e = tangent_state_error(env, entity, qpos_ref[k], qvel_ref[k])
      v = torch.einsum("bi,bij,bj->b", e, P[k], e)
      vals[t] = v.float().cpu().numpy()
      valid[t] = (~fresh).cpu().numpy()
      steps_r[t] = rm._step_reward.float().cpu().numpy()
      fresh = dones.bool()

  V = vals[valid]
  per_term = steps_r[valid]
  # Same discount the shaping term is configured with, or its telescoping is off.
  ppo_gamma = unitree_g1_tracking_prior_ppo_runner_cfg().algorithm.gamma
  gamma = cfg.gamma if cfg.gamma is not None else ppo_gamma

  pct = {f"p{p}": float(np.percentile(V, p)) for p in (5, 10, 25, 50, 75, 90, 95, 99)}
  v_half = pct["p50"]
  task_per_step = float(np.median(np.abs(per_term.sum(axis=1))))

  # F = kappa * (phi(V_prev) - gamma * phi(V)) with phi = V / (V + v_half). Take the
  # pairs where one env stayed in the same episode across both steps, then solve for
  # the kappa that puts |F| at the target share of the task reward.
  phi = vals / (vals + v_half)
  pair = valid[:-1] & valid[1:]
  f_unit = float(np.median(np.abs(phi[:-1][pair] - gamma * phi[1:][pair])))
  kappa = cfg.target_share * task_per_step / max(f_unit, 1e-12)

  print(f"\nsamples {V.size} over {cfg.steps} steps x {cfg.num_envs} envs")
  print(f"consecutive-step pairs {int(pair.sum())}")
  print("\nV = e^T P e percentiles:")
  for name, value in pct.items():
    print(f"  {name:>4}  {value:12.3f}")
  print(f"  mean  {V.mean():12.3f}   max {V.max():12.3f}")
  print("\nper-step task reward, median |weighted term| (pre-dt):")
  for i, name in enumerate(term_names):
    print(f"  {name:26s} {np.median(np.abs(per_term[:, i])):8.4f}")
  print(f"  {'|total|':26s} {task_per_step:8.4f}")
  # clf_decrease_rbf scores the relative violation of V' <= (1 - alpha) V, so report
  # where that ratio sits and what the kernel pays at the configured sigma.
  prev, cur = vals[:-1][pair], vals[1:][pair]
  viol_rel = np.clip(cur - prev + cfg.alpha * prev, 0.0, None) / (prev + 1e-6)
  rbf = np.exp(-viol_rel / cfg.sigma**2)
  print(f"\nclf_decrease_rbf at alpha {cfg.alpha}, sigma {cfg.sigma}:")
  print(f"  satisfied V' <= (1-alpha)V: {float((viol_rel <= 0).mean()):.1%} of steps")
  print(
    f"  relative violation p50 {np.median(viol_rel):.4f}  "
    f"p95 {np.percentile(viol_rel, 95):.4f}"
  )
  print(f"  reward median {np.median(rbf):.4f}  mean {rbf.mean():.4f}  (in (0, 1])")

  print("\nsuggested constants:")
  print(f"  v_half = {v_half:.1f}   (median V; the potential is steepest around it)")
  print(
    f"  kappa  = {kappa:.1f}   (|F| ~ {cfg.target_share:.0%} of the task reward, "
    f"gamma {gamma})"
  )

  summary = {
    "checkpoint": cfg.checkpoint,
    "motion_file": cfg.motion_file,
    "tape_file": cfg.tape_file,
    "samples": int(V.size),
    "gamma": float(gamma),
    "V": {**pct, "mean": float(V.mean()), "max": float(V.max())},
    "task_reward_per_step_median": task_per_step,
    "per_term_median": {
      n: float(np.median(np.abs(per_term[:, i]))) for i, n in enumerate(term_names)
    },
    "suggested_v_half": v_half,
    "clf_violation_rel_p50": float(np.median(viol_rel)),
    "clf_violation_rel_p95": float(np.percentile(viol_rel, 95)),
    "clf_decrease_rbf_mean": float(rbf.mean()),
    "suggested_kappa": float(kappa),
  }
  if cfg.out:
    Path(cfg.out).write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {cfg.out}")


def main():
  run(tyro.cli(MeasureConfig, prog=sys.argv[0], config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
