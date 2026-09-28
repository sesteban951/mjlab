"""Policy-only ONNX export for the gait-library tracking envs."""

import json
from typing import cast

import wandb
from rsl_rl.env.vec_env import VecEnv

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.crawling_fwd.mdp.commands import LibraryMotionCommand


class LibraryTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  """Export the bare policy, not the library.

  ``MotionTrackingOnPolicyRunner`` bakes the whole motion reference into the ONNX as
  buffers and emits the reference frame as extra outputs -- BeyondMimic's deploy reads
  the clip out of the file. A library env's actor is REFERENCE-FREE (proprioception,
  phase and the commanded twist), so that bundle is dead weight at deployment: on the
  walk+jog library it is ``(456, 70, ...)`` arrays, ~55 MB against a few hundred KB of
  policy, and it is indexed on the clip axis besides. This runner exports like the
  velocity task does and attaches as METADATA the one thing a deploy target needs from
  the library: each clip's twist label and frame count (plus fps), so it can reproduce
  the nearest-clip snap and the per-clip phase clock. That is a few kilobytes of JSON.
  """

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    # train.py passes this to every tracking task; a library has no motion artifact.
    del registry_name

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name
        if self.logger.logger_type in ("wandb", "WandbLogWriter") and wandb.run
        else "local"
      )  # type: ignore[assignment]
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      cmd = cast(
        LibraryMotionCommand, self.env.unwrapped.command_manager.get_term("motion")
      )
      metadata.update(
        {
          "twist_command_range": json.dumps(cmd.cfg.twist_command_range),
          "twist_metric_weights": json.dumps(list(cmd.cfg.twist_metric_weights)),
          "rel_static_envs": float(cmd.cfg.rel_static_envs),
          "library_fps": float(1.0 / self.env.unwrapped.step_dt),
          # Per clip: [vx, vy, wz, n_frames]. The snap is argmin over the weighted
          # squared twist distance; the phase clock advances one frame per control step
          # and wraps at n_frames, so a deploy target needs nothing else from the
          # library.
          "library_clips": json.dumps(
            [
              [*map(float, tw), int(n)]
              for tw, n in zip(
                cmd.motion.lib_twists.tolist(),
                cmd.motion.n_frames.tolist(),
                strict=True,
              )
            ]
          ),
        }
      )
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if (
        self.logger.logger_type in ("wandb", "WandbLogWriter")
        and self.cfg["upload_model"]
      ):
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
