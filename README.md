![Project banner](https://raw.githubusercontent.com/mujocolab/mjlab/main/docs/source/_static/mjlab-banner.jpg)

# mjlab

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/mujocolab/mjlab/ci.yml?branch=main)](https://github.com/mujocolab/mjlab/actions/workflows/ci.yml?query=branch%3Amain)
[![Documentation](https://github.com/mujocolab/mjlab/actions/workflows/docs.yml/badge.svg)](https://mujocolab.github.io/mjlab/)
[![License](https://img.shields.io/github/license/mujocolab/mjlab)](https://github.com/mujocolab/mjlab/blob/main/LICENSE)
[![Nightly Benchmarks](https://img.shields.io/badge/Nightly-Benchmarks-blue)](https://mujocolab.github.io/mjlab/nightly/)
[![PyPI](https://img.shields.io/pypi/v/mjlab)](https://pypi.org/project/mjlab/)
[![PyPI downloads](https://img.shields.io/pypi/dm/mjlab?color=blue)](https://pypistats.org/packages/mjlab)

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s manager-based API with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), a GPU-accelerated version of [MuJoCo](https://github.com/google-deepmind/mujoco).
The framework provides composable building blocks for environment design,
with minimal dependencies and direct access to native MuJoCo data structures.

## Getting Started

mjlab requires an NVIDIA GPU for training. macOS is supported for evaluation only.

**Try it now:**

Run the demo (no installation needed):

```bash
uvx --from mjlab --refresh demo
```

Or try in [Google Colab](https://colab.research.google.com/github/mujocolab/mjlab/blob/main/notebooks/demo.ipynb) (no local setup required).

**Install from source:**

```bash
git clone https://github.com/mujocolab/mjlab.git && cd mjlab
uv run demo
```

For alternative installation methods (PyPI, Docker), see the [Installation Guide](https://mujocolab.github.io/mjlab/main/source/installation.html).

## Training Examples

### 1. Velocity Tracking

Train a Unitree G1 humanoid to follow velocity commands on flat terrain:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```

**Multi-GPU Training:** Scale to multiple GPUs using `--gpu-ids`:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids "[0, 1]" \
  --env.scene.num-envs 4096
```

See the [Distributed Training guide](https://mujocolab.github.io/mjlab/main/source/training/distributed_training.html) for details.

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 2. Motion Imitation

Train a humanoid to mimic reference motions. See the [motion imitation guide](https://mujocolab.github.io/mjlab/main/source/training/motion_imitation.html) for preprocessing setup.

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name your-org/motions/motion-name --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 3. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP before training:

```bash
uv run play Mjlab-Your-Task-Id --agent zero  # Sends zero actions
uv run play Mjlab-Your-Task-Id --agent random  # Sends uniform random actions
```

When running motion-tracking tasks, add `--registry-name your-org/motions/motion-name` to the command.


## Documentation

Full documentation is available at **[mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/)**.

## Development

```bash
make test          # Run all tests
make test-fast     # Skip slow tests
make format        # Format and lint
make docs          # Build docs locally
```

For development setup: `uvx pre-commit install`

## Citation

mjlab is used in published research and open-source robotics projects. See the [Research](https://mujocolab.github.io/mjlab/main/source/research.html) page for publications and projects, or share your own in [Show and Tell](https://github.com/mujocolab/mjlab/discussions/categories/show-and-tell).

If you use mjlab in your research, please consider citing:

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```

## License

mjlab is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

Some portions of mjlab are forked from external projects:

- **`src/mjlab/utils/lab_api/`** — Utilities forked from [NVIDIA Isaac
  Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause license, see file
  headers)

Forked components retain their original licenses. See file headers for details.

## Acknowledgments

mjlab wouldn't exist without the excellent work of the Isaac Lab team, whose API
design and abstractions mjlab builds upon.

Thanks to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for
answering our questions, giving helpful feedback, and implementing features
based on our requests countless times.

---

# Sergio's Custom Commmands
Some commands that are more direct to me. 

## Motion Tracking

### Parse a Motion
To parse a motion, your provide a csv file with a genearilzed configuration `qpos`, whose order is `[q_base, q_x, q_y, q_z, q_w, q_joint]`:

```bash
# Parse a motion from csv to npz format required for motion tracking.
uv run python -m mjlab.scripts.csv_to_npz \
  --input-file trajectories/srb_ik_jump_fwd/srb_ik_jump_fwd_29dof.csv \
  --output-name srb_ik_jump_fwd_29dof \
  --input-fps 50.0 \
  --output-fps 50.0 \
  --render True
```

where:
- `--input-file` is the path to your csv file.
- `--output-name` is the name of the output npz file.
- `--input-fps` is the fps of YOUR trajectory.
- `--output-fps` is the fps for training (the policy query frequency).
- `--render` produces an mp4 video when `True`.

### Train
To train with a already parsed motion file, you can use the following command:
```bash
# Train a tracking policy given a motion file in the registry.
uv run train G1-Tracking-Custom \
  --registry-name wandb-registry-motions/srb_ik_jump_fwd_29dof \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 30001 \
  --video True --video-iter-interval 500 \
  --env.viewer.width 1280 --env.viewer.height 720
```

where:
- `--registry-name` is the parsed motion file in the wandb registry.
- `--env.scene.num-envs` is the number of parallel envs for training.
- `--agent.max-iterations` sets how many PPO iterations to train for (overrides the task default). Use a trailing `+1` (e.g. `30001`) so the final iteration lands on a `--agent.save-interval` boundary and the last checkpoint/ONNX is written at a round number (`model_30000.pt`); plain `30000` would stop at `model_29500.pt`.
- `--video` records checkpoint videos of training progress when `True`.
- `--video-iter-interval` is how often (in PPO iterations) a video clip is recorded.

### Play
To play a trained motion tracking policy, run the following command:

```bash
# Play a trained motion tracking policy.
uv run play G1-Tracking-Custom \
  --wandb-run-path <wandb-run-path> \
  --num-envs 10
```

where:
- `--wandb-run-path` is the path to your trained model in Weights & Biases (e.g. `<entity>/mjlab/<run-id>`); fetches the latest checkpoint and the motion artifact used in that run.
- `--num-envs` is the number of robots to view at once.

Unlike velocity tasks, tracking needs a reference motion. `--wandb-run-path` resolves it automatically from the run. If you instead load a local checkpoint with `--checkpoint-file`, you must also pass `--motion-file <path>.npz`.


## Contact-Rich Motion Tracking
`G1-Tracking-ContactRich-Custom` is a tracking variant for highly contact-rich
motions such as crawling, where many body links (not just the feet) contact the
ground. Every body collision geom is made frictional (condim=3, priority=1) and
its tangential friction is randomized, unlike the foot-only friction of the base
tracking env.

### Train

```bash
# Train a contact-rich motion tracking policy (e.g. crawling).
uv run train G1-Tracking-ContactRich-Custom \
  --registry-name wandb-registry-motions/crawl_ff_loop_180_R_001__A229_29dof \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 30001 \
  --video True --video-iter-interval 500 \
  --env.viewer.width 1280 --env.viewer.height 720
```

Play it with `uv run play G1-Tracking-ContactRich-Custom` using the same flags as
the motion tracking Play command above.


## Twist Command Tracking
### Train

```bash
# Train a velocity command tracking policy.
uv run train G1-Velocity-Custom \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 20001 \
  --video True --video-iter-interval 500 \
  --env.viewer.width 1280 --env.viewer.height 720
```

where:
- `--env.scene.num-envs` is the number of parallel envs for training.
- `--agent.max-iterations` sets how many PPO iterations to train for (overrides the task default). Use a trailing `+1` (e.g. `20001`) so the final iteration lands on a `--agent.save-interval` boundary and the last checkpoint/ONNX is written at a round number (`model_20000.pt`); plain `20000` would stop at `model_19900.pt`.
- `--video` records checkpoint videos of training progress when `True`.
- `--video-iter-interval` is how often (in PPO iterations) a video clip is recorded.
- `--env.viewer.width` / `--env.viewer.height` set the recorded video resolution (here 720p).

### Play
To play a trained velocity command tracking policy, run the following command:

```bash
# Play a trained velocity command tracking policy.
uv run play G1-Velocity-Custom \
  --wandb-run-path <wandb-run-path> \
  --num-envs 10
```

where:
- `--wandb-run-path` is the path to your trained model in Weights & Biases (e.g. `<entity>/mjlab/<run-id>`); fetches the latest checkpoint of that run.
- `--num-envs` is the number of robots to view at once.

## Other Flags
These options apply to both the motion tracking and velocity playback commands above:
- `--checkpoint-file` loads a local checkpoint instead of W&B (e.g. `logs/rsl_rl/<experiment>/<run>/model_500.pt`).
- `--wandb-checkpoint-name` pins a specific checkpoint within the run (e.g. `model_4000.pt`).
- `--agent` selects the policy: `trained` (default), or `zero` / `random` for a no-checkpoint sanity check.
- `--num-envs` sets how many robots to view at once.
- `--viewer` chooses the viewer: `auto` (default), `native`, or `viser`.
- `--no-terminations` disables episode resets (keeps the robot going for viewing).
- `--video` / `--video-length` / `--video-width` / `--video-height` record the playback to an mp4.
- `--camera` selects a camera by index or name.
- `--device` overrides the compute device (e.g. `cpu`, `cuda:0`).

## Contact-Rich Locomotion
Contact-rich locomotion (e.g. crawling) tracks a *library* of periodic clips — one per forward speed — instead of a single motion, and a commanded speed selects the nearest clip at runtime (see `Mjlab-Crawling-NoStopping-Flat-Unitree-G1`).

### Parse the Gait Library
The library ships as raw solver output: one `.npz` per speed storing per-frame full MuJoCo state `[qpos | qvel]`, named so the speed is parsable from the filename (`crawl_vp0_280` → +0.28 m/s). Convert the whole folder into the tracking-format npz the crawling task loads:

```bash
# Convert every clip in a library folder to tracking-format npz (local files, no wandb).
uv run python -m mjlab.scripts.library_to_npz \
  --input-dir trajectories/crawl_ff_loop_180_R_001__A229_library \
  --output-dir trajectories/crawl_ff_loop_180_R_001__A229_tracking \
  --input-fps 200.0 \
  --output-fps 50.0
```

where:
- `--input-dir` is the folder of raw library clips (`[qpos | qvel]` per frame).
- `--output-dir` is where the converted per-body world-kinematics clips are written, one per input clip (basename preserved).
- `--input-fps` is the fps of YOUR library clips.
- `--output-fps` is the fps for training; it MUST equal the env control rate `1/step_dt` (50 Hz).

The `--input-dir`/`--output-dir` defaults already point at the shipped library, so you can run it with no flags to regenerate the tracking clips.