![Project banner](https://raw.githubusercontent.com/mujocolab/mjlab/main/docs/source/_static/mjlab-banner.jpg)

# mjlab

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/mujocolab/mjlab/ci.yml?branch=main)](https://github.com/mujocolab/mjlab/actions/workflows/ci.yml?query=branch%3Amain)
[![Documentation](https://github.com/mujocolab/mjlab/actions/workflows/docs.yml/badge.svg)](https://mujocolab.github.io/mjlab/)
[![License](https://img.shields.io/github/license/mujocolab/mjlab)](https://github.com/mujocolab/mjlab/blob/main/LICENSE)
[![Nightly Benchmarks](https://img.shields.io/badge/Nightly-Benchmarks-blue)](https://mujocolab.github.io/mjlab/nightly/)
[![PyPI](https://img.shields.io/pypi/v/mjlab)](https://pypi.org/project/mjlab/)

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

# Sergio's Custom Command Hints

Before you run any commands, ensure you add your WandB API key (https://wandb.ai/authorize) and entity to your `~/.bashrc`:
```bash
export WANDB_API_KEY=<your-wandb-api-key>
export WANDB_ENTITY=<your-wandb-entity>
```

If you are training on a GPU cluster you can prepend the following traning commands with `CUDA_VISIBLE_DEVICES=#` where `#` is the GPU id you want to train on. You can also specify multiple GPU ids with `CUDA_VISIBLE_DEVICES=#,#`.

## Locmotion Velocity Tracking
I made a custom environment configuration and RL configuration for the flat velocity tracking task. You can train and evaluate with the following commands:
```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1-Custom --env.scene.num-envs 4096
```
and
```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1-Custom \
    --wandb-run-path sesteban-california-institute-of-technology-caltech/mjlab/bysdsnbu
```

## 23-DOF Variant (No Wrist / Waist Pitch & Roll)

A reduced 23-DOF G1 model is available that removes the waist pitch/roll and wrist pitch/yaw joints (keeps wrist roll). This is useful when you don't need upper-body dexterity and want a smaller action space.

The 23-DOF tasks mirror the full-DOF tasks with a `-23dof` suffix:

Train and evaluate velocity tracking:
```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1-Custom-23dof --env.scene.num-envs 4096
```
```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1-Custom-23dof \
    --wandb-run-path <your-wandb-entity>/mjlab/<run-id>
```

Train and evaluate motion imitation:
```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1-Custom-23dof \
    --registry-name=wandb-registry-Motions/walk1_subject1:latest \
    --env.scene.num-envs 4096
```
```bash
uv run play Mjlab-Tracking-Flat-Unitree-G1-Custom-23dof \
    --wandb-run-path <your-wandb-entity>/mjlab/<run-id>
```

All other available 23-DOF task IDs:
- `Mjlab-Velocity-Flat-Unitree-G1-23dof`
- `Mjlab-Tracking-Flat-Unitree-G1-23dof`

## Motion Imitation
The instructions are available at the following link:
https://mujocolab.github.io/mjlab/main/source/training/motion_imitation.html

### Overview
#### Create a Motion Registry in WandB
Make sure you create a motion registry in WandB to store your parsed motions. Name it `Motions` and set the artifact type to `All Types`.

#### Parse and Upload Motion Data
Your custom motions will be placed inside the `custom_motions`. You just need a `.csv` file containing the full configuration trajectory. Take care in noting you trajectories `fps`.

Here is an example of the command that you run from root to parse and upload the motion to WandB:
```bash
MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
    --input-file ./custom_motions/walk1_subject1.csv \
    --output-name walk1_subject1 \
    --input-fps 30 \
    --output-fps 50 \
    --render True
```
This parses the motion using mujoco joint indexing (rather than Isaac Lab breadth first ordering).

In your WandB, this should create a new project called `csv_to_npz` and a new artifact in the `Motions` registry called `walk1_subject1`. You can play the parsed video by going to WandB `csv_to_npz` > `walk1_subject1` > `Files` > `media` or by locally playing it in the root of the repo where it's called `motion.mp4`.

### Training and Evaluation
Once you have your motion parsed and uploaded to WandB, you can use it for training and evaluation. 

Train and evaluate a motion tracking:
```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1-Custom \
    --registry-name=wandb-registry-Motions/walk1_subject1:latest \
    --env.scene.num-envs 4096
```
and
```bash
uv run play Mjlab-Tracking-Flat-Unitree-G1-Custom \
    --wandb-run-path sesteban-california-institute-of-technology-caltech/mjlab/8ldzx1bk
```