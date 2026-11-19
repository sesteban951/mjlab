# G1-JogRunSprint-Custom

One reference-free PPO policy for fast locomotion on the Unitree G1. It covers jog, run and sprint, plus the jog library's backward arcs, fast in-place pivots and standing idle. It is built the same way as `G1-WalkJog-Unicycle`: a merged "ragged" library, where every clip loops at its own stride length, and the jog task's environment pointed at it.

This is the **minimum version** of a larger plan. The plan comes from a literature review and from reading the released code of the most relevant papers, both in [`research/`](research/). Below: what is built, the full plan, and which parts are still open.

## Train and play

```sh
# Build the libraries, in this order (only needed when the mj-nlp grids change)
uv run python -m mjlab.scripts.build_library jog_unicycle
uv run python -m mjlab.scripts.build_library run_fwd
uv run python -m mjlab.scripts.build_library sprint_fwd
uv run python -m mjlab.scripts.build_library jogrunsprint

# Train
uv run train G1-JogRunSprint-Custom --env.scene.num-envs 4096 \
  --agent.run-name jogrunsprint_v0 --agent.max-iterations 30001 \
  --video True --video-iter-interval 500

# Play (library tasks need --motion-file pointed at the library)
uv run play G1-JogRunSprint-Custom --wandb-run-path <entity/project/run_id> \
  --motion-file trajectories/g1/library/jogrunsprint_tracking
```

Play pins a straight 2.0 m/s run. To see the whole command range, use the viewer's twist controls.

## The library

| family | source | vx [m/s] | wz [rad/s] | stride | frames at 50 Hz | clips |
|---|---|---|---|---|---|---|
| jog forward | `jog_unicycle` | 0.5 – 1.4 | ±0.5 | 0.85 s | 43 | 110 |
| jog backward | `jog_unicycle` | −1.0 – −0.5 | ±0.5 | 0.85 s | 43 | 66 |
| fast pivots | `jog_unicycle` | 0 | ±1.0 – ±2.0 | 0.85 s | 43 | 42 |
| idle | `jog_unicycle` | 0 | 0 | — | 43 | 1 |
| run | `run_fwd` | 1.5 – 2.4 | ±0.5 | 0.74 s | 37 | 110 |
| sprint | `sprint_fwd` | 2.5 – 3.5 | ±0.5 | 0.63 s | 32 | 121 |

The source grids (`jog_fwd`, `run_fwd`, `sprint_fwd`) each share a vx column with their neighbour, at 1.5 and 2.5 m/s. The merge uses half-open bands, so each shared column goes to the faster family. The spec is `LIBRARY_SPECS["jogrunsprint"]` in `crawling_common/library.py`. The sprint stride is 31.5 frames at 50 Hz, so the converter stretches it onto 32 frames, and sprint clips play about 1.6% slower than their labels.

## What the minimum version does

1. **One foot phase for all three families.** jog, run and sprint used to be built as separate libraries, each phase-aligned to its own reference clip. Their strides therefore started on different feet. Measured as the fraction of the stride where each foot's swing peaks, on the straight clips:

   | | jog | run | sprint |
   |---|---|---|---|
   | before, left / right | 0.28 / 0.78 | 0.65 / 0.16 | 0.80 / 0.32 |
   | after, left / right | 0.28 / 0.78 | 0.27 / 0.77 | 0.25–0.31 / 0.78–0.81 |

   Without this, the fraction-of-stride remap at every jog↔run or run↔sprint switch would ask the robot to swing its stance foot. `library_to_npz` now takes `phase_ref`, a built clip to align to: the jog library's own reference, `jog_fwd_vx+1000_vy+000_wz+000`. It compares foot-height profiles on a common phase grid, so clips of any length can be aligned. `LibrarySpec.phase_ref` passes it through `build_library`. `tests/test_jogrunsprint.py` checks the alignment.

2. **Merged library and task.** `config/g1/env_cfgs.py` reuses `G1-Jog-Unicycle` unchanged except for:
   - forward vx 0.5–3.5 m/s;
   - backward arcs 20% of arcs instead of 40%, since backward clips only exist at jog speed;
   - twist-reward width 0.9.

   The width comes from each clip's own twist scatter about its mean: jog forward 0.32, jog backward 0.54, run 0.59, pivots 0.90, sprint 1.03. At 0.9, a perfectly tracking clip scores 0.88 (jog), 0.65 (run), 0.27 (sprint) and 0.37 (pivots), about the balance the jog task chose. Everything else is inherited: blended switches over 0.4 s, 3–8 s command resampling, 15% standing, 30% pivots, reward-free actor, rewards, randomization and actuator delay.

   Measured command mix: jog 14%, run 15%, sprint 15%, backward 11%, idle 15%, pivots 30%.

## The full plan, and what is done

Smooth transitions are the priority. The main finding of the review: every gait-library controller that changes gait smoothly fixes the *reference* first. They all do four things:
- give every gait one phase origin tied to a named foot event;
- switch at contact events;
- keep phase continuous and change only its rate;
- interpolate or inertialize instead of cross-fading misaligned poses.

See `research/literature_review.md`, sections 1–2.

| # | Step | Status | Source / precedent |
|---|---|---|---|
| 1 | Shared foot-phase origin across families | **done** (run/sprint aligned to jog) | HZD libraries, Walk These Ways (WTW) per-foot offsets, unitree_rl_gym `phase + 0.5` |
| 2 | Merged library + single task | **done** | `G1-WalkJog-Unicycle` pattern |
| 3 | Offline reference check at switches, plus a joint-speed headroom check | not done | review §"transition benchmark" |
| 4 | Reward-gated (vx, wz) grid curriculum, seeded at jog | not done | WTW / Rapid Locomotion `RewardThresholdCurriculum` |
| 5 | Transition-heavy commands (bias resamples across the 1.5 and 2.5 m/s boundaries, both directions) | not done | RPG: 5–65% → 70–95% transition success |
| 6 | Integrated phase `φ += dt/T` with a ramped period, and `1/T` in the actor observation | not done | WTW, HugWBC `gait_indices` |
| 7 | Inertialized switch (decaying offset) instead of the 0.4 s linear cross-fade | not done | Holden `spring.h`, half-life 0.1–0.2 s |
| 8 | Switch only at left touchdown, with hysteresis | not done | HZD libraries; no RL code does this |
| 9 | Phase-registered interpolation between neighbouring clips | not done | Gong / Da gait-library interpolation |
| 10 | Dedicated transition clips from trajectory optimization (pivots first) | not done | Da, Murali & Ames |
| 11 | Torque–speed derating (`DcMotorActuatorCfg`) for the sprint band | not done | Chasing Autonomy, Booster |
| 12 | Mixture-of-experts actor, or specialists distilled into one policy | out of scope (PPO only) | MoE-Loco, CoRe-MoE, GMT |

### Why only steps 1–2 now

`G1-WalkJog-Unicycle` already trains one good policy across a larger stride jump (70 → 43 frames) with the current blending. So the period difference here (43 → 37 → 32) is not expected to break training. The wrong-foot remap would have, and step 1 fixes it. Everything else waits for evidence from a trained policy.

### What to watch, and what to add next

- **Collapse to standing, or no progress at run/sprint speeds:** add **step 4**, the curriculum. Uniform sampling over a wide speed range caused exactly this "jitter in place" failure in Rapid Locomotion, and in this repo's own velocity tasks.
- **Falls or tracking spikes right after band switches** (check with play, commanding 1.2↔1.8 and 2.2↔2.8 m/s): add **step 5**. Then add **steps 6–7**; both are local to `crawling_fwd_blending/mdp/commands.py` and the phase observation.
- **Rough jog↔pivot switches:** the pivots' feet are not half a stride apart (left/right swing peaks at 0.42/0.74 for left turns and 0.25/0.92 for right turns), and left and right turns are not mirror images. No roll lines them up exactly with the jog. Consider **step 10**, or regenerating or mirroring the pivots.
- **Good in simulation but too fast to be believable:** the G1 actuators clip torque at a flat limit, with no torque–speed derating, and the hip and knee motors' ~20 rad/s limits are never enforced. See **step 11** before trusting sprint results on hardware.

Two implementation notes for later steps:
- **Adaptive sampling does almost nothing on this library.** It uses `bin_count = time_step_total // 50 + 1`, which is 2 bins for the whole library. The curriculum in step 4 should replace it as the failure-weighted sampler.
- **Phase rounding at switches.** `_phase_remap` rounds a switch to the nearest frame, about 3% of a sprint stride. Step 6 removes the rounding.

## Expectations

The best documented G1 result for this approach is Caltech's 3.3 m/s, on a treadmill, from a 50 Hz trajectory-optimized gait library (preprint, code at [Zolkin1/robot_rl](https://github.com/Zolkin1/robot_rl)). A smooth policy up to about 3 m/s is a realistic first milestone, and 3.5 m/s is a stretch. The riskiest transitions are sprint→run (slowing down is harder than speeding up in every study that measured both), sprint while turning, and getting into and out of pivots.

## Research files

- `research/literature_review.md`: the full review, with evidence tagged peer-reviewed / preprint / company claim / news.
- `research/notes/`: the five source notes:
  - phase clocks: `phase_clocks_multigait.md`;
  - learning from a motion library and transitions: `motion_library_imitation.md`;
  - gait-library transitions: `gait_library_transitions.md`;
  - high-speed running: `high_speed_running.md`;
  - combining skills: `consolidation_distillation.md`.
- The released code read for steps 4–7, pinned to the commits read:
  - WTW [`legged_robot.py` gait clock](https://github.com/Improbable-AI/walk-these-ways/blob/0e7236bdc81ce855cbe3d70345a7899452bdeb1c/go1_gym/envs/base/legged_robot.py#L826-L861) and [`curriculum.py`](https://github.com/Improbable-AI/walk-these-ways/blob/0e7236bdc81ce855cbe3d70345a7899452bdeb1c/go1_gym/envs/base/curriculum.py#L113-L159);
  - [Rapid Locomotion curriculum](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/f5143ef940e934849c00284e34caf164d6ce7b6e/mini_gym/envs/base/legged_robot.py#L573-L604);
  - [HugWBC gait clock](https://github.com/InternRobotics/HugWBC/blob/eb831af76ea70ebe5d435786b3bbebee13c75bbd/legged_gym/envs/h1/h1.py#L766-L810);
  - Holden's [inertialization `spring.h`](https://github.com/orangeduck/Motion-Matching/blob/57b7250e0d34a4e456a34d47e24c2f05fdcc711e/spring.h#L115-L187).
