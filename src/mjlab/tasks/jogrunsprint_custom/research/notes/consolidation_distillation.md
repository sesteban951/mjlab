# Consolidating Multiple Locomotion Skills into One Deployable Policy

Scope: (a) one PPO policy trained on a merged gait library, (b) per-gait specialists distilled into one student, (c) MoE / gated networks, (d) hybrids. The user's setup: G1 in mjlab, rsl-rl PPO at 50 Hz, a reference-free actor (proprio + twist + phase clock), and a reference-aware critic and rewards. Gaits: jog 0.5-1.5, run 1.5-2.5, sprint 2.5-3.5 m/s, plus turns at |wz| 1-2 rad/s.

Note on verification: items marked "(abstract/primary, from prior knowledge)" cite the canonical arXiv page, but I did not re-fetch them in this session. All numeric tables were fetched from the papers' full text in this session.

## Q1. Policy distillation (student-teacher, multi-teacher to one student): what works, how much data, failure modes

### Takeaway
Multi-specialist to one-student DAgger distillation is a proven, hardware-validated recipe (Robot Parkour Learning, Barkour, BumbleBee, GMT student). The usual result: the student comes within a few points of the specialists on most skills, falls clearly short on one or two, and gives smoother transitions. Two things make it work: the student must be recurrent or history-based, and a specialist must label the states the student itself visits (on-policy DAgger, not offline BC). For highly dynamic humanoid skills, pure DAgger is reported as insufficient, and an RL term is added.

### Cited Findings
- **Robot Parkour Learning (Zhuang et al., CoRL 2023), A1 quadruped.**
  - Setup: five specialists (climb, leap, crawl, tilt, run) are trained with privileged physics and vision. They are distilled with DAgger into one depth-vision student: a CNN feeding a 1-layer, 256-unit GRU, then an MLP [512,256,128]. The student rolls out, and the specialist matching the current obstacle type labels its actions. Because outputs are tanh-bounded, the loss is BCE-style. — [arXiv 2309.05665](https://ar5iv.labs.arxiv.org/html/2309.05665)
  - Sim success (%, climb/leap/crawl/tilt/run): distilled student 86/80/100/73/100 vs privileged specialists 95/82/100/100/100. — [arXiv 2309.05665](https://ar5iv.labs.arxiv.org/html/2309.05665)
  - Ablations: an MLP student (no memory) scores 0/1/63/43/100. "No distill" (direct vision PPO) scores 0/0/73/0/100. RMA can't merge specialists with different parameters (tilt only, 74%). There is no BC-vs-DAgger ablation, and DAgger iteration and data counts are not reported. — [arXiv 2309.05665](https://ar5iv.labs.arxiv.org/html/2309.05665)
  - Hardware (A1): about 80% climb (0.40 m), 80% leap (0.60 m gap), 90% crawl (0.2 m). — [arXiv 2309.05665](https://ar5iv.labs.arxiv.org/html/2309.05665)
- **Barkour (Caluwaerts et al., 2023), Google quadruped, hardware.**
  - Setup: specialists (walk, slope climb, jump, ...) are trained with RL. Their rollouts become state-action datasets that are distilled offline into one "Locomotion-Transformer." This is compared against hand-designed switching between the specialists by robot position. — [arXiv 2305.14654](https://arxiv.org/abs/2305.14654v1); [Google Research blog](https://research.google/blog/barkour-benchmarking-animal-level-agility-with-quadruped-robots/)
  - Result: the generalist scores slightly lower on the Barkour benchmark than the switching approach, but transitions between gaits more smoothly. — [arXiv 2305.14654](https://arxiv.org/abs/2305.14654v1)
- **BumbleBee / "From Experts to a Generalist" (NeurIPS 2025), humanoid whole-body tracking.**
  - Pipeline: 8,179 motions are clustered with an autoencoder into K=6 clusters: Jump 351, Walk-slow 3,355, Walk-fast 307, Stand-up 2,378, Stand-mid 578, Stand-low 229. A general policy is trained, then fine-tuned per cluster into experts. The experts are refined with real-data delta-action models and distilled by DAgger with a KL(student‖cluster expert) loss. — [arXiv 2506.12779](https://arxiv.org/html/2506.12779v3)
  - Student: a Gated Transformer-XL (1 block, 6 heads, hidden 128, memory 10). Teachers: MLPs [1024,1024,512]. Clusters are sampled equally (1/6 each) during distillation. Adam, LR 1e-4. — [arXiv 2506.12779](https://arxiv.org/html/2506.12779v3)
  - Success rate in MuJoCo (sim-to-sim proxy): single general policy 33.01%, experts on a random partition 35.36%, BumbleBee distilled generalist 66.84%. IsaacGym numbers are nearly flat across methods (88.69 / 86.25 / 89.58%). — [arXiv 2506.12779](https://arxiv.org/html/2506.12779v3)
  - Student vs experts: the student loses to specialists on Stand-mid and Walk-fast, and beats them on Jump and Walk-slow (authors: it inherits stable behavior from several experts). Real-robot results are qualitative only. — [arXiv 2506.12779](https://arxiv.org/html/2506.12779v3)
- **GMT (Chen et al., 2025), Unitree G1 general motion tracking, 50 Hz.**
  - Setup: a privileged PPO teacher (with MoE and adaptive sampling) is distilled into a deployable student with DAgger and an ℓ2 action loss. The student sees proprio plus about 20 steps of history plus motion targets. — [arXiv 2506.14770](https://arxiv.org/html/2506.14770v2)
  - Student vs teacher, MPKPE in mm: AMASS 42.01 vs 42.07, LAFAN 46.14 vs 45.16, so the student is essentially lossless.
  - Training budget: about 6.8B samples per policy, 4,096 envs. Teacher about 3 days and student about 1 day on one RTX 4090.
  - Hardware results are qualitative only. — [arXiv 2506.14770](https://arxiv.org/html/2506.14770v2)
- **Perceptive Humanoid Parkour (2026)** distills motion-tracking experts into one depth student. It reports that pure DAgger is insufficient for highly dynamic humanoid skills, so it adds RL to the distillation. — [arXiv 2602.15827](https://arxiv.org/html/2602.15827v1)
- **"Parkour in the wild" (2025)** trains terrain-specific experts, distills them with DAgger into a unified policy, then fine-tunes that policy with RL on new terrains. — [arXiv 2505.11164](https://arxiv.org/html/2505.11164v1)
- **Foundations (abstract/primary, from prior knowledge).**
  - Rusu et al., "Policy Distillation" (2015): distills multiple Atari DQN teachers into one multi-task student via a KL loss on temperature-softened Q outputs. — [arXiv 1511.06295](https://arxiv.org/abs/1511.06295)
  - Lee et al., ANYmal teacher-student (Sci. Robot. 2020): a privileged teacher is distilled into a proprioceptive TCN student that observes history. — [arXiv 2010.11251](https://arxiv.org/abs/2010.11251)
  - RMA (Kumar et al., 2021): instead of copying actions, a student adaptation module regresses the teacher's latent extrinsics from history. — [arXiv 2107.04034](https://arxiv.org/abs/2107.04034)
- **Tooling.** The rsl-rl distillation module ships a `Distillation` algorithm with an "mse" or "huber" student loss. Policies are `StudentTeacher` and `StudentTeacherRecurrent` (rnn_type "lstm"/"gru", optional recurrent teacher). In rsl-rl 4.0+, these are configured through `RslRlMLPModelCfg` / `RslRlRNNModelCfg` model entries. — [IsaacLab distillation_cfg source](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab_rl/rsl_rl/distillation_cfg.html)
- **Offline diffusion distillation.**
  - DiffuseLoco (Huang et al., CoRL 2024) learns one diffusion policy from an offline multimodal dataset of locomotion skills with no skill labels: walking at three speeds plus turning left and right. It switches smoothly between skills and runs on edge hardware with zero-shot real quadruped transfer.
  - It reports better stability and velocity tracking than RL and non-diffusion BC baselines. The abstract gives no numbers. — [arXiv 2404.19264](https://arxiv.org/abs/2404.19264); [CoRL PMLR](https://proceedings.mlr.press/v270/huang25a.html)

### Inferences
- **Mapping to the user's setup.** Training jog, run, sprint and turn specialists, then DAgger-distilling them, is well supported. Labeling is simple because the commanded twist determines which specialist is "correct": select the teacher by the command bin, as Parkour selects by obstacle type and BumbleBee by cluster.
- **Teacher observations.** The teachers can be the user's existing reference-aware policies (privileged critic and obs). The student keeps the deployable obs set.
- **Mode averaging at bin boundaries.** Near boundaries (vx about 1.5 or 2.5 m/s), the two adjacent teachers use different stride periods. An ℓ2/MSE student given ambiguous inputs will average them. Mitigations:
  - Make the label assignment deterministic from the command, so it is never ambiguous for a given input.
  - Overlap the teachers' training ranges, so that both teachers are competent at the boundary and one is chosen consistently.
  - Feed the stride period or phase clock to the student, so the input disambiguates the mode.
  - Since the user's actor already has a phase clock with a different period per gait, the clock disambiguates the gait and the averaging risk is reduced.
- **Distillation alone may not be enough.** Expect a gap of a few points to about 25 points on the hardest skill (Parkour tilt 73 vs 100; BumbleBee Walk-fast). Budget for an RL fine-tune of the student, as in Parkour-in-the-wild and Perceptive Humanoid Parkour. This matters most for sprint, the most dynamic skill.

### Gaps
- I found no paper that reports DAgger iteration counts or data volume needed for locomotion multi-teacher distillation; Parkour and BumbleBee omit them. GMT gives only wall-clock: about 1 day for the student vs 3 for the teacher.
- I found no quantitative BC-vs-DAgger or MSE-vs-KL comparison for multi-gait locomotion.
- I did not retrieve DiffuseLoco's numeric baseline comparison (MLP-BC mode-collapse numbers).
- Not covered in this session: "Puppeteer", OmniH2O's teacher-student numbers, and the PHC/PULSE distillation details.

## Q2. Mixture-of-experts / gated architectures for multi-gait locomotion: does gating help vs a monolithic MLP?

### Takeaway
In the 2025-2026 papers I found, a soft MoE (gating weights times expert actions, summed) beats a parameter-matched MLP when tasks are behaviorally distinct: quadruped vs biped gaits, walk vs run, stairs vs flat. The gains are large on the hard or minority tasks and small or slightly negative on the easy ones. On motion-tracking libraries (GMT), the MoE gain is small on average and concentrated on the hardest motions. Gating directly on commanded velocity is used successfully for walk/run on a G1 (CoRe-MoE).

### Cited Findings
- **MoE-Loco (Huang et al., 2025), Unitree Go2.**
  - Architecture: 6 experts (MLP [256,128,128] each) and a gating MLP [128] driven by the LSTM hidden state. MoE is in both actor and critic with a shared gate. Trained with PPO, 4,096 envs, IsaacGym. Commands are joystick velocity plus a one-hot gait flag (quadruped or biped). No load-balancing loss is mentioned. — [arXiv 2503.08564](https://arxiv.org/html/2503.08564v2)
  - Sim success (MoE / parameter-matched no-MoE):

    | Task | MoE | no-MoE |
    |---|---|---|
    | Mixed course | 0.879 | 0.571 |
    | Baffle (q) | 0.924 | 0.264 |
    | Biped slope | 0.961 | 0.504 |
    | Biped stair | 0.964 | 0.453 |
    | Biped walk | 0.932 | 0.826 |
    | Quad slope | 0.956 | 0.988 (only case where MoE loses) |

    — [arXiv 2503.08564](https://arxiv.org/html/2503.08564v2)
  - Gradient conflict: MoE raises the cosine similarity between task gradients. For example, Bar vs biped Slope-up goes from −0.132 to 0.278, and Slope-up vs Slope-down from 0.709 to 0.806. Not every pair improves: Baffle vs Stair goes 0.384 to 0.369. — [arXiv 2503.08564](https://arxiv.org/html/2503.08564v2)
  - Experts specialize, with gating weights clustering by task. A new skill can be added by adding one expert, freezing the others, and training only the gate (qualitative). Real-robot zero-shot results are in figures only. — [arXiv 2503.08564](https://arxiv.org/html/2503.08564v2); [project page](https://moe-loco.github.io)
- **CoRe-MoE (2026), Unitree G1, walk plus run up to 2.5 m/s outdoors.**
  - Stage 1: two AMP experts (walk, run), with a gate whose input is the commanded velocity. Commands are 0-1 m/s near the target and 1-2.5 m/s far from it.
  - Stage 2: an 8-expert terrain residual with contrastive (SwAV) gating. — [arXiv 2606.04718](https://arxiv.org/html/2606.04718)
  - Interference evidence: one network trained on merged walk+run data ("Mixed AMP") gives an intermediate "shuffling" gait. A single network with velocity-selected references ("Sel AMP") over-strides at high speed, with only 54.20% success vs 99.13% for CoRe-MoE. Both are qualitative or partial evidence; interference is not measured directly. — [arXiv 2606.04718](https://arxiv.org/html/2606.04718)
  - "w/o MoE" (single actor-critic) scores 97.06% flat success and 0.817 tracking, vs 99.13% and 0.853 for CoRe-MoE. On stairs, w/o MoE collapses to 35.6% vs 94.3%. Hardware results are qualitative. — [arXiv 2606.04718](https://arxiv.org/html/2606.04718)
- **GMT (2025), G1.**
  - Soft "Motion MoE": experts and gate all take robot state plus motion targets, and a = Σ pᵢaᵢ. The expert count is not stated in the text. — [arXiv 2506.14770](https://arxiv.org/html/2506.14770v2)
  - Ablation (teacher MPKPE in mm, AMASS / LAFAN): full 42.07 / 45.16, without MoE 42.53 / 48.26, without adaptive sampling 43.54 / 49.61, without both 44.34 / 52.34. Gains concentrate in the top 5-10% hardest motions. — [arXiv 2506.14770](https://arxiv.org/html/2506.14770v2)
  - The follow-up EGM uses a "Composite Decoupled MoE" with separate upper- and lower-body experts. — [arXiv 2512.19043](https://arxiv.org/html/2512.19043v1)
- **Multi-Expert Learning Architecture, MELA (Yang et al., Sci. Robot. 2020), real quadruped.**
  - Separately pretrained expert networks are blended by a gating network, which continuously synthesizes a new network's weights at runtime (weight blending, not output blending).
  - It produces coherent trotting, steering and fall recovery on hardware, plus transitional skills between modes. — [arXiv 2012.05810](https://arxiv.org/pdf/2012.05810)
- **Reske et al. (2021), MPC-Net with a mixture-of-experts network, quadruped hardware.** Performance improves when each expert specializes in one gait (hybrid-system mode). The authors add losses that encourage expert selection, and one learned policy replaces the MPC teacher for multiple gaits. I did not retrieve the numbers. — [arXiv 2103.14331](https://arxiv.org/abs/2103.14331v1)
- **Animation precedents.**
  - MANN (Zhang et al., SIGGRAPH 2018): a gating network blends expert weight sets into the motion-prediction network, so each expert specializes in particular movements across quadruped gaits. — [ACM DOI 10.1145/3197517.3201366](https://www.springerprofessional.de/doi/10.1145/3197517.3201366)
  - PFNN (Holden et al., SIGGRAPH 2017): the network weights are a cyclic function of the gait phase (abstract/primary, from prior knowledge). — [PFNN paper](https://theorangeduck.com/media/uploads/other_stuff/phasefunction.pdf)
- **MCP (Peng et al., NeurIPS 2019)** composes Gaussian primitives multiplicatively, with state- and goal-dependent gate weights (abstract/primary, from prior knowledge). — [arXiv 1905.09808](https://arxiv.org/abs/1905.09808)
- **PHC (Luo et al., ICCV 2023).** Progressive multiplicative control (PMCP) allocates new network capacity, adding primitives for harder motion sequences. This lets one controller scale to about 10k clips without catastrophic forgetting. — [arXiv 2305.06456](https://arxiv.org/abs/2305.06456)

### Inferences
- Soft MoE is cheap to add in rsl-rl: it is just a different actor module. Gate on (commanded twist, phase clock, proprio) or on the RNN hidden state.
- This is the lowest-effort change most likely to remove a jog/sprint "averaged gait" in a single PPO run. The evidence is MoE-Loco's gradient-cosine data and CoRe-MoE's shuffling gait when walk+run were merged into one network.
- A gate on commanded vx alone is essentially a learned, soft version of specialist switching. It can be initialized from pretrained specialists, as in MELA and CoRe-MoE Stage 1.
- Expected cost: little or no gain on whichever gait dominates the sampling, as with MoE-Loco quad slope and GMT's small average gain.
- The user's phase clock already acts like a PFNN-style phase input. With different stride periods per gait, a gate on the clock frequency is a natural, interpretable routing signal.

### Gaps
- No found study sweeps expert count, or compares hard vs soft gating, for locomotion speed ranges specifically.
- No found paper compares MoE against simply widening the MLP for a humanoid jog-to-sprint range.
- MANN's exact expert count and gating inputs could not be verified this session. (My recollection is foot velocities, action variables and target velocity, with 4 or 8 experts; treat as unverified.)

## Q3. Multi-task balancing within one PPO run: sampling ratios, normalization, multi-critic, PCGrad, adaptive sampling, curricula, slow/fast interference

### Takeaway
The best-evidenced balancing mechanisms for locomotion and tracking are:
- failure- or error-driven adaptive sampling (GMT; PHC's hard-motion focus);
- grid or local-expansion command curricula that grow only from mastered commands (Rapid Locomotion);
- reward routing by gait ID (gait-conditioned RL);
- equal per-cluster sampling during distillation (BumbleBee).

Uniform sampling over a wide speed range fails outright: the robot jitters in place, the same symptom the user saw. Multi-critic and gradient-surgery methods exist, but I found little locomotion-specific evidence for them on gait or speed tasks.

### Cited Findings
- **Rapid Locomotion (Margolis et al., RSS 2022), Mini Cheetah at 3.9 m/s on hardware.**
  - Failure without curriculum: uniform sampling over vx in [−4, 4] and ωz in [−5, 5] failed to learn. Most commands are high-speed, so reward is sparse, and the no-curriculum policy "jitters in place" with tracking error equal to the command. — [arXiv 2205.02824](https://ar5iv.arxiv.org/html/2205.02824)
  - Grid Adaptive curriculum: a joint (vx, ωz) grid at 0.5 m/s × 0.5 rad/s resolution, initialized uniform on [−1, 1]². Commands are sampled per episode from the grid.
  - Expansion rule: at episode end, if both vx and ωz tracking rewards exceed a threshold γ (value not given) and the command is among the hardest in the distribution, its 4-connected neighbors are set to probability 1. Probabilities are never removed. The joint grid respects centrifugal coupling between linear and angular speed.
  - Architecture: MLP [512,256,128] plus a 15-step history adaptation module. The paper does not analyze forgetting of low-speed commands. — [arXiv 2205.02824](https://ar5iv.arxiv.org/html/2205.02824)
- **GMT adaptive sampling**, for imbalanced difficulty in a library.
  - Long clips are split into segments of at most 10 s. Each motion has a completion level cᵢ that starts at 10 and is multiplied by 0.99 per completion, down to 1.
  - Termination threshold: Eᵢ = 0.25·exp((cᵢ−1)/9·log(0.6/0.25)).
  - Sampling weight: if cᵢ > 1, sᵢ = cᵢ; else sᵢ = (min(E_maxkeybody/0.15, 1))⁵. Probabilities are proportional to sᵢ.
  - Without it, hard segments are jittery and undeployable. The ablation shows MPKPE rising 45.16 to 49.61 on LAFAN. — [arXiv 2506.14770](https://arxiv.org/html/2506.14770v2)
- **Gait-Conditioned RL (Peng, Bao, Zhou, 2025), G1.**
  - One recurrent policy covers stand, walk and run plus transitions. A one-hot gait ID routes which reward terms are active, "to mitigate reward interference", and a multi-phase curriculum expands gait complexity and command range.
  - Running and transitions are shown only in sim. Hardware covers stand, walk and walk-to-stand. The abstract gives no ablation numbers. — [arXiv 2505.20619](https://arxiv.org/abs/2505.20619v3)
- **Interference evidence between gaits.**
  - BumbleBee: jumping (high torque) and in-place balancing conflict. A shared delta model degrades Jump from 59.64% to 50.71%, while cluster-specific training reaches 68.92%. — [arXiv 2506.12779](https://arxiv.org/html/2506.12779v3)
  - CoRe-MoE: merged walk+run gives a shuffling intermediate gait. — [arXiv 2606.04718](https://arxiv.org/html/2606.04718)
  - MoE-Loco: negative gradient cosines between tasks in an MLP. — [arXiv 2503.08564](https://arxiv.org/html/2503.08564v2)
- **Multi-critic.**
  - RobotKeyframing (ETH/Disney, 2024) learns separate value functions for dense-style rewards and sparse keyframe rewards within PPO. It reports better convergence and less tuning than a single critic. — [arXiv 2407.11562](https://arxiv.org/html/2407.11562v2)
  - Reward-Adaptive RL for bipedal locomotion uses a multi-head critic with one head per reward component and a shared trunk; the total value is the sum of heads. — [arXiv 2107.01908](https://arxiv.org/pdf/2107.01908)
- **General multi-task RL tools (abstract/primary, from prior knowledge; no locomotion-gait evidence found).**
  - PCGrad projects each task gradient onto the normal plane of any conflicting task gradient. — [arXiv 2001.06782](https://arxiv.org/abs/2001.06782)
  - PopArt adaptively normalizes value targets per task, so tasks with large returns don't dominate. A single agent was trained on all 57 Atari games. — [arXiv 1809.04474](https://arxiv.org/abs/1809.04474)
- **Behavior-parameter conditioning.** Walk These Ways (CoRL 2022) trains one policy over randomized gait parameters (frequency, contact timing, foot clearance, posture), with a reward-based curriculum widening the parameter ranges.
  - Gait choice matters at deployment. For example, a low-frequency gait fails to sprint on slippery ground and a high-frequency one succeeds. — [project page](https://gmargo11.github.io/walk-these-ways); [arXiv 2212.03238](https://export.arxiv.org/pdf/2212.03238)

### Inferences
- **Apply the grid curriculum to the user's library.** Index the curriculum by (vx, wz) cells that align with the library's twist grid. Start from the jog cells and expand to neighbors only when tracking (or reference-tracking) reward in a cell exceeds a threshold. Never remove cells, which protects against forgetting jog.
- **Adaptive sampling on top.** Add GMT-style failure-weighted sampling per cell so sprint and fast-turn cells get extra samples without starving jog. This directly addresses the user's "collapse/stand still when the curriculum widens too fast."
- **Reward scale.** Reward magnitudes likely differ across gaits: sprint has larger velocity terms and different reference errors. A cheap fix is per-gait normalization of the reference-tracking terms, or making the critic gait-aware with gait/period in its obs.
- **Multi-critic.** Per-gait critics or PopArt are plausible but unproven for this case. Use them only if value loss is visibly dominated by one gait.
- **Gait-ID routing is nearly free here.** Reference-based rewards are already gait-specific, because each library has its own reference, so this design is implicitly present already.

### Gaps
- I found no locomotion paper that applies PCGrad, PopArt or multi-critic specifically to speed or gait ranges and reports numbers.
- I found no quantitative study of forgetting slow gaits when fast gaits are added (Rapid Locomotion doesn't analyze it).
- The γ threshold value in Rapid Locomotion's grid curriculum is not stated in the text I retrieved.

## Q4. Recurrent vs MLP actors for multi-gait; does memory help infer gait or phase?

### Takeaway
Memory (GRU, LSTM, TCN, transformer or a history stack) is near-universal in successful multi-skill students. Where ablated, a memoryless MLP student fails badly (Parkour). For a gait library where the actor already gets an explicit phase clock and a commanded twist, memory matters less for gait identity and more for estimation (velocity, contact state, disturbances). The evidence for "memory infers phase" specifically is indirect.

### Cited Findings
- Robot Parkour: the GRU student succeeds on climb/leap/crawl/tilt at 86/80/100/73, vs 0/1/63/43 for an MLP student. Running is 100 for both, so memory was unnecessary for the run skill. — [arXiv 2309.05665](https://ar5iv.labs.arxiv.org/html/2309.05665)
- MoE-Loco's gate is driven by an LSTM hidden state, so memory supplies the routing signal. — [arXiv 2503.08564](https://arxiv.org/html/2503.08564v2)
- BumbleBee's generalist uses a Transformer-XL with 10-step memory, and its experts are MLPs. — [arXiv 2506.12779](https://arxiv.org/html/2506.12779v3)
- GMT's student uses a 20-step proprio history. Motion-target ablations show that future context plus the immediate frame matters: MPKPE is 42.07 with both vs 49.52 with the future window only. — [arXiv 2506.14770](https://arxiv.org/html/2506.14770v2)
- Siekmann et al. (ICRA 2021, Cassie hardware) use one LSTM policy conditioned on gait parameters (phase ratios and the like). It transitions between all two-beat gaits (walk, run, hop, gallop/skip), driven by a parametric periodic reward. — [arXiv 2011.01387](https://ar5iv.labs.arxiv.org/html/2011.01387)
- Gait-Conditioned RL (G1) uses a single recurrent policy for stand/walk/run. — [arXiv 2505.20619](https://arxiv.org/abs/2505.20619v3)
- Rapid Locomotion reaches 3.9 m/s on hardware with an MLP body plus a 15-step history encoder, not an RNN. — [arXiv 2205.02824](https://ar5iv.arxiv.org/html/2205.02824)

### Inferences
- The user's actor gets an explicit normalized phase clock and twist, so gait identity is given rather than inferred. An MLP with a short history stack is likely sufficient for PPO-from-scratch, as in Rapid Locomotion's MLP plus history.
- Recurrence is most valuable for a distilled student, and for estimating base velocity at sprint speeds, where slip and flight phases make single-frame estimates noisy.
- If phase-clock frequency differs between gaits, a transition crossing a bin boundary changes the clock rate. Memory can help smooth that change, but the clock design itself (continuous frequency interpolation vs discrete jump) likely matters more.

### Gaps
- I found no direct ablation of an RNN vs an MLP+clock actor on a humanoid multi-speed library.

## Q5. Direct evidence on "one policy vs specialists" quality gaps across locomotion speed ranges

### Takeaway
Across the papers I found, the pattern is consistent: a naive single policy on merged, heterogeneous skills underperforms. Specialists distilled into one student, or a gated or MoE single policy, recover most of the specialist quality, often with smoother transitions. Typical residual gaps:
- 0-10 points on most skills;
- up to about 25 points on the hardest skill;
- the student sometimes beats experts on skills that share structure.

I found no paper that isolates a speed-range split (jog vs run vs sprint) with numbers.

### Cited Findings
- **BumbleBee (sim, MuJoCo):** naive single generalist 33.01%, distilled from learned-cluster experts 66.84%. Clustering by behavior matters: random partitions give 35.36%. — [arXiv 2506.12779](https://arxiv.org/html/2506.12779v3)
- **Parkour (sim):** distilled vs specialist 86/80/100/73/100 vs 95/82/100/100/100. Single-stage PPO without distillation largely fails (0/0/73/0/100). — [arXiv 2309.05665](https://ar5iv.labs.arxiv.org/html/2309.05665)
- **Barkour (hardware):** distilled transformer is slightly below specialist switching on score, with smoother transitions. — [arXiv 2305.14654](https://arxiv.org/abs/2305.14654v1)
- **MoE-Loco (sim):** single MoE policy 0.879 vs single MLP 0.571 on the mixed course. — [arXiv 2503.08564](https://arxiv.org/html/2503.08564v2)
- **CoRe-MoE (G1, walk/run 0-2.5 m/s):** single network on merged walk+run data gives an intermediate gait. A single network with velocity-selected references reaches 54.2% success, vs 99.13% for velocity-gated 2-expert MoE plus terrain experts. The comparison is confounded by Stage 2 terrain experts. — [arXiv 2606.04718](https://arxiv.org/html/2606.04718)
- **Counter-evidence that a single PPO policy can cover a wide speed range** without experts, if the curriculum is right: Mini Cheetah to 3.9 m/s with one MLP-plus-history policy and a grid curriculum (hardware). — [arXiv 2205.02824](https://ar5iv.arxiv.org/html/2205.02824)
- **DeepMimic-style per-skill training as the baseline practice.** Peng et al. 2020, "Learning Agile Robotic Locomotion Skills by Imitating Animals", trains a separate policy per reference motion (abstract/primary, from prior knowledge). — [arXiv 2004.00784](https://arxiv.org/abs/2004.00784)

### Inferences
- **Decision guidance for the user's four-gait fast-locomotion policy:**
  - **(a) Single PPO on the merged library.** Viable if paired with a grid/local-expansion curriculum over the twist grid, failure-weighted cell sampling, and a gait-disambiguating input (the per-gait phase clock period). The user's prior walk+jog success fits this. The main risk is a shuffling or averaged stride near 1.5 and 2.5 m/s, and sprint starving: the "jitter in place" collapse.
  - **(c) Soft MoE actor (and optionally critic) in the same PPO run.** Gate on twist, clock and proprio. The cheapest upgrade from (a), with the strongest direct evidence of reduced interference (MoE-Loco, CoRe-MoE).
  - **(b) Specialists plus DAgger.** The most robust route to per-gait quality. Teachers can use privileged and reference observations, and the selector is a command-bin lookup. Costs: N+1 training runs, a likely gap on the hardest skill (sprint), and a probable RL fine-tune of the student.
  - **(d) Hybrids** are what the most recent humanoid papers actually do. Examples: MoE teacher plus distilled student (GMT); pretrained experts to initialize gate or MoE (MELA, CoRe-MoE Stage 1); distill then RL fine-tune (Parkour-in-the-wild, Perceptive Humanoid Parkour).
- **Suggested order of trials for the user:** (a) with a grid curriculum and adaptive sampling, then (c) as a drop-in, then (b)/(d) only if sprint or turns lag the specialists by a meaningful margin.

### Gaps
- No paper found reports one-policy vs specialist gaps specifically across jog/run/sprint speed bins on a humanoid, with numbers.
- No hardware quantitative numbers exist for humanoid multi-gait MoE: GMT, BumbleBee and CoRe-MoE give qualitative real-world results only.
- Not investigated in this session: OmniH2O, Puppeteer, HumanoidVerse/ProtoMotions implementation specifics, and PCGrad-in-locomotion papers.
