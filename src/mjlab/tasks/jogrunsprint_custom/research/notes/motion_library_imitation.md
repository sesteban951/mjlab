# Training one policy from a library of reference motions (multi-clip imitation, velocity-conditioned locomotion)

Scope: how multi-clip imitation methods select, sample, phase, blend, and condition on reference motions, and how that maps onto a twist-indexed, one-stride periodic gait library (G1, mjlab, 50 Hz; jog/run/sprint plus fast pivots; reference-free actor with a phase clock; reference-aware critic and rewards).

How well each claim was checked: items marked **[verified]** were read in the full paper text or in local source code during this session. Items marked **[abstract]** come only from the abstract or project page. Items marked **[recall]** are well-known details cited to the primary paper but not re-read in this session. The report writer should prefer [verified] items for load-bearing statements.

---

## Q1. How each method handles many clips, different lengths, timing/phase, and joystick (velocity) conditioning

### Takeaway
Methods fall into four families:
1. **Phase-indexed tracking of one clip at a time.** Examples: DeepMimic, BeyondMimic, KungfuBot, ASAP, and the Cassie gait library. The policy, or at least the critic, gets a clip identity plus a phase or a time-indexed reference.
2. **Phase-free distribution matching.** Examples: AMP, ASE, CALM. A discriminator scores state transitions against the whole dataset, so clip lengths and phase do not matter, and a task reward handles the velocity command.
3. **Kinematic generator plus a physics tracker.** Examples: DReCon (motion matching), RuN (a learned conditional motion generator), SPRINT (a spectral gait prior), MVAE, and the BeyondMimic diffusion planner. The command drives a kinematic generator, and the RL policy tracks or corrects what it produces.
4. **Large-scale universal trackers.** Examples: PHC/PHC+, GMT, OmniH2O, ExBody2, MaskedMimic. These condition on reference frames, and future windows of them, rather than on a phase, and they use failure-based clip sampling.

The closest precedents to the user's setup are the Cassie HZD gait library (1331 trajectory-optimized periodic gaits indexed by command) and SPRINT (5 single-cycle clips turned into a speed- and frequency-continuous prior, 5.9 m/s on G1 hardware).

### Cited Findings

**DeepMimic (Peng et al. 2018), three multi-clip mechanisms [verified]**
- The state includes a phase φ ∈ [0,1] that goes from 0 at the start of a motion to 1 at the end and resets each cycle for cyclic motions. The authors note that because phase advances linearly in time, it "limits the ability of the policy to adjust the timing of the motion." — [DeepMimic](https://ar5iv.labs.arxiv.org/html/1804.02717)
- **Multi-clip reward.** The imitation reward is r^I_t = max_j r^j_t over k clips. The test case is walking with a changing target heading, using 5 clips (1 forward, 4 turns). The best-matching clip switches to a turn when the heading changes. Caveat: mixing very different motions (sideflip and frontflip) "can result in the policy imitating only a subset of the clips." — [DeepMimic](https://ar5iv.labs.arxiv.org/html/1804.02717)
- **Skill selector.** The goal is a one-hot clip ID, trained with imitation reward only. A new random goal is sampled at the start of each cycle, so the policy learns transitions between all clips. To give clips in a set a common cycle period, the authors duplicated first and last frames. — [DeepMimic](https://ar5iv.labs.arxiv.org/html/1804.02717)
- **Composite policy.** Separately trained sub-policies are mixed by a softmax over their value functions (values normalized to [-1,1], temperature 0.3). A new skill is sampled each cycle, never the same skill twice in a row, and get-up policies take over automatically after a fall. — [DeepMimic](https://ar5iv.labs.arxiv.org/html/1804.02717)
- Reference state initialization (RSI) and early termination (ET) are critical. Backflip normalized return was 0.791 with both, 0.730 with ET only, and 0.379 with RSI only. Walking was insensitive (0.980 / 0.981 / 0.974). — [DeepMimic](https://ar5iv.labs.arxiv.org/html/1804.02717)

**AMP (Peng et al. 2021), simulated humanoid [verified]**
- There is no phase input: "our policies are not trained to explicitly imitate any specific motion from the dataset." RSI samples start states randomly from all clips. — [AMP](https://ar5iv.labs.arxiv.org/html/2104.02180)
- **Target-heading task.** The goal is (local target direction, target speed v*), with v* ∈ [1,5] m/s. The locomotion dataset is 434 s over 56 clips (walk 229.6 s / 9 clips, run 204.4 s / 47 clips). Gaits emerge with speed: walking at about 1 m/s, jogging at about 2.5 m/s, and a fast run near 4.5 m/s. Policies trained only on walking clips never run; policies trained only on running clips cannot go slow. Task and style weights are 0.5 / 0.5. — [AMP](https://ar5iv.labs.arxiv.org/html/2104.02180)
- The discriminator sees root linear and angular velocity, local joint rotations (normal and tangent vectors), local joint velocities, and end-effector positions, all in the local frame. Stated limitation: "AMP is susceptible to mode collapse" on large, diverse datasets. — [AMP](https://ar5iv.labs.arxiv.org/html/2104.02180)

**AMP for real-world quadrupeds (Escontrela et al. 2022) [verified]**
- The dataset is only **4.5 s** of German Shepherd mocap (pace, trot, canter, turn-in-place), retargeted to A1 by IK, with velocities from finite differences. — [Escontrela et al.](https://ar5iv.labs.arxiv.org/html/2203.15103)
- The velocity command (vx ∈ (−1,2) m/s, vy ∈ ±0.3 m/s, ω ∈ ±1.57 rad/s) enters through the task reward. The discriminator takes (s, s′) transitions with joint angles, end-effector positions, joint velocities, and base linear and angular velocities. It is an MLP with [1024, 512] hidden units and gradient penalty 10. Style weight is 0.65 and task weight 0.35, with style reward max[0, 1 − 0.25(D−1)²]. No phase. — [Escontrela et al.](https://ar5iv.labs.arxiv.org/html/2203.15103)
- Gait transitions emerge on hardware: pace at 0.8 m/s, trot at 1.7 m/s, and pace to canter as the command rises from 1 to 2 m/s. Cost of transport (simulation) was about 0.93–1.12 for AMP versus 1.37–1.54 for a hand-designed style reward and 5–14 for no style reward. Speed and COT tables are simulation results; hardware results are qualitative. — [Escontrela et al.](https://ar5iv.labs.arxiv.org/html/2203.15103)

**ASE / CALM / MaskedMimic (NVIDIA line)**
- ASE pre-trains a latent-conditioned low-level policy on a motion dataset using adversarial imitation plus a skill-discovery objective. Downstream task policies output latents. There is no clip phase. [abstract/recall] — [ASE](https://arxiv.org/abs/2205.01906)
- CALM learns a motion encoder plus a latent-conditioned policy and discriminator, so a specific motion can be requested via its latent. High-level direction control and finite-state composition are done at inference. [abstract/recall] — [CALM](https://arxiv.org/abs/2305.02195)
- MaskedMimic distills a full-body motion tracker into a masked, conditional-VAE-style controller. The controller accepts partial constraints (keyframes, root trajectory, text, objects), which gives joystick-like control from a tracker without a phase. [abstract/recall] — [MaskedMimic](https://arxiv.org/abs/2409.14393)

**PHC / PHC+ (Luo et al.) [PHC verified; PHC+ recall]**
- The goal is the next-step reference only (t+1, relative to the simulated humanoid). The paper says this gives no future information and limits highly dynamic motions. — [PHC](https://arxiv.org/html/2305.06456)
- **Progressive multiplicative control policy (PMCP).** Primitive P(1) trains on the full AMASS set (11,313 training sequences) until success on the hard set plateaus. The sequences it fails form Q_hard(1). P(1) is then frozen, and P(2) is warm-started and trained on the hard set, and so on (Q_hard(k) ⊆ Q_hard(k−1)). Four primitives were used, including a fail-state recovery primitive; a fifth did not help. A multiplicative composer (MCP) weights all primitives and scored slightly better than a top-1 mixture of experts. — [PHC](https://arxiv.org/html/2305.06456)
- Start points use RSI ("randomly select a starting point"). Success on AMASS-train was 98.9%. The recovery primitive is trained on a locomotion subset with point-goal reward (0.5) plus AMP (0.5). — [PHC](https://arxiv.org/html/2305.06456)
- PULSE introduces PHC+, an improved imitator, and distills it into a latent motion representation for downstream tasks. [abstract/recall] — [PULSE](https://arxiv.org/abs/2310.04582)

**BeyondMimic (Liao et al. 2025) [verified from paper; mjlab code read locally]**
- **Actor observations:** motion phase cue ψ (reference joint positions and velocities "used only for progress"), 9-D anchor-pose error, IMU twist, joint positions relative to default, joint velocities, and previous action. No history stacking. — [BeyondMimic](https://arxiv.org/html/2508.08241)
- **Critic observations:** the actor observations plus per-body poses relative to the anchor. — [BeyondMimic](https://arxiv.org/html/2508.08241)
- The paper presents per-motion policies under one shared formulation ("each motion is trained with its own policy"), yet also says many clips "were trained jointly with multiple other skills". The paper does not fully reconcile these. — [BeyondMimic](https://arxiv.org/html/2508.08241)
- **Joystick control is not done with RL.** A latent state-action diffusion model, built over a conditional VAE of the tracking policies' actions, is steered at inference by classifier guidance with cost G_js = ½ Σ_i ‖V_xy,i − g_v‖² over a 16-step horizon. Running reached "over 50 m on a track", with simulation velocity-tracking error of 13.65% (running) versus 12.14% (walking). Deployed zero-shot. — [BeyondMimic](https://arxiv.org/html/2508.08241)
- **Termination:** anchor or end-effector height error above 0.25 m, or anchor orientation error above 0.8 rad. **Reset:** reference state at the sampled phase plus perturbations (xy ±0.05 m, yaw ±0.2 rad, linear velocity ±0.5 m/s, and so on). — [BeyondMimic](https://arxiv.org/html/2508.08241)

**GMT (General Motion Tracking, 2025) [verified]**
- The data is 8,925 clips (33.12 h) from AMASS and LAFAN1, filtered in two stages: rule-based first, then by the completion rate of a preliminary policy. — [GMT](https://arxiv.org/html/2506.14770)
- **Motion input:** about 2 s (100 frames) of future targets compressed by a 1D-conv encoder into a 128-D latent, plus the immediate next frame. Dropping the immediate frame degraded tracking, and longer windows improved it. Each frame contains joint positions, base velocities, roll/pitch, root height, and heading-aligned key-body positions. — [GMT](https://arxiv.org/html/2506.14770)
- A soft mixture of experts in the teacher. The gating weights shift across phases of composite motions. — [GMT](https://arxiv.org/html/2506.14770)

**KungfuBot / PBHC (2025) [verified]**
- One policy per motion. The actor sees a 5-step proprioceptive history and **a time-phase φ ∈ [0,1] with no reference positions**. The critic additionally gets reference motion positions, root linear velocity, and randomized physics parameters. This is the same reference-free-actor plus phase-clock design the user has. — [KungfuBot](https://arxiv.org/html/2506.12851v3)
- **Adaptive tracking factor.** Reward width σ ← min(σ, EMA of tracking error). It is non-increasing from a large σ_init, aiming at σ* equal to the mean optimal tracking error. — [KungfuBot](https://arxiv.org/html/2506.12851v3)
- ASAP, which KungfuBot builds on, likewise trains phase-conditioned single-motion tracking policies and adds a delta-action model learned from real data. [abstract/recall] — [ASAP](https://arxiv.org/abs/2502.01143)

**Trajectory-optimized gait libraries indexed by command (Cassie, Li et al.) [verified]**
- **2021 version.** Library p = [vx, vy, height] on an 11×11×11 grid: vx ∈ [−1,1] m/s, vy ∈ [−0.3,0.3] m/s, height 0.65–1.0 m, for 1331 HZD gaits. Each gait is a pair of 5th-order Bézier steps (right and left stance), normalized by step period, with step switches triggered by foot impact. "A reference gait g^r is constructed by interpolating the parameterized gait library with respect to p^d" (interpolation scheme not stated). — [Li et al. 2021](https://ar5iv.labs.arxiv.org/html/2103.14295)
- **What the 2021 policy observes:** command; reference motor positions and velocities at t, t+1, t+4, t+7; 5-step state history; and 4 past actions. The value function gets privileged dynamics. Commands are resampled every 8 s. — [Li et al. 2021](https://ar5iv.labs.arxiv.org/html/2103.14295)
- **Out-of-library commands.** Training commands extend beyond the library (vx up to ±2 m/s), but a 2 m/s command gives only about 1.2 m/s (high-fidelity simulation). — [Li et al. 2021](https://ar5iv.labs.arxiv.org/html/2103.14295)
- A follow-up notes the robot "could not walk much faster than the maximum speed of the gait library." — [Adaptive Mimic, Zhang et al.](https://ar5iv.labs.arxiv.org/html/2112.03735) (via search snippet)
- **2024 version (IJRR).** Same 1331-gait walking library (0.8 s period, no yaw in the library). Running uses a single retargeted mocap clip (average 3 m/s, about 0.1 m flight foot clearance), with a separate running-to-standing transition motion. The policy previews the reference at t+1, t+4, t+7 and uses a 2 s I/O-history CNN. No explicit phase input is reported. — [Li et al. 2024](https://arxiv.org/html/2401.16889)

**Motion matching plus RL (DReCon, Bergamin et al. 2019) [verified]**
- **Database and feature vector.** About 10 min of mocap at 60 fps, matched on a 27-D feature: local foot positions and velocities, hip velocity, and trajectory positions and headings 20, 40, and 60 frames ahead. — [DReCon](https://theorangeduck.com/media/uploads/other_stuff/DReCon.pdf)
- **Search.** Runs every 10 frames, or when input changes quickly. The query's future trajectory comes from extrapolating the gamepad input with a critically damped spring. The nearest neighbor (KD-tree) under squared Euclidean distance wins. On a switch, inertialization blending is used rather than cross-fades. Styles are handled by restricting search ranges instead of one-hot encoding. — [DReCon](https://theorangeduck.com/media/uploads/other_stuff/DReCon.pdf)
- **Policy.** The state is 110-D: kinematic and simulated center-of-mass velocity, desired velocity v_des, the difference between simulated and desired velocity, positions and velocities of 6 bodies for both the simulated and kinematic characters, and the previous action. "No phase information is needed since information relating to kinematic character tracking is provided explicitly." Actions are offsets on the motion-matching PD targets for a joint subset. — [DReCon](https://theorangeduck.com/media/uploads/other_stuff/DReCon.pdf)
- **Training.** An artificial user samples random, uniformly distributed desired direction, heading, and turn rates. The kinematic character is initialized from a random database pose. — [DReCon](https://theorangeduck.com/media/uploads/other_stuff/DReCon.pdf)
- **Related work [recall].** SuperTrack reuses this kinematic-generator-plus-tracker idea with a learned world model ([SuperTrack](https://doi.org/10.1145/3478513.3480527)). Learned Motion Matching replaces the database search with compact networks ([LMM](https://doi.org/10.1145/3386569.3392440)). Motion VAEs use an autoregressive conditional VAE whose latent is driven by an RL joystick controller ([MVAE](https://arxiv.org/abs/2103.14274)).

**Learned command-conditioned generators plus residual RL (G1 humanoid)**
- **RuN [verified].** A mixture-of-experts conditional motion generator maps (current joint positions and velocities, velocity command) to the next frame. It is trained on 400 filtered and mirrored AMASS walk/run sequences, mostly 0.5–1.5 m/s, with sampling weighted by a velocity histogram. The actor sees base angular velocity, gravity, command, joints, and last action. **The critic additionally sees the generated reference.** The action is a residual added to q_ref. Imitation reward exp(−0.6‖q−q_ref‖²) with weight 1.0. Simulation and **G1 hardware 0–2.5 m/s**. Walk-to-run changes emerge from the generator, with no explicit switch logic. — [RuN](https://arxiv.org/html/2509.20696)
- **SPRINT [verified].** Five single-cycle LAFAN1 clips at speeds {0.66, 1.10, 2.29, 2.87, 3.40} m/s with primary frequencies {0.68, 0.86, 1.25, 1.36, 1.58} Hz. A frequency-conditioned VAE plus FiLM generator over phase harmonics φ = 2πft (6 harmonics) generates 10 sagittal joints. An FFT shows a half-cycle offset between contralateral limbs. Action is q_target = q_ref + α·a_res. **The actor has no reference input; the critic sees the prior's reference.** The speed curriculum expands to vx ∈ [0,6] m/s (f ∈ [0.6,2.3] Hz). — [SPRINT](https://arxiv.org/html/2605.28549)
- **VMP (Disney).** A VAE over short motion windows plus a latent-conditioned tracking policy, evaluated on unseen motions in simulation and on a bipedal robot. [abstract] — [VMP](https://la.disneyresearch.com/wp-content/uploads/VMP_paper.pdf)

**Large-scale humanoid teleoperation trackers [abstract/recall]**
- H2O filters infeasible AMASS motions with a privileged simulation policy ("sim-to-data"). OmniH2O trains a privileged teacher on full references and distills, via DAgger, to a student with sparse inputs (head and hands) plus history. — [H2O](https://arxiv.org/abs/2403.04436), [OmniH2O](https://arxiv.org/abs/2406.08858)
- ExBody has the upper body track reference poses while the legs follow root velocity commands. — [ExBody](https://arxiv.org/abs/2402.16796)
- ExBody2 uses a teacher-student setup with automatic feasibility filtering, decouples velocity tracking from keypoint tracking, and shows walk, crouch, and dance on hardware. — [ExBody2](https://arxiv.org/abs/2412.13196)
- HumanPlus (shadowing transformer on about 40 h of AMASS) and TWIST (teleoperation with an RL+BC teacher-student) are general trackers with no gait-library or velocity-indexed clip structure. — [HumanPlus](https://arxiv.org/abs/2406.10454), [TWIST](https://arxiv.org/abs/2505.02833)

### Inferences
- The user's design (reference-free actor, phase clock, twist command; reference only in the critic and rewards) is an established pattern. KungfuBot uses it per clip, and SPRINT and RuN use it with a generated reference. What is new is a phase clock shared across clips with *different periods*. DeepMimic sidestepped this by padding clips to a common period. The Cassie work used time-indexed reference previews instead of a phase scalar.
- The Cassie results suggest a nearest-neighbour or interpolated library caps achievable speed near the library maximum, even when commands go beyond it. The sprint top end (3.5 m/s) is therefore probably a hard ceiling unless the tracking reward is relaxed at the extremes.
- AMP's ablation (walk-only data never runs; run-only data cannot go slow) supports training jog, run, and sprint jointly in one policy. Mode collapse toward easier clips is the documented risk, which argues for failure-based or per-clip sampling (see Q3).

### Gaps
- No primary source found for BeyondMimic distilling to a *reference-free, velocity-commanded RL policy*. Its velocity control is diffusion guidance at inference.
- Li et al. 2021 does not state the interpolation scheme (coefficient interpolation versus nearest neighbour) or how the reference is updated when the command changes mid-stride.
- The exact mechanics of ASE, CALM, MaskedMimic, PHC+, OmniH2O, ExBody2, HumanPlus, TWIST, and ASAP were not re-read in full this session (abstract or recall level only).

---

## Q2. Reference-free vs reference-conditioned actors; student-teacher setups

### Takeaway
Three actor designs work:
- **Reference-conditioned actors** (PHC, GMT, OmniH2O teacher, DReCon, Cassie library). They track best and need no phase, but require the reference at deployment.
- **Phase-only actors with a reference-aware critic** (KungfuBot, BeyondMimic's ψ cue, which is reference-derived but used only for progress). These are deployable without a reference pipeline, but each phase value must map to one motion, so they are per-clip or need the command to disambiguate clips.
- **Command-only actors acting as residuals on a generated reference, with the reference in the critic** (SPRINT, RuN). These are the most direct precedent for one twist-commanded multi-gait policy.

Teacher-student distillation (privileged teacher, then a deployable student via DAgger) is standard in GMT and OmniH2O. In those works the student still receives motion targets; distillation removes privileged state, not the reference.

### Cited Findings
- **KungfuBot.** The actor gets proprioception plus φ only. The critic adds reference positions, root linear velocity, and physics parameters. Deployed zero-shot on G1. — [KungfuBot](https://arxiv.org/html/2506.12851v3)
- **BeyondMimic.** The actor gets phase cue ψ and anchor error. The critic adds body poses relative to the anchor. — [BeyondMimic](https://arxiv.org/html/2508.08241)
- **GMT.** The teacher is PPO with privileged information (root linear velocity, root height, key-body positions, contacts, mass and motor randomization) plus motion targets. The student gets proprioception, 20-step history, and motion targets, trained with DAgger and an L2 action loss. Teacher training took about 3 days and student training about 1 day on an RTX 4090. — [GMT](https://arxiv.org/html/2506.14770)
- **SPRINT.** The actor gets angular velocity, gravity, commands, joints, and previous residual (MLP + LSTM). The critic adds linear velocity and the prior's reference trajectories. Removing the asymmetric actor-critic ("w/o AAC") lowered max speed from 5.96 to 5.46 m/s. — [SPRINT](https://arxiv.org/html/2605.28549)
- **RuN.** The critic gets privileged base linear velocity plus (q_ref, q̇_ref). The actor gets command and proprioception. The residual is added to q_ref. — [RuN](https://arxiv.org/html/2509.20696)
- **DReCon.** The actor sees the kinematic target explicitly, so no phase is needed. — [DReCon](https://theorangeduck.com/media/uploads/other_stuff/DReCon.pdf)
- **Li et al. 2021.** The policy sees reference previews. The value function gets ground-truth state and dynamics parameters. — [Li et al. 2021](https://ar5iv.labs.arxiv.org/html/2103.14295)
- **OmniH2O.** Privileged teacher on full references, DAgger to a student with sparse VR keypoints. [abstract/recall] — [OmniH2O](https://arxiv.org/abs/2406.08858)

### Inferences
- In SPRINT and RuN the reference is still used at deployment, as the residual base q_ref, even though the actor never observes it. The user's actor outputs absolute targets with no reference base, which is the KungfuBot style. Across clips that requires the (twist, phase) pair to identify the motion uniquely.
- With snap-to-nearest-clip, the twist command determines the clip, so this condition holds. Near clip boundaries, though, the same twist plus phase may correspond to different target postures depending on blend state. Observing a clip-ID or stride-period scalar (or phase rate dφ/dt) is a cheap way to remove that ambiguity, analogous to DeepMimic's one-hot skill selector.
- SPRINT's AAC ablation (about 0.5 m/s top-speed loss without the privileged critic) suggests a reference-aware critic matters most at the high-speed end.

### Gaps
- No paper found that ablates *phase-only actor versus reference-observing actor* on a multi-clip periodic locomotion library.
- No paper found distilling a multi-clip tracking teacher into a reference-free, twist-commanded student for humanoid running.

---

## Q3. Clip sampling strategies: uniform, adaptive/failure-based, curriculum; handling much harder clips

### Takeaway
Failure-driven sampling is standard:
- BeyondMimic: EMA of failure counts per time bin, plus a uniform floor and a non-causal kernel.
- GMT: completion levels with error-scaled weights.
- PHC: hard-negative mining into new network capacity.
- KungfuBot: shrinking reward tolerance.
- Speed curricula from slow to fast (SPRINT) matter for top speed. Removing SPRINT's curriculum cut max speed from 5.96 to 4.48 m/s.

For a short-clip library, BeyondMimic's per-clip 1-s time bins collapse to a single bin per clip, so sampling should be re-indexed per clip (or per clip and command cell).

### Cited Findings
- **BeyondMimic adaptive sampling.**
  - The motion is split into S one-second bins. Each bin's failure rate is smoothed with an EMA, f̄_s ← 0.999 f̄_s + 0.001 f_s.
  - A uniform floor of 0.1/S is added to each bin, and a non-causal kernel ρ^u (ρ = 0.8, u ∈ {0,1,2}) shifts mass to the bins *before* a failure.
  - The start bin is drawn from the normalized multinomial. Sampling "gradually reverts to uniform" once hard segments are mastered.
  - Source: [BeyondMimic](https://arxiv.org/html/2508.08241)
- **mjlab implementation** (`src/mjlab/tasks/tracking/mdp/commands.py`, read locally; upstream [mjlab](https://github.com/mujocolab/mjlab)).
  - Terminated envs vote for bin ⌊t·S/T⌋, giving `bin_failed_count`. Each update computes p = conv1d(failed + 0.1/S, kernel) and normalizes it.
  - The EMA is applied with `adaptive_alpha = 0.001` on global reset.
  - Defaults: `adaptive_kernel_size = 1` (no smoothing, unlike the paper's 3-tap kernel), `adaptive_lambda = 0.8`, `adaptive_uniform_ratio = 0.1`.
  - Bins index fractions of **one** motion's length; there is no notion of multiple clips.
- **GMT.**
  - Clips over 10 s are cut into ≤10 s sub-clips with a random offset of up to 2 s, and re-clipped periodically.
  - Each clip has a completion level c_i that starts at 10, is multiplied by 0.99 per completion, and has a floor of 1. The termination error threshold tightens with c_i.
  - Sampling weight is c_i, or (min(E_maxkeybody/0.15, 1))^5 once c_i = 1.
  - Adaptive sampling and the mixture of experts helped most on the hardest-percentile motions.
  - Source: [GMT](https://arxiv.org/html/2506.14770)
- **PHC.** Failed sequences form a hard set that trains a *new* primitive while earlier ones are frozen. This adds capacity instead of reweighting a single network. — [PHC](https://arxiv.org/html/2305.06456)
- **KungfuBot.** σ ← min(σ, EMA tracking error), an automatic curriculum on reward tolerance. — [KungfuBot](https://arxiv.org/html/2506.12851v3)
- **SPRINT curriculum.** The command range expands progressively from standing to 6 m/s. "w/o curriculum" reached max 4.48 m/s versus 5.96 m/s. Stable 4 m/s came after about 3 h; full convergence after about 6.5 h on one RTX 4090. — [SPRINT](https://arxiv.org/html/2605.28549)
- **RuN.** The motion generator is trained with weighted sampling by a velocity histogram, because the data is concentrated at 0.5–1.5 m/s. — [RuN](https://arxiv.org/html/2509.20696)
- **DeepMimic.** The skill selector samples clips uniformly at each cycle boundary. The composite policy never repeats a skill twice in a row. — [DeepMimic](https://ar5iv.labs.arxiv.org/html/1804.02717)
- **AMP.** Initial states are sampled uniformly from all clips. Mode collapse onto subsets is a known risk with diverse data. — [AMP](https://ar5iv.labs.arxiv.org/html/2104.02180)
- **Li et al. 2024.** Staged training: a fixed single command first, then task randomization, then dynamics randomization. — [Li et al. 2024](https://arxiv.org/html/2401.16889)

### Inferences
- **Re-key the sampler for the library.** Clips are 32–43 frames (0.64–0.86 s at 50 Hz), so BeyondMimic's 1-s bins give one bin per clip. The natural re-key is per clip or command cell (vx, wz) as the sampling unit, possibly crossed with a few phase bins (for example, stance-L / flight / stance-R / flight).
- **Keep a uniform floor.** A uniform floor (BeyondMimic's 0.1/S) prevents easy jog clips from being forgotten. The non-causal kernel generalizes naturally to neighbours in command space: smooth failure mass to adjacent vx and wz cells, since a failure at sprint vx = 3.4 often implies difficulty at 3.2.
- **Ramp sprint in gradually.** For a much harder sprint sub-library, combine failure-weighted sampling with a speed curriculum: gate the top end of the vx range on success at lower speeds, as SPRINT does. SPRINT's ablation shows the curriculum is worth about 1.5 m/s of top speed. GMT-style completion levels, with tighter termination thresholds only after a clip is mastered, also stop sprint failures from dominating early training.
- **Escalate only if needed.** PHC's progressive primitives, or GMT's mixture of experts, are the escalation if one network cannot fit jog through sprint plus pivots. The documented signal is a plateau in hard-set success.

### Gaps
- No source evaluated failure-based sampling over a *command-indexed* library of short periodic clips specifically.
- BeyondMimic's paper does not define exactly how a failure event f_s is counted. The mjlab code counts terminations attributed to the bin of the current time step.

---

## Q4. Transition handling between clips (deepest coverage, the user's top priority): motion matching, blending, phase alignment, transition clips, smoothness rewards, command filtering, evaluation

### Takeaway
No surveyed system hard-swaps between periodic references of different periods mid-stride without help. The working recipes fall into four groups:
- **Re-selection plus inertialization** (DReCon, motion matching). Features include foot position and velocity, so the matched frame has the same stance foot. Search is periodic, and the offset is decayed with a spring rather than two clips being cross-faded.
- **Step-synchronous parameter updates** (HZD libraries: Da et al. 2016 on hardware, Li et al. on Cassie). The library holds left- and right-stance steps. Interpolated gait coefficients are updated at foot impact. The phase is reset on impact and can be pulled toward a mechanical (state-based) phase by an observer.
- **Continuous in-phase splicing of interpolated references** (NaviGait).
- **Removing discrete clips altogether** (SPRINT, RuN, AMP, Siekmann). A single prior that is continuous in command or frequency, a learned command-conditioned generator, or a clock-parameterized reward makes gait changes emerge.

**Robustness to switches.** Policies that will see switches must be trained with randomized switches and phase jumps (RPG, DeepMimic's skill selector). Otherwise transition success collapses: in RPG, hard expert switching gave 5–65% success versus 70–95% with transition randomization.

**Evaluation.** The quantitative transition metrics in the literature are transition success rate, and mean joint acceleration within ±50 steps of the switch (RPG). Smoothness is otherwise enforced by action-rate penalties, residual-rate penalties, and command acceleration limits.

**No direct A/B evidence** was found comparing reference-free and reference-observing actors on transition smoothness. The indirect evidence is in the Inferences below.

### Cited Findings

**A. Continuous re-selection and inertialization (motion matching)**
- **DReCon search.**
  - Every 10 frames, "or when the user input changes quickly", a query is built from the playing frame's features plus future trajectory points (20, 40, and 60 frames ahead) from critically damped spring extrapolation of the gamepad. The nearest database frame (squared Euclidean distance on normalized features, via KD-tree) is selected.
  - A transition happens only if the best match differs from the frame already playing.
  - Features include **left and right foot local positions and velocities** and hip velocity. A match therefore needs the same foot configuration and swing direction, which is what prevents left/right swaps.
  - Source: [DReCon](https://theorangeduck.com/media/uploads/other_stuff/DReCon.pdf)
- **DReCon blending.** Inertialization is used instead of cross-fade, so only "a single track of animation" is evaluated at a time. The RL tracker is trained under an artificial user that randomizes desired direction, heading, and turn rate. The policy therefore sees the kinematic reference's transitions during training and learns to absorb them. It observes the kinematic target explicitly, with no phase. — [DReCon](https://theorangeduck.com/media/uploads/other_stuff/DReCon.pdf)
- **Inertialization mechanics.**
  - At the switch, store off_x = (src_x + off_x) − dst_x and the matching velocity offset. Then output dst + off, with the offset decayed by a critically damped spring of chosen half-life.
  - Pros: no fixed blend window, and a single animation is evaluated at a time. Con: possible slight overshoot.
  - The same spring, driven by stick input, predicts the future velocity trajectory used as the query (as in Learned Motion Matching). No canonical half-life value is given.
  - Source: [Holden, "Spring-It-On / spring roll call"](https://theorangeduck.com/page/spring-roll-call)
- **Phase manifolds.** DeepPhase's periodic autoencoder learns per-channel phase, frequency, amplitude, and offset from unstructured motion. It assumes motion is locally periodic "or transitions between periodic motions," and its phase-space distances improve motion-matching queries. [abstract/secondary] — [DeepPhase PDF](https://i.cs.hku.hk/~taku/deepphase.pdf)
- **WalkTheDog** extends this to a vector-quantized phase manifold in which "motions within the same component are aligned temporally by the phase variable", used with an improved motion-matching framework. [abstract] — [WalkTheDog](https://arxiv.org/abs/2407.18946)

**B. Step-synchronous gait-library transitions (trajectory-optimized periodic gaits, the closest analogue to the user's library)**
- **Da et al. 2016 (hardware).**
  - Library: 8 HZD gaits (time-based for −0.4 to 0.2 m/s, phase-based for 0.4 to 1.0 m/s, 0.2 m/s spacing).
  - Bézier coefficients are **linearly interpolated** in speed via ζ(v) = (v − v_i)/(v_{i+1} − v_i).
  - "Gait library updates gait parameters **step to step** based on the current speed," and the gaits are "suitable for the conditions of the current step."
  - Hardware transitions from 0 to 0.8 m/s following a reference speed.
  - Source: [Da et al. 2016](https://grizzle.robotics.umich.edu/files/Da_AccessMay2016.pdf)
- **Da et al. 2016 unified phase observer.**
  - τ̇ = 1/T + (L/T)(s(q) − τ), with τ⁺ = 0 at impact. Here s(q) is a mechanical phase computed from the robot state.
  - Gain L = 0 for |v| < 0.4 m/s (pure time-based) and L = 7v otherwise, so the phase is "approximately purely phase-based" at v ≥ 0.7 m/s.
  - Rationale: time-based phase suits stepping in place, where mechanical phase is ill-defined. Mechanical phase is more robust at speed. This is "a Luenberger observer for the phase."
  - Source: [Da et al. 2016](https://grizzle.robotics.umich.edu/files/Da_AccessMay2016.pdf)
- **Li et al. 2021 (Cassie).**
  - Each library gait is **two steps, right stance and left stance**, normalized by step period. Step switching is triggered by foot impact.
  - A reference for arbitrary commands is built by interpolating the library.
  - The policy observes the reference at t, t+1, t+4, t+7 rather than a phase scalar, so it can see an upcoming change.
  - Commands are resampled every 8 s during training.
  - Source: [Li et al. 2021](https://ar5iv.labs.arxiv.org/html/2103.14295)
- **ATRIAS stepping-stone libraries.** Built with **2-step periodic optimization**, so that step-to-step transitions between different gaits are feasible. The library is linearly interpolated for the next step's parameters instead of precomputing every pairwise transition. [search snippet] — [Nguyen et al. RSS 2017](https://hybrid-robotics.berkeley.edu/publications/RSS2017_DiscreteTerrain_Walking.pdf), [IJRR 2018](https://hybrid-robotics.berkeley.edu/publications/IJRR2018_ATRIAS_StochasticTerrainWalking.pdf)
- **NaviGait (BRUCE humanoid, hardware at 50 Hz).**
  - Library: 39 FROST/HZD gaits (degree-7 Bézier, forward and lateral velocities, no turning gaits). Off-grid commands use a **convex combination of the 3 nearest gaits' Bézier coefficients**.
  - **Splicing on a command change.** The current and desired splines are joined at the current phase τ(t1). The blended spline takes its first coefficients from the current gait, averages the middle ones, and takes the last ones from the desired gait. Phase is remapped as τ̂ = (t − t1)/(T − (t1 − t0)) so the remainder of the step maps to [0,1]. This repeats recursively. Base phase is time-based, τ = (t − t0)/T. Turning is done by rotating the reference relative to heading.
  - **Policy outputs** a joint residual Δq and a velocity residual Δv. The library is queried at v_d + Δv, so the policy can choose which gait to track.
  - **Smoothness** comes only from Δv-rate and Δq-rate penalties. No transition metric is reported.
  - **Versus reference-free RL.** A reference-free "Canonical RL" baseline showed more velocity drift and weaker push rejection, and trained slower (56 min versus 38 min).
  - Source: [NaviGait](https://arxiv.org/html/2510.11542v1)

**C. Transition clips, discrete gait modes, and hysteresis**
- **Li et al. 2024.** Running uses a dedicated **running-to-standing transition reference motion**. A training sub-stage teaches standing-to-running-to-standing transitions within one policy. The authors argue this is faster and more robust than switching separate policies. — [Li et al. 2024](https://arxiv.org/html/2401.16889)
- **DeepMimic skill selector.**
  - Clips are padded (first and last frames duplicated) to a **common cycle period**.
  - The one-hot goal changes only **at cycle boundaries**, sampled randomly so all pairwise transitions are seen.
  - The composite policy runs each skill for a full cycle and never repeats it twice in a row.
  - Source: [DeepMimic](https://ar5iv.labs.arxiv.org/html/1804.02717)
- **Gait-conditioned recurrent policy for G1 (reference-free).**
  - One LSTM policy with a one-hot gait ID (stand, walk, run, transition) plus a sin/cos gait phase. Reward terms are masked per mode.
  - **Switching rules use hysteresis and dwell timers:** walk-to-stand when |v_cmd| < 0.1 m/s, committed only after low speed and double support persist for 1.5 s. Run is enabled at Froude number v²/(gl) > 0.5. Run-to-walk lasts until a 2.5 s stable period completes.
  - Ablation: without reward routing, episode length was 11.6 versus 972.6.
  - Hardware covers stand, walk, and walk-to-stand only. Running and run-to-walk are simulation only (up to 3 m/s shown).
  - Source: [Gait-conditioned RL, arXiv 2505.20619](https://arxiv.org/html/2505.20619)
- **Switch.** A skill graph links frames across skills by kinematic similarity, with buffer nodes. The tracking policy trains on graph paths. An online scheduler graph-searches a feasible path when a switch is requested or tracking drifts. [abstract; a secondary summary's "100% switching success" is unverified] — [Switch](https://arxiv.org/abs/2604.14834)

**D. Training for robustness to switches, and transition metrics**
- **RPG (G1, humanoid fighting).** Built from frozen per-skill ASAP-style experts, whose actor sees phase, current reference, and 2 future reference frames.
  - **Policy-transition randomization:** each step, with probability p_trans, switch to a uniformly sampled new motion starting at frame 0.
  - **Temporal randomization:** with probability p_jump, skip ahead k ~ U{1,K} reference frames.
  - A gating network (separate upper- and lower-body weights) mixes the experts, with a torque-difference penalty of −0.5.
  - **Transition success over 20 trials per pair, hard-switched baseline versus RPG:** Jump→Kick 0.05 vs 0.75; Punch→Jump 0.35 vs 0.80; Kick→Jump 0.30 vs 0.70; Sword→Kick 0.65 vs 0.90.
  - **Smoothness metric:** mean joint acceleration over ±50 steps around the switch, for example Jump→Kick 1.76 (baseline), 1.53 (no gating), 1.45 (RPG). Not uniformly better: Punch→Punch was 1.24 for the baseline versus 1.35 for RPG.
  - Hardware is qualitative only. p_trans, p_jump, and K are not reported.
  - Source: [RPG](https://arxiv.org/html/2604.21355v2)
- **BeyondMimic** gets transitions from guided diffusion and keyframe inpainting over the learned latent state-action space, not from reference switching. Walking-to-running includes a light jog at low commanded speed. — [BeyondMimic](https://arxiv.org/html/2508.08241)
- **SPRINT** uses one frequency-conditioned prior (phase φ = 2πft, 6 harmonics, contralateral limbs a half-cycle apart), so there are no discrete switches. Commands are smoothed with an acceleration limit (value not given). Hardware transitions from standing through walking (double support) and jogging (flight) to 4 m/s running. — [SPRINT](https://arxiv.org/html/2605.28549)
- **RuN.** An autoregressive, command-conditioned generator (next frame from current frame plus velocity) produces walk-to-run transitions over 0–2.5 m/s on G1 hardware, with no switch logic. Action-rate (−0.04) and action-smoothness (−0.06) penalties apply. — [RuN](https://arxiv.org/html/2509.20696)
- **Siekmann et al. (Cassie, reference-free).**
  - Clock inputs are sin(2π(φ + θ_left)/L) and sin(2π(φ + θ_right)/L), plus phase ratios r.
  - Von Mises-smoothed phase-indicator rewards ("a smoothing effect on the reward function at phase boundaries").
  - Varying only the swing ratio, with the left/right offset fixed at 0.5, gives walk-to-run transitions.
  - Varying offsets too (walk to hop) produced asymmetric gaits unless hop-symmetry and standing transition penalties were added.
  - Clock parameters were not explicitly rate-limited.
  - Source: [Siekmann et al. 2021](https://ar5iv.labs.arxiv.org/html/2011.01387)
- **AMP and AMP-quadruped.** Gait transitions (walk, jog, run; pace, trot, canter) emerge from a command or task reward plus a transition-level discriminator, with no phase or switching logic. — [AMP](https://ar5iv.labs.arxiv.org/html/2104.02180), [Escontrela et al.](https://ar5iv.labs.arxiv.org/html/2203.15103)

### Inferences (applied to the jog/run/sprint/pivot library)

1. **Canonicalize the left/right convention first.** Every clip should start at the same event (for example, left touchdown), or each clip should store an explicit contact schedule. Fraction-of-stride remapping is only correct when phase 0 means the same foot event in all clips. This is the property HZD libraries guarantee by storing separate left- and right-stance steps and switching on impact (Li 2021, Da 2016).
   - If clips differ in **duty factor** (sprint has a shorter stance than jog), equal fractions do not put touchdowns at the same place. Map phase piecewise-linearly between contact events (touchdown L, liftoff L, touchdown R, liftoff R) rather than by raw fraction.
   - Mirrored clips should be produced by a left/right mirror operator, so the convention is exact by construction.

2. **Switch at a gait event, not at an arbitrary frame.** Da 2016 and the Cassie libraries update gait parameters step to step, at impact. DeepMimic changes goals at cycle boundaries. For the user's setup:
   - Latch a new clip only at the next touchdown, or at a flight-phase midpoint for run and sprint.
   - Add hysteresis on the command grid (DReCon only re-searches every 10 frames or on large input changes; the gait-conditioned G1 policy uses 1.5–2.5 s dwell timers and a Froude threshold).
   - Snap-to-nearest with per-step re-evaluation will chatter when the commanded vx sits near a library boundary (1.5 or 2.5 m/s).

3. **Integrate phase; don't recompute it.** With periods of 43, 37, and 32 frames, phase must evolve as φ̇ = 1/T(t), with T itself blended over the transition window. Otherwise the clock's rate jumps by about 16% at jog to run and 16% at run to sprint. NaviGait's τ̂ remap and Da's observer are both explicit phase-continuity mechanisms.
   - Because the actor sees only the clock, a rate step is an observation discontinuity. Either ramp T, or also feed the actor T (or φ̇), so the clock rate is not a hidden variable.
   - Option: Da-style correction τ̇ = 1/T + k(s_contact − τ), which pulls the reference phase toward the robot's measured contact phase. This prevents the reference and the robot drifting out of step after a disturbance or switch.

4. **Blend with inertialization, aligned in phase.** Cross-fading two references at the same phase fraction averages postures that may have different contact states. Better:
   - Align the target clip's phase to the robot's current contact state (step 1).
   - Record the pose and velocity offset between the old reference and the new one at the switch.
   - Decay that offset with a critically damped spring over about 0.2–0.4 s. The 0.4 s figure is the user's choice; no literature value was found.
   - This is DReCon's and Learned Motion Matching's approach. It avoids evaluating two references and never produces in-between postures that are not continuous in velocity.
   - A cheaper alternative is NaviGait-style coefficient splicing in the time-normalized stride, if clips are stored as splines.

5. **Consider interpolating between clips rather than snapping.** Li 2021 and NaviGait (3-nearest convex combination), Da 2016 (linear in speed), and SPRINT (continuous frequency prior) all avoid discrete snaps. With periodic clips from trajectory optimization, phase-aligned interpolation between neighbouring library cells is usually feasible within a sub-library:
   - First time-normalize each clip to its stride and align by contact events.
   - Then interpolate joint trajectories, and the period T, linearly in (vx, wz).
   - Interpolating across sub-libraries (jog to run boundary at 1.5 m/s) is valid only if the gait topology matches (both have flight phases). Otherwise generate a **dedicated transition clip** with trajectory optimization: a two-step, non-periodic trajectory from the end state of clip A's stride to the start state of clip B's stride. This mirrors ATRIAS's 2-step periodic gait design and Li 2024's running-to-standing transition clip. Fast pivots (|wz| 1–2 rad/s) are the likeliest to need explicit transition clips, since their contact pattern differs most.

6. **Train on switches.** DeepMimic resamples goals at random cycle boundaries, DReCon randomizes artificial-user input, and RPG randomizes transitions and frame jumps; RPG lifted success from as low as 5% to 70–95%. For the user:
   - Resample the twist command mid-episode often enough that every episode contains multiple library switches, across sub-library boundaries too, not only within jog.
   - Include occasional abrupt jumps (sprint to pivot) in addition to ramps.
   - Make adaptive sampling over (source clip → target clip) pairs, not just over clips, so failure-prone transitions get oversampled.

7. **Penalize discontinuity where it happens.**
   - Use action-rate and action-smoothness penalties (RuN: −0.04 and −0.06), and joint-acceleration and torque-rate penalties. NaviGait's Δv-rate and Δq-rate penalties are the residual analogues.
   - Optionally scale tracking rewards down, or widen their σ (KungfuBot-style), during the blend window, so the critic does not punish the policy for not matching a reference that is itself being blended.
   - Rate-limit the command with an acceleration limit (SPRINT) or a critically damped spring (motion matching), so the reference cannot jump sprint to pivot in one step.

8. **Evaluate transitions explicitly.** Build a transition test matrix of all ordered pairs (jog_i → run_j → sprint_k → pivot) at several switch phases. Suggested metrics:
   - transition success rate (RPG);
   - mean joint acceleration over ±50 steps of the switch (RPG's E_maccj);
   - peak base-velocity error and its recovery time to within 10% of command;
   - foot contact-sequence violations (two consecutive stances on the same foot, or a missed flight phase);
   - peak torque and torque-rate;
   - foot slip during stance.
   Of these, only success rate and joint acceleration have literature precedent with numbers (RPG). The rest are proposed.

9. **Reference-free versus reference-conditioned actors.** No direct ablation of transition smoothness was found. Indirect evidence:
   - DReCon says observing the kinematic target removes the need for phase and lets the policy see the transition as it happens.
   - Li et al. give a preview of future reference frames (t+1, t+4, t+7), and GMT found longer future windows improved tracking. Both suggest previewing an upcoming change helps.
   - Reference-free, clock-only policies (Siekmann; the gait-conditioned G1 policy) do transition smoothly. They needed explicit transition penalties, mode routing, or hysteresis, and Siekmann saw asymmetric gaits when offsets changed.
   - SPRINT and RuN keep the actor reference-free, but act as residuals on a reference that is itself continuous by construction. Smoothness comes from the generator, not from the actor.
   - My reading: with a reference-free actor, the burden of smoothness falls entirely on making the *observed clock and command* continuous (steps 2–3 and 7), plus critic and reward behaviour during blends. A reference-observing actor could exploit a reference preview to anticipate a switch, but it would also inherit any reference discontinuity directly into its inputs.

### Gaps
- No paper was found that measures foot-phase (left/right) mismatch errors when switching between periodic clips of different periods. The pitfalls above are reasoned from mechanisms.
- No published inertialization half-life recommendation for robot reference blending was found. The 0.4 s window is the user's choice and is unvalidated against literature.
- RPG did not report p_trans, p_jump, or K. Switch's mechanism details and the 100% success claim were not verified from the full text.
- No source quantitatively compared reference-free and reference-conditioned actors on transition smoothness.
- Da et al. 2016's exact timing of within-step versus at-impact parameter updates for speed changes is described as step-to-step, but within-step behaviour was not fully read.

---

## Q5. Imitation-based humanoid and biped running: reported speeds

### Takeaway
The fastest verified humanoid running from a motion prior is **SPRINT on Unitree G1: 5.91 m/s measured on hardware at a 6 m/s command (5.96 m/s in simulation)**. It used 5 clips (0.66–3.40 m/s) via a frequency-conditioned spectral prior and a speed curriculum, extrapolating well beyond the fastest clip.

Other results:
- RuN: 0–2.5 m/s on G1 hardware.
- BeyondMimic: continuous running over 50 m on G1 (no speed reported).
- Cassie (bipedal, not humanoid), tracking a 3 m/s mocap running reference: a 100 m dash in 27.06 s on hardware.
- AMP in simulation: emergent running near 4.5 m/s.

### Cited Findings
- **SPRINT (G1).**
  - Simulation max average speed 5.96 m/s. Hardware at a 6 m/s command measured 5.9143 m/s, with mean velocity error 0.0428 m/s across commands.
  - Baselines' maximum average speed: Humanoid-Gym 3.90, AMP 4.75, AI-CPG 4.82 m/s.
  - "w/o prior reward" reached 6.06 m/s but FID was 12.23 versus 0.80 (unnatural).
  - Reward shaping: target air time = 0.4/f_cmd; forward-tilt target 0.1 + 0.05·v_cmd; penalties on both feet grounded at high speed and both feet airborne at low speed.
  - Source: [SPRINT](https://arxiv.org/html/2605.28549)
- **RuN (G1).** Simulation vx 0–2.5 m/s (vy ±0.3, wz ±0.5 rad/s). Hardware from standstill to a 2.5 m/s run. Velocity error 0.287 versus AMP 0.290 and a tracking prior (GMP) 0.317. — [RuN](https://arxiv.org/html/2509.20696)
- **BeyondMimic (G1).** Running "over 50 m on a track"; simulation velocity-tracking error 13.65% for running; users preferred it 84.7% of the time. No m/s figure in the parts read. — [BeyondMimic](https://arxiv.org/html/2508.08241)
- **Li et al. 2024 (Cassie hardware).** 100 m in 27.06 s (≈3.7 m/s average, computed) and 400 m in 2:34 (≈2.6 m/s average, computed), from a single 3 m/s mocap running reference. — [Li et al. 2024](https://arxiv.org/html/2401.16889)
- **AMP (simulated humanoid).** Emergent fast run at about 4.5 m/s for target speeds up to 5 m/s. — [AMP](https://ar5iv.labs.arxiv.org/html/2104.02180)
- **GMT (G1).** Running appears in qualitative figures; no speeds reported. — [GMT](https://arxiv.org/html/2506.14770)
- **Other running work found in search but not read.**
  - CLF-guided RL running on G1 with flight phases (non-imitation): [CLF-RL](https://www.alphaxiv.org/abs/2509.19573)
  - GaitSpan, which grows a walking policy into running via multiple clocks plus residual. Robot and speeds not in the abstract: [GaitSpan](https://arxiv.org/abs/2607.12114)

### Inferences
- The user's sprint ceiling (3.5 m/s) is close to SPRINT's fastest *clip* (3.40 m/s). SPRINT reached about 6 m/s only by making the prior continuous in frequency and letting the residual policy extrapolate, with a curriculum.
- A snap-to-nearest library with per-clip tracking rewards is unlikely to extrapolate past 3.5 m/s. The Cassie library precedent shows it saturating near the library maximum.
- SPRINT's ablation shows that dropping prior tracking increases raw speed but destroys naturalness. Down-weighting tracking rewards near the top of the command range is a lever, at a cost in naturalness.

### Gaps
- No independent replication of SPRINT's 6 m/s G1 hardware number was found. The figure comes from the authors' own table (5.91 m/s measured).
- BeyondMimic's sprint speed in m/s was not found in the sections read.
- GaitSpan's robot and speeds were not available from the abstract.
