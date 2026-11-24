# Align the phase before blending gaits

Smooth gait-library transitions come from fixing the reference first and the policy second. Every library controller in the notes that changes gait smoothly does the same four things. These include the HZD libraries on Cassie, MARLO and BRUCE, motion matching, and DeepMimic's skill selector. They give all gaits one shared phase origin tied to a named foot event. They commit gait changes at a contact event, not at an arbitrary frame. They keep phase continuous and change only its rate. They interpolate or inertialize between neighbouring motions instead of cross-fading misaligned poses. None of them switches between periodic clips of different periods at an arbitrary point mid-stride, which is what your nearest-twist snap with timer resampling does. For your G1, the highest-leverage changes are five fixes to the reference and the training:

- **Re-anchor every clip so phase 0 is left touchdown.** Your walk/jog left-swing peaks at 0.73 vs 0.28 of stride are a 0.45-stride (about 160°) offset. A fraction-of-stride remap preserves that offset exactly, so every cross-family switch tells the reference to put the wrong foot down.
- **Integrate phase and ramp the period.** Use φ += dt/T with T ramped over the transition, and show the frequency to the actor.
- **Commit family switches at the next left touchdown,** with about 0.1 m/s of hysteresis around 1.5 and 2.5 m/s.
- **Replace the 0.4 s linear cross-fade.** Within a family, use phase-registered interpolation of neighbouring clips. Across families, use contact-event time-warping or an inertialization offset decay.
- **Train on dense boundary-crossing command sequences,** weighted toward slowing down. Training on switches is not optional: in one G1 study, hard expert switching succeeded 5–65% of the time, against 70–95% with transition randomization (preprint) ([RPG](https://arxiv.org/html/2604.21355v2)).

For consolidation, one PPO policy over the merged library is a sound first bet, given your walk+jog success. It needs a reward-gated (vx, wz) grid curriculum that never drops jog cells. The stand-still collapse you saw matches the "jitter in place" failure of uniform wide-range sampling ([Rapid Locomotion](https://ar5iv.arxiv.org/html/2205.02824)). If that stalls, add a soft mixture-of-experts actor gated on twist and clock. Distilling specialists is the fallback, and it tends to transition more smoothly. The 3.5 m/s target sits at the edge of documented G1 hardware running. Your library tasks also run on a flat torque-clip actuator model with no torque–speed derating, so the sprint band is where the simulator is most likely to be optimistic.

*Evidence tags used below: **peer-reviewed**, **preprint** (arXiv, not yet peer-reviewed), **company claim**, and **news**. "Inference" marks reasoning in the research notes rather than a published result. "Repo check" marks something I verified in your code for this report.*

## Every smooth gait library shares one phase origin and switches at foot events

Model-based gait libraries get continuity at switches by construction, and the five design choices they share map almost one-to-one onto your setup's known issues:

- **Shared phase.** Every gait uses the same normalized phase variable, and usually the same step duration.
- **Interpolation, not snapping.** Coefficients are interpolated between neighbouring gaits instead of jumping to the nearest one.
- **Event timing.** The interpolation parameter is updated once per step, at impact.
- **Feedback indexing.** The library is often indexed by filtered measured state rather than the raw command.
- **Planned large changes.** Big changes use explicit transition gaits or a graph of allowed switches with a minimum dwell time.

On Cassie, Gong et al. used seven gaits from −0.5 to 1.0 m/s, all with "a constant" step time. They linearly interpolated the gaits by the robot's *filtered sagittal velocity*, not by the command (peer-reviewed, hardware) ([Gong et al. 2019](https://arxiv.org/pdf/1809.07279)). On MARLO, Da et al. interpolated Bézier coefficients bilinearly, extracted "at each step". They added one-step and three-step transition controllers that converge in three steps, with the learned policy needing at most one more (peer-reviewed, hardware) ([Da et al. 2017](https://grizzle.robotics.umich.edu/files/XingyeDa_ICRA2017.pdf)). The notes found no published system, model-based or RL, that switches mid-stride between periodic references of different periods on a raw command.

### A 0.45-stride offset turns every remap into a wrong-foot landing

Model-based libraries fix foot identity structurally. Li et al.'s 1331-gait Cassie library defines each gait as "2 steps, referred to as right stance and left stance". Transitions between the steps "are triggered by a foot impact" (peer-reviewed) ([Li et al. 2021](https://arxiv.org/pdf/2103.14295)). In RL clocks, phase zero always means a named foot:

- Walk These Ways uses per-foot offsets of a single global timer ([WTW](https://arxiv.org/html/2212.03238)).
- HugWBC ties the right foot to the left with a fixed offset ψ ([HugWBC](https://arxiv.org/html/2502.03206v3)).
- unitree_rl_gym's G1 uses `phase_right = phase + 0.5` ([unitree_rl_gym](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/legged_gym/envs/g1/g1_env.py)).

The notes found **no paper that deals with gait families whose stride starts on opposite feet**, because no published system allows that state.

Your walk family peaks left swing at 0.73 of stride and jog at 0.28, so the two families are offset by about 0.45 of a stride. A fraction-preserving remap keeps the phase fraction, and with it this 160° error. At the switch, the reference asks the robot to swing the foot it is standing on. *Inference (notes):* detect each clip's left-touchdown frame k_LTD from its contact flags. Then either circularly shift the stored clip by −k_LTD, or store φ_offset = k_LTD / n_clip and sample at (φ + φ_offset) mod 1. After that, φ = 0 means left touchdown in every family, and a fraction-preserving remap also preserves foot identity. Moving the wrap point puts the clip's seam mid-array. Your walk/jog clips are not exactly periodic, so the seam-velocity and finite-difference handling must follow the new seam index.

The same reasoning gives a free consistency check for turning. *Inference (notes):* for a left-right symmetric robot, the −wz clip should approximately equal the mirror of the +wz clip shifted by half a stride, mirror(clip(vx, +wz))(φ + 0.5). The mirror swaps left and right joints and negates roll and yaw. You can generate only wz ≥ 0 and mirror the rest, which guarantees consistent foot order. Alternatively, the residual between the optimized −wz clip and the mirrored +wz clip measures asymmetry in your trajectory optimization. For wz = 0 clips, the same test checks left/right symmetry. Check this per family, including run_fwd and sprint_fwd, which the notes did not inspect. The pivot family's 0.85 s stride matches jog. With a common origin, translation↔pivot blending reduces to the in-family recipe, but only if the pivots actually alternate left and right like the translating gaits. That still needs verifying.

### Integrate phase and ramp the period instead of remapping frames

Every multi-frequency RL paper in the notes keeps one integrated phase accumulator, φ += f·dt, wrapped to [0, 1). Changing the frequency then changes only the rate, never the current value:

- **Walk These Ways** increments a global timer by f_cmd/f_control. Frequency and gait switch "in quick succession", and hardware tracked f_cmd alternating between 2 and 4 Hz (peer-reviewed, Go1) ([WTW](https://arxiv.org/html/2212.03238)).
- **HugWBC** integrates φ₁ += f·dt over a curriculum range of [1.5, 3.5] Hz (peer-reviewed, H1) ([HugWBC](https://arxiv.org/html/2502.03206v3)).
- **Yang et al.** update the gait command at 20 Hz "to avoid abrupt changes of gait commands" (peer-reviewed) ([Yang et al. 2021](https://arxiv.org/pdf/2104.04644)).

*Inference (notes):* remapping a per-clip frame index by fraction of stride is mathematically the same as integrating φ with a *step* change in T. With clips of 43, 37 and 32 frames, the step is about 16% at jog→run and again at run→sprint. Because your actor sees only sin/cos of the phase, that rate step is a hidden discontinuity in the observation.

*Repo check:* your `BlendingMotionCommand` (`src/mjlab/tasks/gait_library/mdp/blending.py`) plays the outgoing clip at n_prev/n_cur frames per step during a blend. The phase rate therefore jumps to the target clip's rate the moment the switch starts, while the served pose is still almost entirely the old clip. The module docstring also states that "the commanded twist still steps at the resample", so the actor's twist input jumps too. `_phase_remap` in `src/mjlab/tasks/gait_library/mdp/commands.py` rounds to the nearest integer frame, adding up to half a frame of phase error at each switch.

The integrated alternative has the same cost and fixes all three problems:

- Keep a continuous φ per env.
- Advance it as φ += dt / T_blend(t), with T_blend ramped over the same window as the poses.
- Sample every clip at φ with periodic interpolation between frames, so frame rounding disappears.
- Ramp the commanded twist seen by the actor over the same window.

For robustness after pushes, Da et al. used a phase observer, τ̇ = 1/T + (L/T)(s(q) − τ). It pulls time-based phase toward a mechanical phase s(q), with gain L = 0 below 0.4 m/s and L = 7v above, so phase is nearly purely state-based at speed. Da et al. call it "a Luenberger observer for the phase" (peer-reviewed, hardware) ([Da et al. 2016](https://grizzle.robotics.umich.edu/files/Da_AccessMay2016.pdf)). *Inference (notes):* a contact-driven version of this correction would keep your reference and robot in step after a disturbance or a switch.

### Switch at left touchdown, with hysteresis and a dwell

"Update at impact" is how the model-based libraries keep reset maps and constraints consistent. Veer et al. formalized the minimum dwell between switches among limit cycles, and used it to weight a graph of allowed switches (peer-reviewed, simulation) ([Veer et al. 2017](https://arxiv.org/pdf/1703.07197)). DeepMimic changes its skill goal only at cycle boundaries (peer-reviewed) ([DeepMimic](https://ar5iv.labs.arxiv.org/html/1804.02717)). The only G1 multi-gait paper in the notes switches through explicit transitional modes:

- Run is enabled when the Froude number v²/(g·l) exceeds 0.5.
- Run→walk is held for a hand-set 2.5 s.
- Walk→stand requires low speed plus 1.5 s of double support "to prevent premature switching".

(Preprint; running shown only in simulation) ([Peng et al. 2025](https://arxiv.org/html/2505.20619v3)).

*Inference (notes):* commit a family change only when the command crosses the boundary by about Δvx = 0.1 m/s, and then latch it at the next left touchdown, where φ ≡ 0 in both families and the remap is exact. Without hysteresis, snap-to-nearest chatters when a joystick rests near 1.5 or 2.5 m/s. That is exactly where your families share a boundary column, so two clips with different periods exist at the same twist. Gong's choice to index by filtered *measured* speed suggests a further guard. Index the reference by a low-passed blend of commanded and measured twist, so it cannot run ahead of the robot during a step command. Sudden yaw commands during a pivot are the case where waiting up to a stride is undesirable. For those, NaviGait's mid-step splice offers an alternative (described below).

Transitions are asymmetric, and the notes recommend treating slowing down as the harder direction. Veer et al. needed **one switch and about 12 s to speed up from 0.42 to 0.81 m/s, but 12 switches and about 70 s to slow down** ([Veer et al. 2017](https://arxiv.org/pdf/1703.07197)). Murali and Ames found optimized one-step connecting surfaces saved **36.8–57.1% of the cost metric on speed-ups but only 11.6–14.9% on slow-downs** (peer-reviewed, simulation) ([Murali & Ames 2019](http://ames.caltech.edu/murali2019optimal.pdf)). *Inference (notes):* sprint→run→jog deserves longer blends, more dwell and more training samples than the upward direction.

### Interpolate within a family, time-warp across families, inertialize the rest

The libraries that work interpolate in parameter space:

- Gong and Da interpolate linearly or bilinearly.
- NaviGait takes "a convex combination of the Bézier coefficients of the three nearest gaits" (preprint, BRUCE hardware) ([NaviGait](https://arxiv.org/html/2510.11542v1)).
- Li et al.'s RL policy tracks a reference "constructed by interpolating the parameterized gait library". They claim better stability during transitions than an HZD baseline, though they report no transition metric (peer-reviewed, Cassie hardware) ([Li et al. 2021](https://arxiv.org/pdf/2103.14295)).

Every interpolated hardware library in the notes shares one step period across all gaits. Li et al.'s 2024 walking library fixes it at 0.8 s ([Li et al. 2024](https://arxiv.org/html/2401.16889v1)). The notes found **no model-based paper interpolating running gaits of different periods on hardware, and no RL paper that time-warps clips of different periods**. Your design is in uncharted territory here.

Character animation supplies the missing tool. Kovar and Gleicher's registration curves blend clips in three steps. First, time-warp them to corresponding frames. Second, align their root frames. Third, match footplant constraints. Blending then happens in the registered time (peer-reviewed) ([Registration Curves](https://research.cs.wisc.edu/graphics/Gallery/kovar.vol/RegistrationCurves)). Linear blending "produces reasonable results when the input motions are sufficiently similar", and fails otherwise ([course summary](https://research.cs.wisc.edu/graphics/Courses/777-s2006/TextWeb/Summaries/Week05/bryden-regcurves.txt)).

*Inference (notes):* within a family, replace snapping with a phase-registered blend of the 3–4 clips around the commanded (vx, wz). The weights w_i sum to 1, the reference is q_ref(φ) = Σ w_i·q_i(φ), and phase advances as φ += dt·Σ w_i/T_i. Across a family boundary, a uniform warp is valid only if footstrikes fall at the same stride fraction in both clips. Run and sprint have flight phases, and jog may not, so their duty factors differ. In that case, use a piecewise-linear time-warp keyed on left touchdown, left liftoff, right touchdown and right liftoff. It can be precomputed offline for each boundary pair (jog@1.5 ↔ run@1.5, run@2.5 ↔ sprint@2.5).

Your 0.4 s window is the second issue. It covers **47% of a jog stride and 63% of a sprint stride**, so it is shorter than one stride. A linear cross-fade between misaligned poses also kinks the reference velocity at both ends. *Repo check, plus this report's inference:* your blend already adds the ramp's own velocity term α̇·(x_b − x_a), so the served velocity matches the served pose. But α̇ jumps from 0 to 2.5 s⁻¹ at the start of the blend and back to 0 at the end, which makes reference acceleration impulsive at both ends. A smoothstep (C¹) α costs one line and removes this.

The animation industry's standard alternative to cross-fading is **inertialization**. At the switch, record the pose and velocity offset between the source and target. Then output target + offset, with the offset decayed by a polynomial or a critically damped spring, so only one clip is ever evaluated ([Bollo, SIGGRAPH 2017](https://history.siggraph.org/wp-content/uploads/2022/09/2017-Talks-Bollo_High-Performance-Animation-in-Gears-of-War-4.pdf); [Holden, spring roll call](https://theorangeduck.com/page/spring-roll-call)). DReCon pairs this with an RL tracker. Its motion-matching features include foot positions and velocities, so a match must have the same foot configuration (peer-reviewed) ([DReCon](https://theorangeduck.com/media/uploads/other_stuff/DReCon.pdf)).

When the switch must happen mid-step, NaviGait splices the current and target splines at the current phase τ(t1). Early coefficients come from the current gait, middle ones are averaged, and late ones come from the target. The remaining phase is renormalized as τ̂ = (t − t1)/(T − (t1 − t0)), which keeps position and velocity continuous ([NaviGait](https://arxiv.org/html/2510.11542v1)). **The notes found no published half-life or blend duration for robot reference blending.** Your 0.4 s is unvalidated either way, and choosing it is an experiment, not a lookup.

### Generate transition clips where gait topology changes

Interpolating poses is only valid when both clips have the same contact topology. Where they do not, the literature generates dedicated transitions:

- Da et al. optimized one-step and three-step transition gaits between double-support states ([Da et al. 2017](https://grizzle.robotics.umich.edu/files/XingyeDa_ICRA2017.pdf)).
- Murali and Ames chained one-step connecting surfaces (e.g. 0.15→0.20→0.25→0.30 m/s). These beat an optimized speed ramp, and optimized ramps in turn beat a linear ramp by 5.9–28.4% ([Murali & Ames 2019](http://ames.caltech.edu/murali2019optimal.pdf)).
- Li et al. 2024 used a separately retargeted running-to-standing clip as the transition reference, and taught standing→running→standing in one policy ([Li et al. 2024](https://arxiv.org/html/2401.16889v1)).

*Inference (notes):* if jog lacks flight where run has it, or if pivots step in a different order than translation, generate a two-step, non-periodic trajectory-optimization clip in mj-nlp. It would run from the end state of clip A's stride to the start state of clip B's stride. **Fast pivots (|wz| 1–2 rad/s) are the most likely to need this**, because their contact pattern differs most from translation.

## Policies only transition well when they train on transitions

A smooth reference is necessary but not sufficient. The policy must see many switches during training. RPG built a G1 policy from frozen per-skill experts and trained it with random transitions and random frame jumps. Transition success over 20 trials per pair rose from 0.05 to 0.75 for Jump→Kick, 0.35 to 0.80 for Punch→Jump, and 0.30 to 0.70 for Kick→Jump (preprint; hardware results qualitative only) ([RPG](https://arxiv.org/html/2604.21355v2)). Li et al. resampled Cassie's command every 8 s in roughly 83 s episodes ([Li et al. 2021](https://arxiv.org/pdf/2103.14295)). Their 2024 work re-randomized commands after random 1–15 s intervals inside 76 s episodes, so the robot would "experience transitions among tasks" (peer-reviewed) ([Li et al. 2024](https://arxiv.org/html/2401.16889v1)). DeepMimic sampled a new skill at every cycle boundary so that all pairwise transitions were seen ([DeepMimic](https://ar5iv.labs.arxiv.org/html/1804.02717)).

*Inference (notes):* resample the twist every 1–8 s, but deliberately oversample sequences that cross a boundary: jog↔run near 1.5 m/s, run↔sprint near 2.5 m/s, and translation↔pivot. Mix ramps with steps, because joysticks produce both. Weight downward transitions more heavily. Run the adaptive sampler over (source cell → target cell) pairs, not just over cells, so failure-prone transitions get extra samples. Cap command acceleration at deployment and in training, as SPRINT does (value not given) ([SPRINT](https://arxiv.org/html/2605.28549)), or pass the joystick through a critically damped spring. That way the reference cannot jump from sprint to pivot in one control step.

Reward design in the transition window matters too, and the evidence is consistent if mostly qualitative:

- **Action-change penalty on a curriculum.** Li 2024 penalizes joint acceleration and action change. The action-change weight is stronger for running, starts low "to avoid overly conservative behavior", and rises gradually ([Li et al. 2024](https://arxiv.org/html/2401.16889v1)).
- **Action-rate and smoothness weights.** RuN uses an action-rate penalty of −0.04 and an action-smoothness penalty of −0.06 for G1 walk-to-run (preprint) ([RuN](https://arxiv.org/html/2509.20696)).
- **Behaviour-specific transition penalties.** Siekmann et al. found that varying gait offsets and ratios together produced "a fusion of all the different commanded gaits". It took penalties active only while a given behaviour was commanded to separate them (peer-reviewed) ([Siekmann et al. 2021](https://arxiv.org/pdf/2011.01387)).

*Inference (notes):* add a contact-schedule term that rewards the stance foot matching the reference's stance foot at the same φ. It penalizes wrong-foot landings directly. Optionally widen the tracking σ during a blend, KungfuBot-style ([KungfuBot](https://arxiv.org/html/2506.12851v3)), so the critic does not punish the policy for missing a reference that is itself in motion.

### A transition benchmark that scores the reference before the policy

No standard transition metric exists, and **no published humanoid gait-library paper reports quantitative hardware transition metrics**. The G1 gait-conditioned paper and NaviGait report none ([Peng et al. 2025](https://arxiv.org/html/2505.20619v3); [NaviGait](https://arxiv.org/html/2510.11542v1)). The notes assemble a benchmark from scattered precedents. Run it in simulation across many envs, for every boundary in both directions and at several switch phases, over a window from one stride before to three strides after the switch (Murali's five-step window).

| Metric | Precedent | Why it matters for you |
|---|---|---|
| Fall/termination rate per boundary and direction | RPG success rate ([RPG](https://arxiv.org/html/2604.21355v2)) | Headline number; expect the downward direction to be worse |
| Mean joint acceleration within ±50 steps of the switch | RPG ([RPG](https://arxiv.org/html/2604.21355v2)) | The only smoothness metric with published numbers |
| vx/wz settling time to ±10% and overshoot | Da 2017 ([Da et al.](https://grizzle.robotics.umich.edu/files/XingyeDa_ICRA2017.pdf)) | Captures speed-change quality |
| Wrong-foot touchdowns, extra/missed steps, stride-duration CV | Stride CV from DeepTransition ([Shafiee et al.](https://arxiv.org/pdf/2306.07419)); wrong-foot count is proposed | Directly detects the phase-origin bug |
| Mean body angular velocity, peak torque/limit, peak GRF and its rate | DeepTransition oscillation proxy ([Shafiee et al.](https://arxiv.org/pdf/2306.07419)) | Hardware risk during the transition |
| Windowed cost of transport vs the mean of the two steady states | Murali & Ames ([2019](http://ames.caltech.edu/murali2019optimal.pdf)) | Compares transition strategies fairly |
| Max per-frame jump in the *reference's* joint position and FD velocity at a switch | Proposed in the notes | Offline, no training needed; isolates the discontinuity the reference design injects |

The last row should come first. *Inference (notes):* it compares snapping, your current 0.4 s lerp, phase-registered blending and inertialization on the reference alone, before any GPU time is spent. Your `view_reference` ghost viewer is the natural place to inspect the worst cases visually. No accepted "smooth" threshold exists, so set thresholds relative to steady-state stride-to-stride variability.

## Show the actor the clock frequency it currently cannot see

Five clock families appear in the literature:

- **Fixed-frequency.** humanoid-gym; unitree_rl_gym's G1 with a 0.8 s period.
- **Integrated commanded frequency.** Walk These Ways, HugWBC.
- **Policy-modulated rate.** Lee 2020, CPG-RL, Yang 2021.
- **Learned latent phase.** FLD/DeepPhase.
- **Phase-free.** Fu 2021 energy-only; Li 2024 reference preview.

([humanoid-gym](https://raw.githubusercontent.com/roboterax/humanoid-gym/main/humanoid/envs/custom/humanoid_env.py); [WTW](https://arxiv.org/html/2212.03238); [HugWBC](https://arxiv.org/html/2502.03206v3); [CPG-RL](https://arxiv.org/abs/2211.00458); [FLD](https://arxiv.org/pdf/2402.13820); [Fu et al. 2021](https://arxiv.org/abs/2111.01674v1); [Li et al. 2024](https://arxiv.org/html/2401.16889))

Your sin/cos(2πt/n_clip) is a variable-frequency clock whose rate the actor can only infer by differencing frames. *Inference (notes):* adding f = 1/T (or n_clip) as one more observation is the closest published analogue (WTW and HugWBC observe or command frequency). It also removes an ambiguity specific to your reference-free actor. Like KungfuBot's proprioception-plus-phase actor ([KungfuBot](https://arxiv.org/html/2506.12851v3)), it needs (twist, phase) to identify the target posture uniquely. Near a family boundary during a blend, the same twist and phase can correspond to different postures. DeepMimic's one-hot skill selector solved the analogous problem ([DeepMimic](https://ar5iv.labs.arxiv.org/html/1804.02717)), and the frequency scalar does the same job continuously.

Your frequency span is modest by published standards. Stride frequency runs from 1.18 to 1.59 Hz, a 1.35× ratio. Walk These Ways covers 1.5–4.0 Hz (2.7×) in one hardware policy ([WTW](https://arxiv.org/html/2212.03238)), and HugWBC covers 1.5–3.5 Hz (2.3×) on H1 ([HugWBC](https://arxiv.org/html/2502.03206v3)). *Inference (notes):* the hard part of your problem is the 0.5–3.5 m/s speed range and the flight phase, not the frequency range.

Two cautions temper heavy clock structure at the top end. First, Walk These Ways measured a cost to imposing gait structure. Gait-free sprint tracking scored 0.94, trot 0.92, and other gaits 0.83–0.88, and the parameterization was "restrictive for high combined linear and angular velocity" ([WTW](https://arxiv.org/html/2212.03238)). Second, human runners get speed mostly from stride length up to about 7 m/s. Above that, "the strategy … shifted to … increasing stride frequency" (peer-reviewed) ([Dorn et al. 2012](https://pubmed.ncbi.nlm.nih.gov/22573774/)).

WTW and HugWBC both sample frequency independently of speed during training. *Inference (notes):* jittering T a few percent around each family's schedule buys some of that robustness without a full independent sweep. A weaker contact-schedule reward in the sprint band, or a small bounded policy residual on f, would recover some top-speed freedom. Per-foot sine clocks (Siekmann, WTW, HugWBC) and a global sin/cos with a fixed 0.5 offset carry the same information for a two-beat biped gait. Keep your global sin/cos, now anchored at left touchdown.

## One PPO run is viable if the curriculum grows only from mastered cells

The consolidation literature points one way: naive single policies on merged, heterogeneous skills underperform, and gating or distillation recovers most of the gap.

- **BumbleBee (peer-reviewed, NeurIPS 2025).** On MuJoCo sim-to-sim success, a single general policy scored **33.01%**, experts on a random partition 35.36%, and the generalist distilled from behaviour-clustered experts **66.84%** ([BumbleBee](https://arxiv.org/html/2506.12779v3)).
- **Robot Parkour Learning (peer-reviewed).** The distilled student reached 86/80/100/73/100 across five skills, against 95/82/100/100/100 for the specialists. A memoryless MLP student mostly failed ([Parkour](https://ar5iv.labs.arxiv.org/html/2309.05665)).
- **MoE-Loco (preprint).** A soft mixture of experts scored 0.879 on a mixed course against 0.571 for a parameter-matched MLP. It also raised the cosine similarity between task gradients, the measured symptom of interference ([MoE-Loco](https://arxiv.org/html/2503.08564v2)).
- **CoRe-MoE (preprint, G1 walk/run to 2.5 m/s).** One network on merged walk+run data produced a "shuffling" intermediate gait. A single network with velocity-selected references reached 54.20% success against 99.13% for a velocity-gated MoE. The comparison is confounded by added terrain experts ([CoRe-MoE](https://arxiv.org/html/2606.04718)).

Two findings bear directly on your transition priority. Barkour's distilled generalist scored slightly below hand-switched specialists but "transitions between gaits more smoothly" ([Barkour](https://arxiv.org/abs/2305.14654v1)). Fu et al. likewise obtained smooth gait changes from a velocity-conditioned policy distilled from gait experts ([Fu et al. 2021](https://ar5iv.labs.arxiv.org/html/2111.01674)). Hard switching between specialists is the policy-level equivalent of nearest-clip snapping. Whatever you consolidate into should blend, not switch.

The counter-evidence is just as relevant. Rapid Locomotion reached **3.9 m/s on Mini Cheetah hardware with one MLP-plus-history policy**, with no experts, because of a grid adaptive curriculum (peer-reviewed) ([Rapid Locomotion](https://ar5iv.arxiv.org/html/2205.02824)). Your earlier walk+jog success fits this pattern. *Inference (notes):* try options in this order:

1. A single PPO policy with a grid curriculum, failure-weighted sampling and the frequency input.
2. A soft mixture-of-experts actor, gated on twist, clock and proprioception, as a drop-in replacement in the same run.
3. Specialists distilled with DAgger, with the teacher chosen by command bin. Budget for an RL fine-tune of the student, because pure DAgger was reported insufficient for highly dynamic humanoid skills ([Perceptive Humanoid Parkour](https://arxiv.org/html/2602.15827v1)).

For option 3, the student needs memory, and rsl-rl already ships `Distillation` with `StudentTeacherRecurrent` ([IsaacLab distillation cfg](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab_rl/rsl_rl/distillation_cfg.html)). Your phase clock already differs by family, which reduces the risk of an MSE student averaging two teachers at the 1.5 and 2.5 m/s boundaries.

### The stand-still collapse is a curriculum failure, not a speed limit

Your observation that policies collapse to standing still when the speed curriculum widens too fast has a clear published analogue. Uniform sampling over vx ∈ [−4, 4] and ωz ∈ [−5, 5] failed to learn, and the policy "jitters in place" with tracking error equal to the command ([Rapid Locomotion](https://ar5iv.arxiv.org/html/2205.02824)). TransCurriculum reports that "starting with a wide initial range … destabilizes learning" (preprint) ([TransCurriculum](https://arxiv.org/html/2603.14156v1)). Every high-speed paper that ablates the curriculum finds it essential:

- SPRINT's G1 peak dropped from **5.96 to 4.48 m/s** without it ([SPRINT](https://arxiv.org/html/2605.28549)).
- The G1 gait-conditioned policy without a curriculum managed an episode length of 31 against 973 with one ([Peng et al. 2025](https://arxiv.org/html/2505.20619)).

Rapid Locomotion's grid works as follows. Commands are sampled from a joint (vx, ωz) grid at 0.5 m/s × 0.5 rad/s resolution. When a command among the hardest unlocked ones is tracked above a reward threshold, its 4-connected neighbours unlock. Probabilities are never removed. A box-shaped curriculum "excluded the extremes", because running and turning together are harder than either alone ([Rapid Locomotion](https://arxiv.org/pdf/2205.02824)).

*Inference (notes):* align the grid cells to your library's twist grid. Seed it at the jog cells. Let the in-place pivot cells (vx ≈ 0, |wz| 1–2) unlock along their own path while straight running unlocks along vx. Keep a floor of standing and slow-jog envs: Chasing Stability keeps 2% standing ([Chasing Stability](https://arxiv.org/html/2509.19573)), and Chasing Autonomy starts 40% of envs from a walking reference to learn walk-to-run ([Chasing Autonomy](https://arxiv.org/html/2603.25902)). Start sprint episodes from lower-speed library states so acceleration through the boundaries gets learned.

For sampling inside the unlocked region, mjlab's BeyondMimic-style adaptive sampler bins each motion into one-second slices. It uses an EMA of failures, a 0.1/S uniform floor and a smoothing kernel (`src/mjlab/tasks/tracking/mdp/commands.py`) ([BeyondMimic](https://arxiv.org/html/2508.08241)). Your clips are 0.64–0.86 s long, so every clip collapses to a single bin. *Inference (notes):* re-key the sampler per (vx, wz) cell, and per transition pair. Keep the uniform floor so jog is not forgotten. Spread failure mass to neighbouring command cells, since difficulty at sprint 3.4 m/s implies difficulty at 3.2. GMT's completion levels, which tighten termination thresholds only after a clip is mastered, keep early sprint failures from dominating ([GMT](https://arxiv.org/html/2506.14770)).

## 3.5 m/s is the edge of documented G1 running, and your sim is optimistic there

The most transferable result is Caltech's "Chasing Autonomy" (preprint). Its setup matches yours closely: the same robot, a 50 Hz PD-position policy, and a trajectory-optimized gait library spanning 1.2–3.6 m/s, with a reference-free actor that observes sin/cos phase while the critic gets the reference. It reached **3.3 m/s on a treadmill** and sustained outdoor runs. It also reports that "dynamic optimization without human data did not converge above 3 m/s", which is a direct warning for your 2.5–3.5 m/s sprint library ([Chasing Autonomy](https://arxiv.org/html/2603.25902)). The earlier no-human-data version showed hardware robustness at about 2.2 m/s and "had low time to overheat" ([Chasing Stability](https://arxiv.org/html/2509.19573); [Chasing Autonomy](https://arxiv.org/html/2603.25902)).

Other G1-class figures:

- **SPRINT (preprint)** claims 5.91 m/s measured at a 6 m/s command on G1 hardware. The measurement method is unstated, and no independent replication was found ([SPRINT](https://arxiv.org/html/2605.28549)).
- **Noetix-affiliated preprint.** A 1.36 m, 40 kg humanoid averaged 4.55 m/s over 100 m, timed from video ([Push the Limit](https://arxiv.org/html/2610.14840)).
- **Unitree's G1 spec (company claim)** lists 2 m/s ([Unitree G1](https://www.unitree.com/g1)).

Full-size 7–10 m/s claims (H1, MirrorMe Bolt) involve redesigned hardware or company-only measurement and do not carry over to a stock G1 ([Humanoids Daily](https://www.humanoidsdaily.com/news/unitree-h1-reclaims-speed-record-with-blistering-10-m-s-sprint)). Library saturation is a further cap. A 2 m/s command to the Cassie library policy produced only about 1.2 m/s ([Li et al. 2021](https://ar5iv.labs.arxiv.org/html/2103.14295)). *Inference (notes):* your sprint ceiling is effectively the top of the sprint library unless the tracking reward is relaxed at the extreme. SPRINT only extrapolated to about 6 m/s from a 3.40 m/s top clip by making its prior continuous in frequency.

At top speed, the binding constraint is torque at the hip and knee:

- Cassie's hip-roll motors saturate above about 3 m/s ([Cassie 100 m](https://arxiv.org/html/2508.03070v1)).
- Noetix's robots ran hip-pitch and knee at about 99% of their torque limit and ankle-pitch at about 90%. Push recovery fell from 100% at 5.5 m/s to 56% backward and 72% lateral at 6.5 m/s ([Push the Limit](https://arxiv.org/html/2610.14840)).
- Booster set simulated torque and velocity limits from the knee points of measured torque–speed curves ([Booster Lab](https://arxiv.org/html/2606.27813v1)).

*Repo check:* the walk/jog unicycle library tasks inherit `G1-Tracking-Custom` (`src/mjlab/tasks/tracking/config/g1_custom/env_cfgs.py`), which a gunning task would presumably reuse. That config sets the articulation to `G1_MODE11_ARTICULATION`, built from the same `BuiltinPositionActuatorCfg` entries as the stock G1, so actuators have a flat `effort_limit` and **no torque–speed derating**. It adds a per-episode constant actuator delay of 0–40 ms. Your mode-11 table moves hip pitch onto the 7520-22 (139 Nm, 20 rad/s in the constants, vs 88 Nm and 32 rad/s for the stock 7520-14). Hip pitch, hip roll and knee therefore all carry a nominal 20 rad/s that the simulator never enforces. Sprint is the band with the fastest hip-pitch swing, so this is where sim most likely flatters the policy.

mjlab does ship `DcMotorActuatorCfg`, a linear torque–speed limit on top of an explicit PD (`src/mjlab/actuator/dc_actuator.py`), and a `BuiltinDcMotorActuator` with back-EMF. Switching from implicit built-in position actuators to explicit PD changes the integration, so validate it before trusting it. Alternatively, check offline whether your trajectory-optimization clips' peak joint velocities and torques already respect these limits.

The randomization and reward practices credited for high-speed transfer are specific:

- **Randomization.** Caltech found friction and PD-gain randomization "very helpful", with over-damped BeyondMimic gains ([Chasing Autonomy](https://arxiv.org/html/2603.25902)). Noetix randomized kp/kd 0.8–1.2×, joint friction and armature 0.7–1.5×, and actuator delay 0–20 ms ([Push the Limit](https://arxiv.org/html/2610.14840)).
- **Rewards.** Noetix used a motor-power² penalty (−5e-3) against thermal and torque limits. SPRINT used speed-gated contact penalties: −6.0 for both feet airborne below 1.5 m/s and −3.0 for both feet grounded above 3.0 m/s ([SPRINT](https://arxiv.org/html/2605.28549)).

*Inference (notes):* gate these penalties on your band boundaries rather than tuning per-gait rewards. Log per-joint torque/limit and velocity/limit ratios across a speed sweep before pushing the sprint band. Plan short sprint bouts on hardware.

## Implementation plan: fix the reference, then the clock, then the training

The ordering puts cheap, offline, reference-side fixes first. Their effect can be measured with benchmark metric 7 before any training. Training-side and architectural changes come after. File paths are in your repo.

| # | Change | Where | Evidence basis | Done when |
|---|---|---|---|---|
| 1 | **Canonicalize the phase origin.** Detect the left-touchdown frame per clip and circularly shift (or store φ_offset) so φ = 0 is left touchdown in walk, jog, run, sprint and pivot. Move seam-velocity handling to the new seam. Optionally generate −wz clips by mirroring +wz with a half-stride shift, and log the mirror residual. | `src/mjlab/scripts/library_to_npz.py`, `src/mjlab/tasks/gait_library/library.py` (`LIBRARY_SPECS`), rebuild via `build_library` | Structural in all HZD/RL clocks; no paper addresses the mismatch (inference) | Left-swing peak at the same φ (±0.03) in every family; mirror residual small |
| 2 | **Build the transition benchmark, reference-only first.** Measure the max per-frame jump in joint position and FD velocity at switches for snap, current lerp, and each new scheme, at all boundaries, both directions, several switch phases. | New script beside `view_reference.py` | Proposed in notes; RPG for later policy metrics | Baseline numbers for the current 0.4 s lerp recorded |
| 3 | **Integrated continuous phase.** Per-env float φ advanced by dt/T_blend. Sample clips at φ with periodic interpolation instead of integer frames, retiring `_phase_remap` rounding. Ramp T and the actor-visible twist with the blend. Use a smoothstep α. | `src/mjlab/tasks/gait_library/mdp/commands.py`, `src/mjlab/tasks/gait_library/mdp/blending.py` | WTW, HugWBC, Yang (peer-reviewed) | No step in phase rate or twist observation at a switch; metric 7 improves |
| 4 | **Add frequency f = 1/T to the actor** (and critic). Jitter T ±3–5% around the family schedule in training. | `src/mjlab/tasks/gait_library/env_cfgs.py` (`make_actor_reference_free`) | WTW, HugWBC; DeepMimic skill ID (inference) | Obs dim +1; no regression on walk+jog |
| 5 | **Event-latched family switches with hysteresis.** Commit when the command crosses 1.5 or 2.5 m/s by about 0.1 m/s, then switch at the next left touchdown. Optional dwell of 1+ stride. Index by low-passed commanded/measured twist. | `BlendingMotionCommand._resample_reference`, sampler in `src/mjlab/tasks/jog_unicycle/mdp/commands.py` | Gong, Da, Veer (peer-reviewed); Peng (preprint) | No chatter with the command held at 1.50 m/s; zero wrong-foot touchdowns in the benchmark |
| 6 | **Interpolate instead of snapping within a family.** Phase-registered barycentric/bilinear blend of neighbouring clips, with T interpolated too. | Library command | Cassie, MARLO, NaviGait | Reference continuous in (vx, wz); metric 7 near zero for in-family moves |
| 7 | **Cross-family blending.** Contact-event piecewise time-warp precomputed per boundary pair. Compare against inertialization (offset decayed by a critically damped spring, try 0.2–0.4 s and one full stride). | Library build + command | Registration curves, inertialization, DReCon (inference to robots) | Pick the scheme with the lowest metric-7 jump and lowest policy-level joint acceleration |
| 8 | **Transition clips where topology differs** (pivot↔translate first, then jog↔run if jog lacks flight). Two-step non-periodic trajectory optimization from stride end of A to stride start of B. | mj-nlp, library format | Da 2017, Murali & Ames, Li 2024 | Only if 6–7 leave a residual spike at that boundary |
| 9 | **Transition-rich command schedule.** Resample every 1–8 s in long episodes. Oversample boundary crossings and downward transitions. Mix steps and acceleration-limited ramps. Adaptive sampling over cells and (source→target) pairs with a uniform floor. | Sampler; re-key `src/mjlab/tasks/tracking/mdp/commands.py` adaptive logic | Li 2021/2024, RPG, DeepMimic | Every episode has 3+ switches; per-pair failure rates converge |
| 10 | **Grid curriculum over (vx, wz).** Seed at jog, 4-neighbour unlock on a tracking threshold, never remove cells. Keep 2–5% stand and a slow-jog floor. Start some sprint episodes from run states. | Command sampler / curriculum term | Rapid Locomotion (peer-reviewed), SPRINT, Peng | No stand-still collapse; sprint cells unlock without jog regressing |
| 11 | **Transition-window rewards.** Action-rate/joint-accel penalty with a rising weight. Stance-foot-matches-reference-at-φ term. Widened tracking σ during blends. Speed-gated flight/double-support penalties at the band edges. | Task rewards (`jog_unicycle/mdp/rewards.py` lineage) | Li 2024, RuN, SPRINT, Siekmann | Lower joint acceleration ±50 steps around switches |
| 12 | **Actuator realism for the sprint band.** Log per-joint torque and velocity ratios vs the 7520-22's 139 Nm and 20 rad/s across a 0.5–3.5 m/s sweep. Prototype `DcMotorActuatorCfg` derating for hip pitch, hip roll and knee, plus a power penalty. Keep friction/PD/armature DR. | `src/mjlab/asset_zoo/robots/unitree_g1/g1_constants_mode11.py`, `src/mjlab/actuator/dc_actuator.py` | Booster, Noetix, Cassie, Caltech (preprints) | Sprint still tracked with derating on; peak ratios below 1 with margin |
| 13 | **Escalate architecture only on evidence.** If sprint or pivot cells plateau, or a shuffling gait appears at 1.5/2.5 m/s, swap in a soft MoE actor gated on twist and f. If that fails, train per-family specialists and DAgger-distill into a recurrent student with RL fine-tuning. | rsl-rl actor module; rsl-rl `Distillation` | MoE-Loco, CoRe-MoE, BumbleBee, Barkour | Benchmark transition metrics no worse than single policy |

Steps 1–3 unblock everything else: steps 6 and 7 assume a shared phase origin and a continuous φ.

## Conclusion

The research reframes your problem. The notes treat transition smoothness as mostly a *reference-engineering* problem, and only secondarily a policy-learning one. Your actor is reference-free, so the only things it ever sees change at a switch are the clock and the twist. Both currently step discontinuously, and the clock can point at the wrong foot. No amount of PPO tuning or architecture change compensates for a reference that asks for a 160°-out-of-phase stride. Conversely, once phase is canonical, continuous and event-latched, the published evidence suggests a single policy trained on many switches can transition smoothly. The open questions then become empirical ones the literature has not answered. The best blend duration for robot references, and how well cross-period time-warping works for flight-phase running, are both untested in published work. The reference-only benchmark lets you answer them cheaply before you spend training time.

The second takeaway is about where risk concentrates. Every axis of difficulty in the notes converges on the same corner: the downward sprint→run transition, sprint with yaw, and pivot entry and exit. That corner combines the slowest-converging transition direction, the gait-topology change, the trajectory-optimization convergence warning above 3 m/s, and the actuator model's missing torque–speed limits. Spend benchmark coverage, sampler weight and actuator-fidelity work there first, and treat a gunning policy that is smooth up to about 3 m/s as the realistic first milestone.
