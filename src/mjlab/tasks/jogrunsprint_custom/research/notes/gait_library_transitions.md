# Smooth transitions between periodic gaits in gait-library controllers (model-based and RL)

Scope: how controllers built from libraries of one-stride periodic gaits (HZD / trajectory optimization) and RL policies trained on such libraries move smoothly between gaits and speed regimes, and how transition quality is measured. The target is the user's G1 setup: jog, run and sprint families with 43, 37 and 32-frame strides, a pivot family, nearest-clip snapping with a 0.4 s linear blend, and an opposite-foot phase origin in some families. Sibling notes cover phase-clock design in depth (`phase_clocks_multigait.md`), motion-matching and imitation (`motion_library_imitation.md`), and MoE and distillation (`consolidation_distillation.md`). This note avoids repeating them except where needed.

Evidence tags: **[HW]** hardware result, **[SIM]** simulation only, **[QUAL]** qualitative claim with no numbers.

---

## Q1. Model-based gait libraries: how they guarantee continuity at switches

### Takeaway
Model-based gait libraries get smooth transitions from five design choices:
- **Shared phase.** Every gait in the library uses the same phase variable, normalized per step, and usually the same step duration.
- **Interpolation, not snapping.** Controllers interpolate Bézier coefficients between neighbouring gaits instead of jumping to the nearest one.
- **Event-timed updates.** The interpolation parameter is updated once per step, at impact or in double support.
- **Feedback-based indexing.** Several controllers index the library by a filtered measured state rather than the raw command.
- **Large changes are planned.** Big speed changes use explicit one-step or three-step transition gaits, or a graph of allowed switches with a minimum dwell time.

Switching mid-stride on a raw command, as nearest-clip snapping does, is never used.

### Cited Findings
**Cassie gait library: interpolation by filtered speed, constant step time (Gong et al., ACC 2019) [HW]**
- Seven gaits cover v̄x from −0.5 to +1.0 m/s in 0.25 m/s steps. Each gait is a 5th-order Bézier virtual-constraint trajectory h_d(τ, α_i). "The step time for all gaits was chosen to be a constant." The desired output is built "using linear interpolation of a discrete library of gaits" — [Gong et al. 2019, arXiv 1809.07279](https://arxiv.org/pdf/1809.07279)
- "The interpolation parameter is the robot's filtered sagittal velocity". In other words, the library is indexed by measured state, not by the command. The library alone leaves the closed loop unstable, because the (ẋ, ẏ) coordinates act as "approximate integrators". Leg-angle (foot-placement) adjustment is added on top to stabilize speed — [Gong et al. 2019](https://arxiv.org/pdf/1809.07279)

**MARLO (ATRIAS-series) libraries and transition gaits (Da, Hartley, Grizzle, ICRA 2017) [HW]**
- The policy maps features to Bézier coefficients α. The scalar version is linear interpolation, π(φ) = (1−ζ)α_i + ζα_{i+1}, with ζ = (φ−φ_i)/(φ_{i+1}−φ_i), "extended to bilinear interpolation (BiLI) if the feature has two variables". "α and φ are extracted at each step", so the controller is updated once per step — [Da et al. 2017](https://grizzle.robotics.umich.edu/files/XingyeDa_ICRA2017.pdf)
- The speed library has 13 periodic gaits from −1.2 to 1.2 m/s in 0.2 m/s steps (training set) and a 0.05 m/s grid (test set). Gaits minimize squared torque under periodicity and average-speed constraints — [Da et al. 2017](https://grizzle.robotics.umich.edu/files/XingyeDa_ICRA2017.pdf)
- **Dedicated transition gaits:**
  - Between double-support states x_i → x_j, they optimize either one-step transition controllers or three-step ones (x_i → x_a → x_b → x_j). The three-step optimizations target speeds in {−0.4, …, 0.4} m/s, starting from initial speeds within ±0.6 m/s of the target.
  - The learned transition policy uses the features {v_avg, θ_init, v_tgt}: average speed, initial stance-leg angle (an inverted-pendulum state) and target speed.
  - The optimal controllers converge in three steps, and the learned policy needs "at most one more step".
  - After a 200 N push lasting 0.35 s, the transition policy has "shorter settling time and smaller overshoot than the speed policy".

  — [Da et al. 2017](https://grizzle.robotics.umich.edu/files/XingyeDa_ICRA2017.pdf)
- Da et al. survey earlier ways of stitching precomputed controllers together: "switching, finite-state machines, and interpolation". Interpolation has been used both to design transition gaits among constant-speed controllers and to create a continuous family of gaits. A cited online HZD gait generator "updates it at the beginning of the next step" — [Da et al. 2017](https://grizzle.robotics.umich.edu/files/XingyeDa_ICRA2017.pdf). The later journal version adds supervised learning and model structure to embed a low-dimensional model so that walking stays locally exponentially stable — [Da & Grizzle, IJRR 2019, arXiv 1711.02223](https://www.arxiv.org/abs/1711.02223)

**Provably bounded switching among limit cycles (Veer, Motahar, Poulakakis, CDC 2017) [SIM]**
- They generate a continuum of limit cycles by "smoothly modulating the virtual holonomic constraints" of one exponentially stable HZD gait. The outputs are chosen so that the zero-dynamics manifold stays hybrid-invariant under switching, which "greatly facilitat[es] the commute from one limit cycle to another". Switches happen at the step-to-step (impact) level of the Poincaré map — [Veer et al. 2017, arXiv 1703.07197](https://arxiv.org/pdf/1703.07197)
- **Dwell time:** at least N steps must pass before the next switch, with N_{p→q} > ½·log((|ζ*_p − ζ*_q|/ε) + 1)/log(1/δ_z). In practice one can instead "monitor when ζ[k] enters the ε-ball of the target fixed point". The bound is used to weight edges of a graph of allowed switches — [Veer et al. 2017](https://arxiv.org/pdf/1703.07197)
- **Numbers:**
  - 79 limit cycles span 0.42–0.81 m/s, with at most 0.01 m/s between neighbours. All respect 100 Nm torque saturation, friction coefficient below 0.8, and at least 100 N upward ground reaction force.
  - Speeding up from 0.42 to 0.81 m/s takes one switch and about 12 s. Slowing down from 0.81 to 0.42 m/s takes 12 switches and about 70 s.

  — [Veer et al. 2017](https://arxiv.org/pdf/1703.07197)
- Follow-ups: switching among straight, left-turn and right-turn primitives on a 3D biped [SIM] — [Veer & Poulakakis, arXiv 1810.00527](https://arxiv.org/pdf/1810.00527.pdf). A theory of switched systems whose subsystems do not share an equilibrium — [Veer & Poulakakis TAC 2019](https://sushantveer.github.io/publications/papers/VePo2019TAC.pdf)

**Optimal one-step transition surfaces vs a linear ramp (Murali & Ames, CDC 2019) [SIM, 5-DOF planar]**
- **Two transition methods:**
  - (a) Ramp the desired speed v_d(t) on one fixed partial-HZD surface. Because those dynamics are exponentially stable and linear, a linear ramp converges "exponentially fast".
  - (b) Optimize a connecting surface PZ_β(v1,v2) that joins the optimal surfaces of two speeds over a single step. The constraints require that the impact map Δ sends the end of gait v1 onto PZ_β, and the end of PZ_β onto gait v2.

  — [Murali & Ames 2019](http://ames.caltech.edu/murali2019optimal.pdf)
- **Metric:** a cost-of-transport-like index J2 = (1/(m·g·d_total))∫uᵀu dt, taken over 5 steps (1 initial, 3 transition, 1 final). — [Murali & Ames 2019](http://ames.caltech.edu/murali2019optimal.pdf)
- **Results:**
  - Optimized ramps beat the linear ramp by 5.9–28.4%.
  - Chained one-step connecting surfaces (e.g. 0.15→0.20→0.25→0.30 m/s) beat the optimized ramp by 36.8–57.1% for speed-ups and 11.6–14.9% for slow-downs.

  — [Murali & Ames 2019](http://ames.caltech.edu/murali2019optimal.pdf)
- An earlier closed-form version connects two optimized partial-HZD surfaces with an "extended canonical walking function", with coefficients set by boundary conditions [Powell, Hereid & Ames, ICRA 2013]. This is described in [Murali & Ames 2019](http://ames.caltech.edu/murali2019optimal.pdf). The other prior work they summarize:
  - Funnel-based HZD switching [Motahar et al. 2016].
  - An orbit library where each orbit is stabilized by transitions from the others [Da et al. 2016].
  - Primitive transitions on uneven terrain [Manchester & Umenberger 2014].

**Motion-primitive graphs with verified transitions (Ubellacker & Ames) [HW, quadruped]**
- They define "dynamic-state-aware conditions for transitions between primitives". Stand, walk and jump primitives and the transitions between them form a graph that is searched online — [arXiv 2106.10310](https://arxiv.org/pdf/2106.10310), [arXiv 2209.07503](https://arxiv.org/pdf/2209.07503)

**Reduced-order orbit composition (Xiong & Ames) [HW Cassie]**
- The H-LIP model has Period-1 orbits (every step identical) and Period-2 orbits (left and right steps differ). These can be chosen separately for each plane, giving four 3D compositions, which are stabilized by step-to-step velocity feedback on Cassie — [Xiong & Ames, arXiv 1910.00684](https://arxiv.org/pdf/1910.00684)

**Splice-at-current-phase interpolation (NaviGait, Janwani et al. 2025) [SIM + HW, BRUCE humanoid]**
- The library has 39 FROST HZD gaits with degree-7 Bézier curves, indexed by (v_x, v_y). Off-grid commands use "a convex combination of the Bézier coefficients of the three nearest gaits", which is barycentric interpolation — [NaviGait, arXiv 2510.11542](https://arxiv.org/html/2510.11542v1)
- Phase is τ = (t − t0)/T. When the command changes at t1, the current and target splines are spliced at τ(t1). The remaining phase is renormalized as τ̂ = (t − t1)/(T − (t1 − t0)). Early Bézier coefficients come from the current gait, middle ones are averaged, and late ones come from the target, so position and velocity stay continuous mid-step. This is applied recursively. An RL residual (Δq plus a Δv that shifts the library index) runs on top at 50 Hz on hardware [HW: QUAL] — [NaviGait](https://arxiv.org/html/2510.11542v1)

### Inferences
- Every HZD library above shares one normalized phase per step, and the Cassie libraries also share one step time: constant in Gong, fixed at 0.8 s in Li 2024 (Q3). That is why plain coefficient interpolation is phase-consistent there. The user's families have **different periods** (0.85, 0.74, 0.63 s), so interpolating across families needs explicit time normalization. Index every clip by a normalized stride phase φ ∈ [0,1) with a common origin (Q4), interpolate poses at equal φ, and advance φ at a blended rate (Q2).
- "Switch only at impact" is how Veer, Da, Gong and Li 2021 keep reset maps and constraints consistent. For the user's RL reference, the analogue is to commit a family change only at a fixed contact event, for example left touchdown, where φ ≡ 0 in both families. The remap is then exact and needs no fraction-of-stride arithmetic.
- **Asymmetry:** slowing down needs more steps than speeding up (Veer: 12 switches vs 1; Murali: smaller savings on slow-downs). Sprint → run → jog transitions probably deserve longer blends, more dwell and more training samples than the upward direction.
- Gong indexes the library by **filtered measured** speed, not the command. An RL analogue is to index the reference by a low-passed blend of commanded and measured twist, so the reference cannot run ahead of the robot during a step command.

### Gaps
- The full Powell, Hereid & Ames 2013 paper was not fetched; its description comes from Murali & Ames.
- I found no model-based paper that interpolates between **running** gaits of different stride periods on hardware. All interpolated hardware libraries found are walking with a fixed step time.
- I did not obtain Reher & Ames's Cassie compliant-library papers ([arXiv 2010.09047](https://arxiv.org/pdf/2010.09047)), so how they schedule interpolation is unverified.

---

## Q2. Interpolating neighbouring clips vs nearest-clip snapping; phase-space blending of clips with different periods

### Takeaway
Libraries that work in practice interpolate continuously in parameter space: linear or bilinear in the Cassie and MARLO libraries, barycentric over the three nearest gaits in NaviGait. Character animation shows how to blend clips whose periods differ: time-warp them to a common phase first (registration curves), align their root frames, and match their footplant constraints, then blend. Clips that are far apart cannot be blended linearly, and offset-decay methods such as inertialization replace cross-fades.

### Cited Findings
- **Cassie RL on an interpolated HZD library (Li et al., ICRA 2021) [HW]:**
  - 1331 gaits on an 11×11×11 grid: q̇x ∈ [−1, 1] m/s, q̇y ∈ [−0.3, 0.3] m/s, height qz ∈ [0.65, 1.0] m.
  - Each gait uses 5th-order Bézier curves "normalized by 1 step period" and consists of two steps, right stance and left stance, with "transitions between the steps … triggered by a foot impact".
  - Given the desired gait parameter p_d, the reference "is constructed by interpolating the parameterized gait library". The policy sees the current command plus reference motor positions and velocities at t, t+1, t+4 and t+7 control steps (30 Hz), a short preview window.
  - Actions are low-pass filtered before the PD loop.
  - The abstract claims it "improves stability during gait transitions compared to a HZD-based baseline" [QUAL; no transition metric in the paper].

  — [Li et al. 2021, arXiv 2103.14295](https://arxiv.org/pdf/2103.14295)
- **Reduced-order library as reference (Green et al., RA-L 2021) [SIM + HW Cassie]:**
  - Energy-optimal 3D actuated-SLIP periodic walking from 0 to 2 m/s in 0.1 m/s steps, with minimum-jerk swing feet. The library is queried at 33 Hz with the velocity command and a clock.
  - In simulation the policy "smoothly transition[s] between discrete reference walk cycles", with pelvis-velocity oscillations matching the SLIP centre-of-mass oscillations.
  - On hardware the operator "sends the robot sudden changes in velocity commands" [QUAL, video only].
  - An action-difference penalty "results in smoother action outputs".
  - Whether references are interpolated between the 0.1 m/s grid points is not stated.

  — [Green et al. 2021, arXiv 2010.11234](https://arxiv.org/pdf/2010.11234)
- **Registration curves (Kovar & Gleicher, SCA 2003):** a registration curve has three parts: a timewarp curve (a strictly increasing map between corresponding frames, found by dynamic programming), a coordinate-alignment curve, and constraint matches such as footplants. Blending is done in the registered (warped) time, and footplant intervals "may simply be blended as with other motion parameters". Stated limitation: no physical constraints such as balance are enforced — [Registration Curves project page](https://research.cs.wisc.edu/graphics/Gallery/kovar.vol/RegistrationCurves), [paper](https://diglib.eg.org/handle/10.2312/SCA.SCA03.214-224), [course summary](https://research.cs.wisc.edu/graphics/Courses/777-s2006/TextWeb/Summaries/Week05/fliu-regcurves.txt)
- **Linear blending needs similar motions.** It "produces reasonable results when the input motions are sufficiently similar", and naive orientation averaging (slerp) can flip unless frames are aligned first — [registration curves summary](https://research.cs.wisc.edu/graphics/Courses/777-s2006/TextWeb/Summaries/Week05/bryden-regcurves.txt). Motion graphs (SIGGRAPH 2002) build a transition by blending a window of frames from both clips (root linearly interpolated, joints slerped), only between frames that a pose-distance metric finds similar — [Motion Graphs paper](https://perso.liris.cnrs.fr/alexandre.meyer/teaching/master_charanim/papers_pdf/motion_graph.pdf), [CS777 write-up](https://research.cs.wisc.edu/graphics/Courses/777-s2006/TextWeb/P3-Handin/krill-777/p3-writeup.html)
- **Inertialization (Bollo, Gears of War 4, SIGGRAPH 2017 and GDC 2018):** at the transition, record the offset between source and target pose and velocity, then decay that offset with a polynomial. The source clip is no longer evaluated. This is now the industry standard alternative to cross-fading. **Dead blending** instead extrapolates the source with its last velocity and blends — [Bollo talk PDF](https://history.siggraph.org/wp-content/uploads/2022/09/2017-Talks-Bollo_High-Performance-Animation-in-Gears-of-War-4.pdf), [Dead Blending](https://theorangeduck.com/page/dead-blending), [Half-Pound filter paper on inertialization's limitations](https://arxiv.org/html/2602.21702)
- **Local motion phases (Starke et al., SIGGRAPH 2020):** each body part gets its own phase, defined by its contacts and computed automatically from mocap. This handles asynchronous limbs where one global phase would force alignment — [SIGGRAPH history entry](https://history.siggraph.org/?p=78734), [paper PDF](https://www.ipab.inf.ed.ac.uk/cgvu/basketball.pdf). PFNN-style phase-conditioned networks and DeepPhase are covered in the sibling notes.

### Inferences
- **Phase-registered bilinear blend.** Recommended replacement for nearest-clip snapping inside a family:
  - Let the four (or three, barycentric) neighbouring clips around the commanded (vx, wz) have weights w_i, which sum to 1.
  - Define the reference as q_ref(φ) = Σ w_i · q_i(φ), sampling each clip at the same normalized phase φ with periodic interpolation between frames.
  - Advance the phase as φ ← φ + dt · Σ w_i / T_i.

  This is the registration-curve recipe with a trivial uniform time-warp. It works because all clips in a family share the timing structure: the same stride period and the same foot order, once Q4 is fixed.
- **Across a family boundary,** the shared boundary vx column gives two clips at the same twist with different periods, for example jog@1.5 (0.85 s) and run@1.5 (0.74 s). Blend from one to the other at equal φ while ramping the period. A uniform warp assumes each clip's footstrikes sit at the same fraction of the stride. If the duty factor differs between families (run and sprint have flight phases, jog may not), use a **piecewise-linear time-warp keyed on contact events**: left touchdown, left liftoff, right touchdown, right liftoff. This is exactly what registration curves' constraint matching does, and it can be precomputed offline per boundary pair.
- **0.4 s blend window.** It covers about 47% of a jog stride and about 63% of a sprint stride, so the blend window is shorter than one stride. A linear cross-fade between misaligned poses gives a velocity kink at both ends. Two options:
  - (a) Use an inertialization-style decaying offset, q_ref = q_target(φ) + offset·decay(t), so reference velocity is continuous.
  - (b) Blend over exactly one stride, from event to event.
- **NaviGait's splice** (keep the current gait's early Bézier coefficients, average the middle, use the target's late ones) is a mid-step alternative when waiting for a touchdown event is undesirable, for example a sudden wz command during a pivot.

### Gaps
- I found no published RL paper that interpolates clips of **different stride periods** with explicit time-warping. Li 2021 and Li 2024 use one fixed period across the library, and Green 2021's period handling is unstated.
- No paper quantifies snapping vs interpolation for an RL policy's transition smoothness.

---

## Q3. RL-side transition smoothing: command schedules, transition-rich training, rewards, clocks

### Takeaway
RL papers make transitions smooth by:
- Changing commands frequently during long episodes.
- Rate-limiting or filtering commands and actions.
- Penalizing action rate and joint acceleration, with the weight raised gradually.
- Adding transition-specific reward terms or transitional modes with dwell times and hysteresis (Siekmann, Peng).
- Driving one integrated phase whose rate, not value, changes.

None of the papers reports a dedicated transition-smoothness metric for a humanoid on hardware.

### Cited Findings
- **Li et al. 2021 (Cassie) [HW]:** a new command is "uniformly sampled every 8 s, and remains unchanged during the 8 s window". Episodes last 2500 steps (about 83 s), there is no ramp, and actions are low-pass filtered — [arXiv 2103.14295](https://arxiv.org/pdf/2103.14295)
- **Li et al. 2024 IJRR (Cassie, walking, running, jumping) [HW]:**
  - **References:** walking uses 1331 trajectory-optimized references, all with a **fixed 0.8 s period**. Running uses a single retargeted human clip (3 m/s, about 0.1 m foot clearance), and a separately retargeted running-to-standing human clip serves as the transition reference.
  - **Command schedule:** commands are re-randomized after random 1–15 s intervals inside 76 s episodes, so the robot "experience[s] transitions among tasks".
  - **Smoothness terms:** rewards penalize joint acceleration and action change r(a_t, a_{t+1}). The action-change weight is stronger for running and jumping, starts low "to avoid overly conservative behavior", and is raised gradually.
  - **Tracking:** hardware command-tracking MAE is about 0.10 m/s sagittal, 0.08–0.10 m/s lateral and 0.05–0.06 m height.

  — [Li et al. 2024, arXiv 2401.16889](https://arxiv.org/html/2401.16889v1)
- **Siekmann et al. ICRA 2021 (Cassie) [HW]:**
  - Holding phase offsets fixed while varying swing ratios gives walk-to-run transitions.
  - Varying both offsets and ratios produced asymmetric walking or "a fusion of all the different commanded gaits".
  - The fix was **transition penalties**: behaviour-specific costs that are active only while that behaviour is commanded, such as feet-together for hopping or a standing penalty.

  — [Siekmann et al. 2021, arXiv 2011.01387](https://arxiv.org/pdf/2011.01387)
- **Peng, Bao & Zhou 2025, Unitree G1 [running and walk↔run SIM only; walk, stand and walk-to-stand HW]:**
  - Gait IDs: run when the Froude number v²/(g·l) > 0.5. Run→walk (R2W) is a separate transitional ID held for a "manually specified" 2.5 s. Walk→stand (W2S) needs low speed plus double support for 1.5 s to "prevent premature switching".
  - Commands update at 10 Hz and ramp 0→3 m/s in simulation. One-hot gait-ID reward routing is used.
  - Transition rewards include "smooth slowdown" and "transition contact pattern".
  - There is no transition metric; the only claim is "transitions occur smoothly without abrupt contact shifts" [QUAL].

  — [arXiv 2505.20619v3](https://arxiv.org/html/2505.20619v3)
- **Yang et al. CoRL 2021 (A1) [HW]:** a high-level gait policy sets per-foot phase frequency and offsets for an MPC. Walk → trot → fly-trot transitions emerge with speed, up to 2.5 m/s, from an energy-minimization reward — [PMLR v164](https://proceedings.mlr.press/v164/yang22d.html). The gait command is updated at 20 Hz "to avoid abrupt changes of gait commands", as noted in the sibling note — [arXiv 2104.04644](https://arxiv.org/pdf/2104.04644)
- **Walk These Ways (Go1) [HW]:** one global timing variable increments by f_cmd/f_control each step, and per-foot timings are fixed offsets of it. Frequency and gait can therefore switch "in quick succession", because frequency changes alter only the phase rate — [arXiv 2212.03238](https://arxiv.org/html/2212.03238)
- **Fu et al. CoRL 2021:** gaits emerge from energy minimization. A velocity-conditioned policy distilled from gait-specific experts gives smooth gait changes as target speed changes — [arXiv 2111.01674](https://ar5iv.labs.arxiv.org/html/2111.01674). Barkour's distilled generalist scores slightly below specialist switching but "transitions between gaits more smoothly" [HW] — [arXiv 2305.14654](https://arxiv.org/abs/2305.14654v1)
- **Shao et al. RA-L 2022 [HW quadruped]:** four per-foot phase signals from a gait generator give walk/trot/pace/bound transitions — [arXiv 2201.00206](https://arxiv.org/abs/2201.00206v1)
- **NaviGait [SIM + HW QUAL]:** rewards penalize residual size and rate (Δv, Δq), and the RL residual can shift the library index — [arXiv 2510.11542](https://arxiv.org/html/2510.11542v1)

### Inferences
- **Training schedule for the user:** resample twist every 1–8 s, as in Li 2021 and 2024, but deliberately oversample **boundary-crossing** sequences: jog↔run at vx ≈ 1.5, run↔sprint at about 2.5, and translation↔pivot. Mix step commands with ramps, because deployment joysticks give both. Weight the downward transitions more heavily (Q1 asymmetry).
- **Hysteresis and dwell.** Snapping on a band boundary chatters when the command sits near 1.5 or 2.5 m/s. Make a family switch need the command to cross the boundary by a margin, Δvx ≈ 0.1, and then wait for the next left touchdown. This is Veer's dwell-time idea combined with Peng's "prevent premature switching".
- **Phase rate.** Use an integrated phase φ += dt / T_blend(t), with T blended over the same window as the poses, and expose the frequency 1/T to the actor (see `phase_clocks_multigait.md`). With an event-synchronized switch at φ = 0, the remap becomes trivial.
- **Rewards for the transition window:** action rate and joint acceleration with a curriculum on the weight (Li 2024). A contact-schedule term that requires the stance foot to match the reference's foot at the same φ penalizes wrong-foot landings directly.

### Gaps
- No RL paper quantifies the effect of the command-ramp duration or the reference-blend duration on transition smoothness.
- Li 2024's Appendix B on transition training was not read; the page was too long to fetch fully.

---

## Q4. Foot-phase conventions across families; mirroring

### Takeaway
Model-based libraries fix foot identity structurally. The Cassie libraries (Li 2021) define each gait as two steps, right stance then left stance, with impact-triggered step changes. Planar HZD gaits are one-step orbits with a leg relabeling at impact. In RL clocks, φ = 0 always means a specific named foot. No paper found handles families with different start feet. Circularly shifting the clips (or adding 0.5 to the phase) of opposite-foot families offline, so that φ = 0 is left touchdown everywhere, is the direct fix. Sagittal mirroring with a half-stride shift turns a +wz clip into a −wz clip.

### Cited Findings
- Cassie library gaits "consist of 2 steps, referred to as right stance and left stance, and transitions between the steps are triggered by a foot impact" — [Li et al. 2021](https://arxiv.org/pdf/2103.14295)
- H-LIP distinguishes Period-1 orbits (each step identical, typical in the sagittal plane) from Period-2 orbits (left and right steps differ, required in the lateral plane). Its theorem labels initial states by stance leg, e.g. "the initial state of SSP with right leg as the…" — [Xiong & Ames, arXiv 1910.00684](https://arxiv.org/pdf/1910.00684)
- Siekmann's per-foot clocks offset the left and right feet by θ_left and θ_right, with walking/running "shifted apart by half of the period length". Walk These Ways and HugWBC similarly anchor timing to a named foot (sibling note) — [Siekmann et al. 2021](https://arxiv.org/pdf/2011.01387), [WTW](https://arxiv.org/html/2212.03238)
- Registration curves match footplant constraints across motions before blending, which makes foot identity part of the alignment — [Registration Curves](https://research.cs.wisc.edu/graphics/Gallery/kovar.vol/RegistrationCurves)

### Inferences
- **Fix for the known issue.** Families where the left-swing peak sits at 0.73 instead of 0.28 of the stride are offset by about 0.45 (close to 0.5). Detect each clip's left touchdown frame k_LTD from its contact flags. Store the clip circularly shifted by −k_LTD, or store φ_offset = k_LTD/n_clip and sample at (φ + φ_offset) mod 1. Every family then has φ = 0 at left touchdown and a left-swing peak near the same φ, and a fraction-preserving remap keeps foot identity.
  - Caveat from the user's own clips: walk and jog clips are not exactly periodic. Shifting moves the wrap seam to mid-array, so any finite-difference velocity or seam handling must use the new seam index.
- **Mirroring.** For a symmetric robot, mirror(clip(vx, wz))(φ) ≈ M·clip(vx, −wz)(φ + 0.5), where M swaps left and right joints and negates roll and yaw joints. Applying this to a +wz clip gives the −wz clip with the correct foot order. Generating only wz ≥ 0 and mirroring guarantees consistency, and the residual |clip(vx,−wz) − mirror(clip(vx,wz))| is a cheap check for asymmetry in the optimization. For wz = 0 clips this also tests left/right symmetry, i.e. the Period-1 vs Period-2 question.
- **Pivot family.** Its 0.85 s stride matches jog, so with a common φ origin, translation↔pivot blending reduces to the same in-family recipe. Check, though, whether the pivots' contact pattern (stepping order around the turn) actually alternates left and right like the translating gaits.

### Gaps
- I found no paper that discusses misaligned phase origins between gait families explicitly; the fix above is inferred.
- The exact mirror sign conventions for G1 joints need checking in the user's code, not in the literature.

---

## Q5. Metrics for transition quality

### Takeaway
No standard transition metric exists. Papers use:
- Settling time and overshoot after a speed step or push (Da 2017).
- Cost of transport integrated over a fixed window spanning the transition (Murali 2019).
- Number of steps or dwell to reach the target orbit (Veer 2017).
- Feasible and safe command sets (Li 2021).
- Success rate, CoT, stride-duration variability and body angular velocity (DeepTransition).
- Command-tracking MAE (Li 2024).

Most RL humanoid papers report transitions qualitatively only.

### Cited Findings
- **Settling time and overshoot** of speed after a 200 N, 0.35 s push: the transition policy beats the speed-library policy. Also, steps to converge (≤3 optimal, ≤4 learned) [HW and SIM] — [Da et al. 2017](https://grizzle.robotics.umich.edu/files/XingyeDa_ICRA2017.pdf)
- **Windowed cost of transport:** J2 = ∫uᵀu dt /(m·g·d_total) over 5 steps (1 before, 3 transition, 1 after), so different transition controllers can be compared fairly [SIM] — [Murali & Ames 2019](http://ames.caltech.edu/murali2019optimal.pdf)
- **Transition duration in steps and time:** dwell-time bound per edge; 0.42→0.81 m/s in about 12 s, 0.81→0.42 m/s in about 70 s [SIM] — [Veer et al. 2017](https://arxiv.org/pdf/1703.07197)
- **Feasible command set vs safe set:** commanded parameters that do not cause a fall in 15 s, vs the parameters actually achieved. The RL set was more than 4× larger than HZD's in SimMechanics — [Li et al. 2021](https://arxiv.org/pdf/2103.14295)
- **DeepTransition (Shafiee et al., Nature Communications 2024) [SIM + HW quadruped]:**
  - CoT vs speed and the energetically optimal transition speed, where the walk and trot CoT curves cross.
  - Coefficient of variation of stride duration and length; it peaks near the transition speed in some animals.
  - Divergent component of motion (DCM) offset.
  - "Average angular velocity of the body" as a smoothness and oscillation proxy.
  - Gap-crossing success rate (100% over 2400 attempts for the best reward weighting).

  — [arXiv 2306.07419](https://arxiv.org/pdf/2306.07419)
- **Hardware tracking MAE across variable commands:** about 0.10 m/s sagittal, measured over more than a year — [Li et al. 2024](https://arxiv.org/html/2401.16889v1)
- **Contact smoothness** as a bounded score of the foot-force rate of change [from search snippet only, not verified] — [ROOT, arXiv 2610.04250](https://arxiv.org/pdf/2610.04250)
- Peng 2025 (G1) and NaviGait report no transition-specific metric — [arXiv 2505.20619v3](https://arxiv.org/html/2505.20619v3), [arXiv 2510.11542](https://arxiv.org/html/2510.11542v1)

### Inferences
Proposed transition benchmark for the user's G1. Run it in simulation over many environments. Test command steps and ramps across every family boundary and in both directions: jog↔run, run↔sprint, jog↔pivot, plus wz sign flips. Log a window from 1 stride before to 3 strides after the switch, matching Murali's 5-step window. Metrics:
1. **Falls and terminations** within the window, as a rate per boundary and direction.
2. **Velocity response:** base vx and wz settling time to within ±10% of the command, overshoot, and peak |Δv_COM| between consecutive strides (Da-style).
3. **Contact-sequence integrity:**
   - Count wrong-foot touchdowns, i.e. consecutive same-foot touchdowns or a stance foot disagreeing with the reference's foot at the same φ.
   - Measure stride-duration CV in the window relative to steady state (DeepTransition).
   - Count extra or missed steps.
4. **Smoothness:** peak and RMS joint acceleration, action rate |a_t − a_{t−1}|, peak joint torque relative to its limit, peak vertical GRF and its rate, and mean |ω_body| (DeepTransition oscillation proxy).
5. **Tracking spikes:** peak and integrated joint and keypoint tracking error against the blended reference, normalized by its steady-state value.
6. **Energy:** windowed CoT, Σ|τ·q̇| dt /(m·g·distance), relative to the mean of the two steady-state CoTs.
7. **Reference smoothness, checked on the reference itself before any training:** the max per-frame jump of the blended reference in joint position and in finite-difference velocity at a switch, compared across snapping, 0.4 s linear blend, phase-registered blend and inertialization. This isolates how much discontinuity the reference design injects, independent of the policy.

Of these, metric 7 is a quick offline check of whether the opposite-foot fix and phase-registered blending remove the discontinuity at its source.

### Gaps
- No published humanoid paper reports quantitative transition metrics on hardware for gait-library-based RL.
- No accepted threshold exists for when a transition counts as "smooth"; any threshold has to be set relative to steady-state variability.
