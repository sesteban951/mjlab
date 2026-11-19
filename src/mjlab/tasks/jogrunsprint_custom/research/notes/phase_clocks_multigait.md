# Phase/clock design and multi-gait (multi-frequency) conditioning in a single RL locomotion policy

Context for the reader: the target is one Unitree G1 "fast locomotion" policy at 50 Hz covering
jog (0.85 s stride, about 1.18 Hz stride, 2.35 steps/s), run (0.74 s, about 1.35 Hz, 2.70 steps/s),
sprint (0.63 s, about 1.59 Hz, 3.17 steps/s) and fast turning (0.85 s). These conversions are
arithmetic on the user's numbers, not sourced. The current actor sees sin/cos(2*pi*t/n_clip)
normalized per clip. At 50 Hz this is 42.5, 37 and 31.5 control steps per stride, so the clip
lengths are not integer multiples of dt.

## Q1. Phase/clock representations: which papers use which, and the trade-offs

### Takeaway
Five families are in use: (a) fixed-frequency clocks (humanoid-gym, unitree_rl_gym); (b) a clock
whose rate is a commanded/observed frequency, integrated each step (Walk These Ways, HugWBC);
(c) a phase rate set by the policy or a higher level (PMTG-style Lee 2020, CPG-RL, Yang 2021);
(d) learned latent phase (DeepPhase/PAE, FLD); (e) phase-free designs (Fu 2021 energy-only; Li 2024
uses a reference preview in place of a clock; upstream mjlab velocity task). Per-foot clocks with
explicit left/right offsets are the most common form in multi-gait work. They make gait timing and
foot ordering explicit in the observation, at the cost of imposing a gait on the policy.

### Cited Findings
**(a) Fixed-frequency clock**
- humanoid-gym: `phase = episode_length_buf * dt / cycle_time` (a single global `cycle_time` from config). The observation gets `sin(2*pi*phase)` and `cos(2*pi*phase)` plus the 3-D velocity command ("5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)"). The stance mask is `stance[:,0] = sin_pos >= 0`, `stance[:,1] = sin_pos < 0`, and both feet count as stance when `|sin_pos| < 0.1` (double support). The reward `feet_contact_number` gives +1 when contact matches the mask and -0.3 otherwise. Clearance uses the swing mask — [humanoid-gym humanoid_env.py](https://raw.githubusercontent.com/roboterax/humanoid-gym/main/humanoid/envs/custom/humanoid_env.py)
- unitree_rl_gym G1: `period = 0.8`, `offset = 0.5`, `phase = (t % period)/period`, `phase_right = (phase + offset) % 1`. The observation gets sin/cos of the global phase with zero noise. The contact reward has `is_stance = leg_phase < 0.55` and rewards `~(contact XOR is_stance)` — [unitree_rl_gym g1_env.py](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/legged_gym/envs/g1/g1_env.py)
- Siekmann et al. 2021 (Cassie): per-foot clock `p = {sin(2*pi*(phi + theta_left)/L), sin(2*pi*(phi + theta_right)/L)}`, where L is the number of timesteps per cycle. There is no cosine term. phi "repeatedly cycles over a normalized time period of [0,1]". The swing ratio r is also observed. Policy at 40 Hz, PD at 2 kHz. No numeric L or cycle-time range is given in the paper — [arXiv 2011.01387v2 HTML](https://arxiv.org/html/2011.01387v2)

**(b) Frequency-parameterized clock, phase integrated from a commanded frequency**
- Walk These Ways (Margolis & Agrawal, CoRL 2022, Go1): a global timing variable t increments by `f_cmd / f_pi` per control step (f_pi = 50 Hz control), wrapping 0 to 1. Per-foot timings are `t_FR = t+theta2+theta3`, `t_FL = t+theta1+theta3`, `t_RR = t+theta1`, `t_RL = t+theta2`. The observation is `[sin(2*pi*t_FR), sin(2*pi*t_FL), sin(2*pi*t_RR), sin(2*pi*t_RL)]` (sine only), with a 30-step history. The 8-D behavior command is [theta1..3, f_cmd, body height, pitch, stance width, swing height]. f_cmd ranges over **1.5–4.0 Hz** — [arXiv 2212.03238](https://arxiv.org/html/2212.03238)
- HugWBC (Xue et al., RSS 2025, Unitree H1): `phi1 += f*dt`, `phi2 = phi1 + psi`, wrapped to [0,1). psi = 0.5 for walking and 0 for jumping. Standing has constant phi = 0.25. For hopping, the flying leg is held at phi = 0.75 and the other leg cycles. The clocks are `sin(2*pi*phi_bar)` per foot, where phi_bar is a "homogenized" phase that maps stance to [0,0.5] and swing to [0.5,1] using the duty cycle. Gait frequency defaults to 2 Hz with a curriculum range of **[1.5, 3.5] Hz**. The duty is fixed at 0.5 for every gait — [arXiv 2502.03206v3](https://arxiv.org/html/2502.03206v3)

**(c) Phase rate set by the policy or a higher-level controller**
- Lee et al. 2020 (ANYmal, Science Robotics): each leg phase advances at a shared base frequency f0 plus a policy-output per-leg offset f_i. The base frequency is 1.25 Hz per course slides. Four PMTG-style foot trajectory generators map phase to foot targets, and the policy adds residuals. Blind controller, hardware — [arXiv 2010.11251](https://arxiv.org/pdf/2010.11251); [UMich slides](https://web.eecs.umich.edu/~stellayu/teach/2023action/slides/2020leeQuadrupedalLocomotionSlides.pdf)
- CPG-RL (Bellegarda & Ijspeert, RA-L 2022, A1 hardware): the policy directly modulates the intrinsic amplitude and frequency of per-limb oscillators. There is no explicit coupling, so coordination emerges through learning — [arXiv 2211.00458](https://arxiv.org/abs/2211.00458)
- AllGaits (Bellegarda, Shafiee, Ijspeert, ICRA 2025): DRL plus CPGs, with gaits selected by changing the oscillator coupling matrix and the pattern-formation parameters. The authors claim all gaits and transitions work "without reward function tuning or any expert demonstrations". Hardware covers 9 animal gaits. They also find trot is not the lowest-COT gait — [arXiv 2411.04787](https://arxiv.org/pdf/2411.04787)
- Yang et al. 2021 (A1): a high-level gait policy (evolution strategies, 20 Hz) outputs stepping frequency (bounded in (0, 4] Hz), swing ratio and phase offsets to a phase integrator. A low-level MPC runs at 500 Hz. The paper frames frequency as a speed vs energy trade-off — [arXiv 2104.04644](https://arxiv.org/pdf/2104.04644)

**(d) Learned / latent phase**
- FLD (Li, Stanger-Jones, Heim, Kim, ICLR 2024): extends the Periodic Autoencoder (PAE/DeepPhase). A differentiable FFT layer gives frequency, amplitude and offset per latent channel, and phase comes from an FC layer plus atan2. FLD adds multi-step latent forward prediction by unrolling the phase. PAE is the zero-step special case. Evaluated with PPO on the MIT Humanoid (in Isaac Gym) — [arXiv 2402.13820](https://arxiv.org/pdf/2402.13820); [project page](https://sites.google.com/view/iclr2024-fld/home)
- A later humanoid bi-level motion imitation paper uses a latent dynamics model "inspired by FLD" — [arXiv 2410.01968](https://arxiv.org/pdf/2410.01968)
- Shao et al. 2022 (RA-L, Black Panther quadruped, hardware): four independent per-foot phase signals are the interface between a gait generator and the RL policy. The robot does walk, trot, pace and bound with transitions, and rhythmic "dancing" from general phase patterns — [arXiv 2201.00206](https://arxiv.org/abs/2201.00206v1)

**(e) Phase-free / implicit phase**
- Li et al. 2024 (Cassie, IJRR) has no explicit clock input. The policy instead sees a *preview of the reference* (desired motor positions at t+1, t+4, t+7), the command and I/O histories (5-step short history, 66-step ~2 s long history through a 1D CNN). Separate policies are trained per skill: walking uses a 1,331-motion gait library with a fixed 0.8 s period, running uses one ~3 m/s mocap reference, and jumping is its own policy — [arXiv 2401.16889](https://arxiv.org/html/2401.16889)
- Fu et al. 2021 (A1, CoRL): no gait prior, only energy minimization plus velocity tracking. Walk, trot and a "bounce" (gallop-like, with flight phase) emerge as speed increases — [CoRL PMLR](https://proceedings.mlr.press/v164/fu22a.html); [arXiv 2111.01674](https://arxiv.org/abs/2111.01674v1)
- Upstream mjlab velocity task: a local grep of `src/mjlab/tasks/velocity` found no phase/period/gait variable, so the upstream velocity task is phase-free. This is a local repo observation, at /home/amberlab/projects/mjlab/src/mjlab/tasks/velocity.

**Trade-offs that sources state**
- Walk These Ways: imposing gait structure reduces in-distribution flat-ground sprint tracking. The gait-free baseline scores 0.94, trot 0.92, and other gaits 0.83–0.88. The parameterization is also "restrictive for high combined linear and angular velocity". In return, structured gaits used less power than the gait-free baseline at 0–3 m/s — [arXiv 2212.03238](https://arxiv.org/html/2212.03238)
- HugWBC: command independence shrinks at high speed. Swing height is the least accurately tracked command — [arXiv 2502.03206v3](https://arxiv.org/html/2502.03206v3)

### Inferences
- The user's per-clip normalized clock, sin/cos(2*pi*t/n_clip), is a variable-frequency clock whose rate the actor *cannot see* from one frame. Two adjacent frames show the angular step, but nothing in the observation states it. Walk These Ways and HugWBC both make frequency an explicit command. Adding f (or 1/n_clip) as an observation is the closest published analogue and costs one dimension.
- Sine-only per-foot clocks (Siekmann, WTW, HugWBC) and sin/cos of a global phase plus a fixed 0.5 offset (unitree_rl_gym, humanoid-gym) carry the same information for a 2-beat biped gait. The per-foot form generalizes to hops and skips with nonzero offsets.

### Gaps
- No primary-source ablation was found comparing "frequency observed" against "frequency hidden" in a variable-period clock.
- DeepPhase (Starke et al., SIGGRAPH 2022) was not fetched directly. Its description here comes from the FLD paper.

## Q2. Siekmann periodic reward composition, Walk These Ways, Cassie versatile, and how each varies stride frequency with speed

### Takeaway
None of these papers learns a speed-to-frequency map. Siekmann conditions on swing ratio and
offsets. WTW and HugWBC treat frequency as a free *user command*, sampled independently of
velocity during training. Li 2024 uses a fixed 0.8 s walking period and a single 3 m/s running
reference. WTW reports empirically that higher frequency is needed to track high speeds.

### Cited Findings
- **Siekmann 2021 reward:** each component is `R_i = c_i * I_i(phi) * q_i(s)`, and the policy trains on the expectation. Phase boundaries are random variables `A_i ~ VonMises(2*pi*a_i, kappa)` and `B_i ~ VonMises(2*pi*b_i, kappa)`, so `P(A<phi<B) = P(A<phi)(1-P(B<phi))`. Swing penalizes foot force (c=-1) and stance penalizes foot velocity (c=-1) — [arXiv 2011.01387v2](https://arxiv.org/html/2011.01387v2)
- Siekmann gaits by offset: hop |theta_L - theta_R| ≈ 0, walk/run ≈ 0.5, gallop in between, skip uses 4 phases, and stand has swing ratio ≈ 0. Normal locomotion uses swing/stance ratios in [0.35, 0.7]. The multi-gait policy varies offsets and ratios during training and transitions between all 2-beat gaits and standing. This needed extra "transition penalties" and a hop-symmetry term. Training used 150M samples and 24–36 h per policy — [arXiv 2011.01387v2](https://arxiv.org/html/2011.01387v2)
- Siekmann hardware (Cassie) results are qualitative: hopping on/off a sidewalk, running over bumps, running down 5 stairs. No speed, frequency or success numbers are given — [arXiv 2011.01387v2](https://arxiv.org/html/2011.01387v2)
- **Walk These Ways** contact reward: C_foot is built from normal-CDF terms approximating Siekmann's Von Mises schedule. Weights are swing-force tracking -0.08, stance foot-velocity -0.08, swing-height -0.6, Raibert-heuristic stance width -0.2. The total is `r_task * exp(0.02 * r_aux)` so progress reward stays positive. Velocity curriculum expands to v_x ∈ [-3,3] m/s and w_z ∈ [-5,5] rad/s. On hardware, gait switching tracked as f_cmd alternated between 2 and 4 Hz. Restricting to 2 Hz makes high speeds harder to reach (their Fig. 10) — [arXiv 2212.03238](https://arxiv.org/html/2212.03238)
- WTW limitation: a human pilot picks the gait and frequency, and automating that selection is left to future work — [arXiv 2212.03238](https://arxiv.org/html/2212.03238)
- **Li et al. 2024 (Cassie):** walking uses a fixed 0.8 s period across v_x ∈ [-1, 1] m/s. Running is a separate policy with one ~3 m/s reference. Hardware: 100 m in 27.06 s and 400 m in 2:34, against a prior work's 24.73 s 100 m. A single policy handles standing-to-skill and back transitions. Walk-to-run is not addressed — [arXiv 2401.16889](https://arxiv.org/html/2401.16889)
- **HugWBC:** the velocity curriculum reaches v_x ∈ [-0.6, 2.0] m/s and w ∈ [-1, 1] rad/s, and gait frequency is an independent command in [1.5, 3.5] Hz. Running is *not* trained. A footnote claims it could come from walking "using high speed and a small duty cycle", but this is not demonstrated. Gait is fixed per episode, with no runtime switching described. On H1 hardware: standing, walking, jumping — [arXiv 2502.03206v3](https://arxiv.org/html/2502.03206v3)

### Inferences
- The user's frequency span is 1.18–1.59 Hz stride, a 1.35x ratio. WTW covers 1.5–4 Hz (2.7x) in one policy on hardware, and HugWBC covers 1.5–3.5 Hz (2.3x) on H1. A single policy spanning the user's range is well within demonstrated territory. The harder part is the speed range (0.5–3.5 m/s) and the flight phase, not the frequency range.
- Both WTW and HugWBC sample frequency independently of speed during training. This yields a policy that is robust to off-schedule (f, v) pairs, at a training-cost penalty. A speed-tied schedule (f = g(v), like the user's clip families) is a narrower distribution. It should train faster but will generalize less to off-schedule frequency.

### Gaps
- Appendix C of Li 2024, which gives the running command range, was not read.
- Siekmann gives no numeric cycle-time range.

## Q3. Speed-dependent stride frequency: biomechanics and RL scheduling or emergence

### Takeaway
In humans, stride frequency rises only modestly with running speed up to about 7 m/s, where
speed comes mostly from stride length. Above about 7 m/s, frequency becomes the main lever.
In RL, gait transitions emerge when frequency and gait are free and energy is penalized (Fu 2021),
or when a high-level policy chooses the frequency (Yang 2021). No humanoid paper found learns
frequency as a function of speed.

### Cited Findings
- Dorn, Schache & Pandy 2012 (J Exp Biol 215:1944): up to 7 m/s, the ankle plantarflexors raise support forces and therefore stride length. Above 7 m/s, "the strategy used to increase running speed shifted to the goal of increasing stride frequency", with hip flexors/extensors driving faster leg swing — [PubMed 22573774](https://pubmed.ncbi.nlm.nih.gov/22573774/); [JEB PDF](https://cob.silverchair.com/jeb/article-pdf/215/11/1944/1445059/1944.pdf)
- Weyand et al. 2000, as summarized by a secondary source: 29 runners showed no significant stride-frequency change from 2 to 4 m/s, and only about a 20% increase up to 8 m/s. A 2020 Frontiers study finds stride frequency rising linearly and duty factor falling quadratically with speed — [Frontiers in Physiology 2020](https://www.frontiersin.org/articles/625557) (secondary summary of Weyand; original J Appl Physiol 89(5) not retrieved)
- Fu et al. 2021: energy minimization alone produced walk → trot → bounce as commanded speed rose. Energy-vs-speed was analyzed at 0.375, 0.9 and 1.5 m/s in sim, and the curves are qualitatively similar to horse data. Rough terrain gave unstructured gaits. Deployed on A1 via RMA online adaptation — [CoRL PMLR](https://proceedings.mlr.press/v164/fu22a.html); [arXiv 2111.01674](https://arxiv.org/abs/2111.01674v1)
- Yang et al. 2021: the learned gait policy outputs walk → trot → fly-trot as speed increases, and uses less energy than baselines. Higher frequency allows faster running, but "stepping unnecessarily fast can cause additional energy consumption due to excessive leg swing" — [arXiv 2104.04644](https://arxiv.org/pdf/2104.04644)
- AllGaits studies per-speed COT across gaits and where transitions should occur — [arXiv 2411.04787](https://arxiv.org/pdf/2411.04787)
- Peng et al. 2025 (G1) uses a *rule-based* transition: Run mode starts when the Froude number v^2/(g*l) > 0.5 — [arXiv 2505.20619v3](https://arxiv.org/html/2505.20619v3)

### Inferences
- The user's clips scale stride frequency 1.18 → 1.59 Hz over 0.5 → 3.5 m/s. Human data for 2–4 m/s shows a flatter curve. This is plausible for a small robot (G1 leg length is well under human), since equal Froude number means higher frequency at shorter leg length. It is an inference, not checked against G1 data.
- Since human-style running gets most of its speed from stride length, a design that ties frequency to speed only loosely is biomechanically reasonable. Examples are a small, smooth f(v), or letting the policy modulate f within a band.

### Gaps
- Exact Weyand 2000 stride-frequency numbers in Hz were not retrieved.
- No source found gives a published G1/H1 cadence-vs-speed curve.
- Exact hardware speeds were not retrieved for Fu 2021 and Yang 2021.

## Q4. Gait transitions within one policy: phase continuity, foot ordering, blending

### Takeaway
Every multi-frequency paper found keeps a single integrated phase accumulator, phi += f*dt, wrapped to
[0,1). Changing f then changes only the rate, never the current phase value, so continuity holds by
construction. Foot ordering is fixed by defining per-foot clocks with offsets relative to a named
foot (left/right, or FR/FL/RR/RL). No paper found re-derives phase from a per-clip index.

### Cited Findings
- WTW: a global t increments by f_cmd/f_pi each step, and per-foot timings are fixed offsets of t. Hardware switching between 2 and 4 Hz plus gait changes (trot/pronk/bound/pace) was tracked, and behaviors "can switch in quick succession at high speed" — [arXiv 2212.03238](https://arxiv.org/html/2212.03238)
- HugWBC: phi1 += f*dt, and phi2 = phi1 + psi ties the right foot to the left via psi. Stand and hop pin the clocks to constant values (0.25, and 0.75 for the flying leg) instead of stopping integration — [arXiv 2502.03206v3](https://arxiv.org/html/2502.03206v3)
- Yang 2021: a phase integrator is driven by the policy-chosen frequency, and the gait command is updated at 20 Hz "to avoid abrupt changes of gait commands" — [arXiv 2104.04644](https://arxiv.org/pdf/2104.04644)
- Siekmann 2021: transitions between 2-beat gaits in one policy needed extra transition penalties and a hop symmetry term — [arXiv 2011.01387v2](https://arxiv.org/html/2011.01387v2)
- Peng 2025 (G1): explicit transitional gait IDs (W2S, R2W) with hand-set dwell times (W2S → Stand after 1.5 s of low speed and double support, R2W completes after 2.5 s). The paper also uses gait-conditioned reward routing. Ablation: without routing, episode length is 11.59 and return 0.00. Without the curriculum, 31.26 and 2.45. With both, 972.6 and 89.3. Running was only shown in simulation — [arXiv 2505.20619v3](https://arxiv.org/html/2505.20619v3)
- Shao 2022: per-foot independent phases from a gait generator. Transitions among walk, trot, pace and bound on hardware — [arXiv 2201.00206](https://arxiv.org/abs/2201.00206v1)

### Inferences
- **Integrated phase vs per-clip normalized phase:** remapping t/n_clip by fraction-of-stride at a clip switch is mathematically the same as integrating phi += dt/T(clip) with a step change in T. The phase value is continuous and only its rate jumps. The two designs differ in that (i) the integrated form makes it trivial to expose f to the actor and to blend f smoothly over the 0.4 s window, and (ii) it avoids the n_clip non-integer and wrap-seam issues. With integration, the reference clip is indexed *by* phi rather than phi being derived from the clip frame index.
- **Opposite-foot starts:** in all per-foot clock designs, phi = 0 means "a specific named foot starts stance" (WTW t_FR..., HugWBC phi1 = left, unitree_rl_gym phase_left / phase_right = phase+0.5). The equivalent fix is to define phi = 0 as left touchdown for every family, and to add 0.5 to the phase of any clip family whose stride starts on the right foot when building the library. Phase remap at a switch then preserves foot identity. Without this, a fraction-preserving remap flips which foot the clock says is in stance, which is a 180-degree phase error.
- Blending f over 0.4 s rather than switching it in one step is consistent with Yang's 20 Hz gait-command rate limiting. That is the only stated rationale found for smoothing gait commands.

### Gaps
- No paper found quantifies the effect of instantaneous vs blended frequency changes on stability.
- No paper found addresses opposite-foot reference conventions explicitly. The fix above is an inference from how the clocks are defined.

## Q5. Humanoid-specific multi-gait / walk-run results with clocks

### Takeaway
On humanoids, the hardware evidence for a *single* clock-conditioned policy covers walking, standing
and jumping with variable frequency (HugWBC, H1). Walk/run in one policy on G1 exists only in
simulation (Peng 2025). Bipedal running on hardware (Cassie, Li 2024) used a separate, reference-driven
policy. No source found demonstrates one hardware humanoid policy that spans jog → sprint with a
variable clock.

### Cited Findings
- HugWBC (H1, 19 DoF): standing, walking and jumping share one policy, and hopping needs its own because it is unstable and had the largest errors. Frequency is 1.5–3.5 Hz and v_x reaches 2.0 m/s. Trained about 16 h on one RTX 4090 in Isaac Gym, deployed at 50 Hz. Hardware validated only standing, walking and jumping, with few commands measured — [arXiv 2502.03206v3](https://arxiv.org/html/2502.03206v3)
- Peng et al. 2025 (G1): one LSTM policy (hidden size 64) for stand, walk, run, W2S and R2W. It observes a sine–cosine gait phase plus a one-hot gait ID. No period or frequency is given for the clock. The curriculum reaches 4.0 m/s and the sim test ramps 0 to 3 m/s. Hardware shows only stand, walk and W2S — [arXiv 2505.20619v3](https://arxiv.org/html/2505.20619v3)
- humanoid-gym and unitree_rl_gym (G1) both use a single fixed period (config `cycle_time`; 0.8 s for G1) — [humanoid-gym](https://raw.githubusercontent.com/roboterax/humanoid-gym/main/humanoid/envs/custom/humanoid_env.py); [unitree_rl_gym](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/legged_gym/envs/g1/g1_env.py)
- Cassie (Li 2024): separate walking (0.8 s period library) and running (3 m/s reference) policies. 400 m in 2:34 on hardware — [arXiv 2401.16889](https://arxiv.org/html/2401.16889)
- Cassie (Siekmann 2021): one multi-gait policy for 2-beat gaits on hardware, qualitative results only — [arXiv 2011.01387v2](https://arxiv.org/html/2011.01387v2)
- FLD learned-phase motion representation evaluated on the MIT Humanoid — [arXiv 2402.13820](https://arxiv.org/pdf/2402.13820)

### Inferences
- Taken together, a suggested design for the user, drawn from the patterns above:
  1. Integrated global phase phi += f*dt, with f blended over the switch window.
  2. Observe per-foot sin (or sin/cos) clocks plus f.
  3. A left-foot-zero convention enforced at library build time.
  4. Velocity command plus frequency sampled from the family's schedule, with some jitter around the schedule to gain the WTW/HugWBC robustness.
  5. Optional gait-routed rewards (Peng 2025) if jog, sprint and turning rewards conflict.
- The WTW finding that gait structure costs some top-speed tracking suggests letting the sprint band's contact-schedule reward be weaker. Another option is to let the policy add a bounded residual to f, as in Lee 2020 and CPG-RL.

### Gaps
- No hardware results were found for Booster T1, Berkeley Humanoid, Digit or G1 multi-frequency running policies. Searches were limited to about 20 tool calls.
- Radosavovic et al. (Digit) and Booster Gym clock designs were not checked.
