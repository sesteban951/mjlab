# High-Speed Legged Running with Learned Controllers (humanoid/biped focus, G1 3.5 m/s target)

Research date: 2026-11-18. Source labels used throughout: **[PEER]** = peer-reviewed venue; **[PREPRINT]** = arXiv, not (yet) peer-reviewed; **[COMPANY]** = company video/spec claim, not independently verified; **[NEWS]** = press/aggregator report; **[LOCAL]** = this repo's code.

## Q1. Documented humanoid/biped running speeds with learned control (incl. G1-specific numbers)

### Takeaway
On a **Unitree G1 specifically**, the strongest hardware evidence is 3.3 m/s (Caltech, treadmill, trajectory-optimized gait library + RL, 50 Hz), which is the closest analog to the user's setup; one 2026 preprint (SPRINT) claims ~5.9 m/s average / 6 m/s peak on a G1, and a 2026 preprint reports a G1-sized 1.36 m / 40 kg humanoid averaging 4.55 m/s over 100 m. So 3.5 m/s on a G1 is at or slightly beyond the best rigorously documented G1 result, but within what several 2026 groups claim. Full-size robots (H1, MirrorMe Bolt, Tiangong Ultra, a Noetix-affiliated 1.8 m robot) now claim 7–10 m/s, mostly as company/competition claims with redesigned hardware.

### Cited Findings

**Unitree G1 (≈1.3 m, ≈35 kg) — research results**
- [PREPRINT] "Chasing Autonomy" (Olkin, Compton, Bena, Ames, Caltech, arXiv 2603.25902, 2026): G1 top speed **3.3 m/s on an indoor treadmill**, full flight phase, >250 m consecutive outdoor sidewalk runs (Fig. 5 caption says >150 m), all onboard compute; outdoors with MPC+CBF obstacle avoidance ~2 m/s. Policy at **50 Hz** outputting joint setpoints to PD with gains taken from BeyondMimic, "deliberately over-damped to prevent high-frequency chatter". Gait library from a single LAFAN human running segment, dynamically optimized (multiple shooting, CasADi/Pinocchio/IPOPT, single-support + flight domains) into gaits **1.2–3.6 m/s**; training command range vx ∈ [1.1, 3.7] m/s, vy ∈ [−0.75, 0.75], yaw ∈ [−0.75, 0.75]. Hardware velocity-tracking accuracy is not quantified (sim only). Key limitation: "dynamic optimization without human data did not converge above 3 m/s." — [arXiv 2603.25902](https://arxiv.org/html/2603.25902)
- [PREPRINT/likely conf.] "Chasing Stability" (Olkin, Li, Compton, Ames, arXiv 2509.19573, 2025): CLF-guided RL running on G1 from multi-domain trajectory optimization **without human data**; command range 1.1–3.0 m/s (2% of envs stand); hardware robustness tests at ~**2.2 m/s on treadmill**; sim tracked a 3.3 m/s trajectory. Speed selection: commanded velocity picks the closest gait in a library. — [arXiv 2509.19573](https://arxiv.org/html/2509.19573)
- The later Caltech paper says this earlier no-human-data result "had low time to overheat and minimal outdoor trials" (thermal limit is a real G1 running constraint). — [arXiv 2603.25902](https://arxiv.org/html/2603.25902)
- [PREPRINT] SPRINT (arXiv 2605.28549, 2026): G1, **6 m/s peak** claimed; Table V real-world average velocity **5.91 m/s at 6 m/s command** (sim 5.96), 5.03 m/s at 5 m/s command, 4.06 m/s at 4 m/s command. Method: residual RL on top of "frequency-adaptive spectral priors" built from **5 LAFAN1 gait cycles at 0.66/1.10/2.29/2.87/3.40 m/s** (retargeted with GMR), i.e. extrapolates well beyond the fastest reference. Isaac Gym, 200 Hz physics / **50 Hz control**, MLP(256)+LSTM(64). How hardware velocity was measured is not described — treat with caution. — [arXiv 2605.28549](https://arxiv.org/html/2605.28549)
- [PREPRINT] SPRINT baseline table (their reimplementations, sim/max average speed on G1): Humanoid-Gym 3.90 m/s, AMP 4.75 m/s, AI-CPG 4.82 m/s, SPRINT 5.96 m/s. — [arXiv 2605.28549](https://arxiv.org/html/2605.28549)
- [PREPRINT] RuN (arXiv 2509.20696): residual policy on G1 with walk–run transitions over **0–2.5 m/s**. — [arXiv 2509.20696](https://arxiv.org/html/2509.20696)
- [PREPRINT] Gait-conditioned multi-phase curriculum (arXiv 2505.20619): G1 (23-DoF action), curriculum expands up to 4.0 m/s in sim; sim-to-real test ramps 0–3 m/s but **hardware results show only standing/walking transitions** (no hardware running speed). — [arXiv 2505.20619](https://arxiv.org/html/2505.20619)
- [PREPRINT] "Realtime Limb Trajectory Optimization for Humanoid Running" reports a G1 running trajectory at 1 m/s with 0.21 s stance and 0.26 s flight (model-based, not RL). — [arXiv 2501.17351](https://arxiv.org/pdf/2501.17351)
- [COMPANY] Unitree G1 official spec "moving speed" 2 m/s. — [Unitree G1 page](https://www.unitree.com/g1); [RobotShop comparison](https://community.robotshop.com/blog/show/comparing-the-g1-h1-humanoids)
- No official Unitree G1 sprint record was found.

**G1-scale and full-size humanoids — research**
- [PREPRINT] "Push the Limit: Learning High-Speed Sprinting and Aggressive Cornering on a Full-Scale Humanoid" (arXiv 2610.14840, authors affiliated with Noetix Robotics): 1.8 m, 17-DoF robot (70 kg in sim; **60 kg weight-reduced build** for the timed run), leg BLDC actuators 330 Nm peak, 19.15 rad/s no-load. **100 m in 14 s → 7.14 m/s average** at 7.0 m/s command (timed from video). A **second humanoid, 1.36 m, 40 kg, 23 DoF (G1-class size), ran 100 m in 22 s → 4.55 m/s average.** 50 Hz policy, 200 Hz physics. — [arXiv 2610.14840](https://arxiv.org/html/2610.14840)
- [PREPRINT] Booster Lab (arXiv 2606.27813): Booster T1 learned policy runs **up to 2.0 m/s** on hardware (upper bound of tested command range), 50 Hz. — [arXiv 2606.27813](https://arxiv.org/html/2606.27813v1)
- [PREPRINT] KSLC (arXiv 2409.16611): XBot-L humanoid, 3.5 m/s velocity tracking **in simulation only** (sim-to-sim validated). — [arXiv 2409.16611](https://arxiv.org/abs/2409.16611)
- Berkeley Humanoid, MIT Humanoid, Fourier GR-1: no learned running speed numbers found in sources reached. — [Berkeley Humanoid arXiv 2407.21781](https://arxiv.org/pdf/2407.21781v1)

**Cassie (biped, OSU/Agility)**
- [NEWS] Guinness record, fastest 100 m by a bipedal robot: **24.73 s**, May 11, 2022 (≈4.04 m/s average); must start and end standing; RL in MuJoCo with randomized friction/CoM, zero-shot deployment; no cameras/external sensors. — [IoT World Today](https://www.iotworldtoday.com/robotics/oregon-state-robot-sets-guinness-world-record-for-100-meter-dash); [ASME](https://www.asme.org/topics-resources/content/with-the-help-of-ai,-bipedal-robot-learns-to-run)
- [PREPRINT] "Optimizing Bipedal Locomotion for the 100m Dash With Comparison to Human Running" (arXiv 2508.03070): hardware 100 m trials 26.11 / 24.37 / 27.38 s (best 24.37 s; 3.83–4.10 m/s avg) at 4 m/s command; sim reaches 5 m/s but "the sim-to-real gap prevents" 5 m/s on hardware. LSTM policy, **40 Hz** PD targets, 2 kHz PD; training speeds 0–5 m/s; gait inputs = clock + stride frequency + swing ratio. Note: the 24.37 s best trial differs from the 24.73 s certified record. — [arXiv 2508.03070](https://arxiv.org/html/2508.03070v1)
- [PEER] Li et al., "RL for Versatile, Dynamic, and Robust Bipedal Locomotion Control", IJRR 2024: Cassie **peak estimated 4.2 m/s** in a fine-tuned 100 m dash policy with notable flight phases; full **400 m lap in 2 min 34 s** then return to standing; running with flight over uneven terrain. Dual-history (long+short I/O history) architecture, task randomization. — [arXiv 2401.16889](https://arxiv.org/pdf/2401.16889); [project page](https://xbpeng.github.io/projects/Cassie_Locomotion/index.html)

**Company demo claims (not peer-reviewed, not independently timed)**
- [COMPANY] Unitree H1 (≈62 kg, 0.8 m leg): **3.3 m/s** "world record" (early 2024), listed on the spec sheet with "potential mobility >5 m/s". — [Humanoids Daily](https://www.humanoidsdaily.com/news/unitree-h1-reclaims-speed-record-with-blistering-10-m-s-sprint); [Robozaps H1 review](https://blog.robozaps.com/b/unitree-h1-review)
- [COMPANY] Unitree H1 **peak 10 m/s** (some outlets 10.1 m/s) in footage released April 11, 2026; company-measured, no independent certification, method not disclosed. — [Humanoids Daily](https://www.humanoidsdaily.com/news/unitree-h1-reclaims-speed-record-with-blistering-10-m-s-sprint); [heise](https://www.heise.de/en/news/Unitree-s-humanoid-H1-robot-reaches-10-1-m-s-in-the-100m-sprint-11255951.html)
- [COMPANY] MirrorMe "Bolt" (175 cm, 75 kg; with Zhejiang University): **10 m/s peak** in outdoor testing (Feb 2026); redesigned joints and rebuilt power system; later unverified claim of >11 m/s. — [Hangzhou gov release](https://www.ehangzhou.gov.cn/2026-02/04/c_296626.htm); [Gasgoo](https://autonews.gasgoo.com/articles/market-industry/mirrorme-technology-unveils-full-size-humanoid-robot-2018710186120552449); [Mugglehead](https://mugglehead.com/humanoid-robot-bolt-hits-10-metres-per-second-in-real-world-test/); [humanoid.guide](https://humanoid.guide/?p=20975)
- [COMPANY/NEWS] RobotEra STAR1 3.6 m/s; Unitree H2 rated under 2 m/s. — [Humanoids Daily](https://www.humanoidsdaily.com/news/unitree-h1-reclaims-speed-record-with-blistering-10-m-s-sprint)
- [COMPANY/NEWS] Tesla Optimus: Dec 2–3, 2025 lab video captioned "new personal record"; **no official speed**; third-party estimates conflict (≈2.5–3 m/s; ≈2.7 m/s; 3.8 m/s). — [Tesla North](https://teslanorth.com/2025/12/02/tesla-drops-new-video-of-optimus-running-smoothly); [Mike Kalil](https://mikekalil.com/blog/tesla-optimus-vs-figure-03-running/); [Humanoids Daily](https://www.humanoidsdaily.com/news/unitree-h1-reclaims-speed-record-with-blistering-10-m-s-sprint)
- [COMPANY/NEWS] Figure 03: shuttle-run video with flight phase; **no official speed data**; one outlet's reading of 4.0 m/s. — [OFweek](https://en.ofweek.com/ofweek-news/detail/figure-ai-edges-out-tesla-optimus-with-a-40-ms-sprint-record.html); [Humanoids Daily](https://www.humanoidsdaily.com/news/figure-03-sprints-in-new-video-answering-tesla-s-challenge)
- Unitree R1, Digit: no running-speed figures found.

**Competitions (news)**
- [NEWS] Beijing E-Town humanoid half-marathon (Apr 19, 2025): Tiangong Ultra won in **2 h 40 min 42 s** (21.0975 km ≈ 2.2 m/s avg) with three battery swaps and one fall. — [CGTN](https://news.cgtn.com/news/2025-04-19/-Tiangong-Ultra-wins-world-s-first-ever-humanoid-robot-half-marathon-1CHdanwJVzG/p.html); [Xinhua](https://english.news.cn/20250419/90fc05c7531c4714ac69bb3e9fac7630/c.html)
- [NEWS] World Humanoid Robot Games 2025 100 m: Tiangong won in **21.50 s** after a 0.8 time coefficient for autonomous robots (it physically finished third); Unitree H1 second at 22.08 s. — [Wikipedia](https://en.wikipedia.org/wiki/World_Humanoid_Robot_Games); [IEEE Spectrum](https://spectrum.ieee.org/amp/world-humanoid-robot-games-2673909826)
- [NEWS] WHRG 2026 (Aug 22–26): Tiangong Ultra 100 m **9.39 s** (Honor "Lightning" 9.47 s), then **8.86 s** semifinal; 400 m 38.15 s and 1500 m 2:21.64 claimed by the developer. — [Wikipedia](https://en.wikipedia.org/wiki/World_Humanoid_Robot_Games); [Yahoo Sports](https://sports.yahoo.com/olympics/article/chinese-robot-beats-usain-bolts-100-meter-world-record-sprinting-in-939-seconds-at-world-humanoid-robot-games-173509467.html). Conflicting reports: one attributes 9.39 s to an "X-Humanoid" machine, another reports an 8.64 s final; a headline cites 8.4 s at a Beijing event — not reconciled. — [Interesting Engineering headline](https://interestingengineering.com/ai-robotics/humanoid-robot-tiangong-ultra-100m-sprint-record)

### Inferences
- The most transferable evidence for the user is Caltech's 3.3 m/s G1 result: same robot, same 50 Hz/PD-position setup, and a trajectory-optimized gait-library reference, just like the user's pipeline. It suggests 3.5 m/s is roughly at the edge of what that approach has shown on hardware. Their note that TO without human data would not converge above 3 m/s is a warning for the user's sprint library (2.5–3.5 m/s).
- SPRINT's 5.9 m/s on a G1, and the 4.55 m/s average from the 40 kg/1.36 m robot, suggest the G1-class hardware is not the hard limit at 3.5 m/s. But SPRINT's hardware measurement method is unstated, so treat it as an upper-bound claim.
- Full-size robots reaching 7–10 m/s mostly involved hardware changes (MirrorMe rebuilt joints and power; Noetix used a 10 kg lighter build), so those numbers do not carry over to a stock G1.

### Gaps
- No peer-reviewed G1 running paper with independently timed hardware sprint speed was found; all G1 >3 m/s numbers are 2026 preprints.
- Unitree's own G1 running/sprint demos (if any) and their speeds were not found; nothing on the Unitree R1.
- The 2026 WHRG 100 m final result is inconsistent across reports.
- Digit, Berkeley Humanoid (Lite), MIT Humanoid, Fourier GR-1 running speeds were not found in this pass.

## Q2. Quadruped high-speed results as a methods reference

### Takeaway
The quadruped speed-learning canon is Margolis et al. (Mini Cheetah 3.9 m/s, RSS 2022), and its key lesson is a **grid adaptive curriculum over the joint (vx, ωz) space**. Ji et al. reached 3.75 m/s with a concurrently trained state estimator. Newer curricula claim more in sim (6.3 m/s) and on Go1 hardware (4.1 m/s).

### Cited Findings
- [PEER, RSS 2022] Margolis, Yang, Paigwar, Chen, Agrawal, "Rapid Locomotion via RL": Mini Cheetah **up to 3.9 m/s** indoor sprint, 3.4 m/s outdoor 10 m dash, **5.7 rad/s spin**, on grass/ice/gravel. Two key components: adaptive curriculum on velocity commands + online system identification for sim-to-real. — [arXiv 2205.02824](https://arxiv.org/abs/2205.02824); [project page](https://agility.csail.mit.edu/)
- Grid adaptive curriculum: discrete grid at **0.5 m/s × 0.5 rad/s**; when a command among the most challenging is tracked above a reward threshold, its 4-connected neighbours are added, so the distribution expands locally. A "Box Adaptive" variant also learned to control the robot but **excluded the extremes** of the command space; the grid covers more area because it accounts for the combined difficulty of running plus turning. — [arXiv 2205.02824 (search excerpt)](https://arxiv.org/pdf/2205.02824)
- RLvRL (Margolis) reached 5.5 m/s in simulation vs 3.9 m/s on hardware (per TransCurriculum's summary). — [arXiv 2603.14156](https://arxiv.org/html/2603.14156v1)
- [PEER, RA-L 2022] Ji, Mun, Kim, Hwangbo, "Concurrent Training of a Control Policy and a State Estimator": Mini Cheetah **3.75 m/s** on flat ground, 3.54 m/s on a μ=0.22 slippery plate; the estimator network outputs base linear velocity, foot height and contact probability. — [arXiv 2202.05481](https://arxiv.org/pdf/2202.05481)
- [PREPRINT] TransCurriculum (arXiv 2603.14156): joint curriculum over commands, terrain and friction/payload; Unitree Go1 **6.3 ± 0.2 m/s in sim, 4.1 ± 0.05 m/s zero-shot hardware**. — [arXiv 2603.14156](https://arxiv.org/html/2603.14156v1)
- [PREPRINT] HACL (arXiv 2505.18429) replaces the fixed threshold rule with a recurrent model over reward history to steer the curriculum. — [arXiv 2505.18429](https://arxiv.org/html/2505.18429v2)
- [COMPANY] MirrorMe Black Panther II quadruped: 10.3 m/s, later 13.4 m/s. — [Gasgoo](https://autonews.gasgoo.com/articles/market-industry/mirrorme-technology-unveils-full-size-humanoid-robot-2018710186120552449)

### Inferences
- Margolis's "running + turning combined" difficulty argument maps directly onto the user's policy: sprint (2.5–3.5 m/s) combined with fast yaw is the hardest corner. A box curriculum would likely never unlock high-vx × high-|ωz|. A grid over (vx, ωz) lets in-place fast turning (vx≈0, |ωz| 1–2) unlock on its own path while high-speed straight running unlocks along the vx axis.

### Gaps
- Exact Margolis reward threshold values, initial grid extents, and domain-randomization ranges could not be extracted (the PDF exceeded fetch limits).
- No sourced Go2/B2 or ANYmal top-speed learned-running numbers were retrieved in this pass.

## Q3. Curriculum over commanded speed: designs, failure modes, avoiding forgetting of slow speeds

### Takeaway
Every high-speed result that reports an ablation finds that a speed curriculum is essential: SPRINT's G1 peak falls from 5.96 to 4.48 m/s without one, and training "gets stuck in local minima". Common ways to keep slow skills are: never drop easy bins from the sampling distribution (grid curricula), always keep a fraction of standing or walking environments, start some episodes from walking references, route rewards by gait, or freeze or residualize on a slow-gait base policy.

### Cited Findings
- [PREPRINT] SPRINT (G1): a progressive curriculum expands commands from standing to sprinting (vx 0–6 m/s, yaw ±0.7 rad/s; gait frequency 0.6–2.3 Hz). Ablation: **without curriculum, peak 4.48 m/s and "stuck in local minima"**; without asymmetric actor-critic 5.46 m/s; without prior reward 6.06 m/s but unnatural motion (FID 12.23 vs 0.80); full 5.96 m/s. It reaches 4 m/s in ~3 h and converges at 6 m/s in ~6.5 h on one RTX 4090. No per-stage schedule given. — [arXiv 2605.28549](https://arxiv.org/html/2605.28549)
- [PREPRINT] Noetix "Push the Limit": velocity curriculum **capped at 5.5 m/s** in training, yet hardware ran under a 7.0 m/s command. A one-hot skill label (standing / in-place turning / walking / running) is part of the command. — [arXiv 2610.14840](https://arxiv.org/html/2610.14840)
- [PREPRINT] TransCurriculum: "Starting with a wide initial range, such as [−5,5], destabilizes learning"; its curriculum starts at [−1, 1] and expands by ±0.5. A non-history MLP scheduler failed (0.5 m/s achieved at a 7.0 m/s command, 0% success). — [arXiv 2603.14156](https://arxiv.org/html/2603.14156v1)
- [PEER] Margolis: grid expansion adds neighbours of mastered hard bins; box curriculum excludes extremes. — [arXiv 2205.02824](https://arxiv.org/pdf/2205.02824)
- [PREPRINT] Gait-conditioned multi-phase curriculum (G1): Phase 1 walk only; Phase 2 adds stand and walk-to-stand (gated at |v|<0.1 m/s, 1.5 s double support); Phase 3 enables run when **Froude number > 0.5**, plus run-to-walk, then co-trains all five modes. Without the curriculum, training failed to converge (episode length 31 vs 973). Without gait-specific reward routing, conflicting terms (stillness vs forward velocity) collapsed learning (return 0.00 vs 89.3). The authors flag that a shared policy "may be prone to catastrophic forgetting or behavioral drift" as gaits grow, but they do not measure it. — [arXiv 2505.20619](https://arxiv.org/html/2505.20619)
- [PREPRINT] Chasing Autonomy (G1): **40% of environments initialized from the 1 m/s walking reference** to learn walk-to-run; a separate walking policy handles run-to-walk. Their weakness: velocity error grows at lower commanded speeds because the reference was a fast run. — [arXiv 2603.25902](https://arxiv.org/html/2603.25902)
- [PREPRINT] Chasing Stability (G1): no speed curriculum. Speeds are sampled uniformly over 1.1–3.0 m/s with **2% standing envs**; the curriculum is on the flight contact-penalty weight instead (reaches its final value at ~8,300 of 15,000 iterations). — [arXiv 2509.19573](https://arxiv.org/html/2509.19573)
- [PREPRINT] KSLC (XBot-L): progressively expands the commanded velocity range **and** shortens the gait cycle time as commanded speed increases (stride frequency curriculum). — [arXiv 2409.16611 (search excerpt)](https://arxiv.org/pdf/2409.16611)
- [PREPRINT] GaitSpan (arXiv 2607.12114): freezes a pretrained walking policy as a "seed skill" and grows running on top via multiple internal clocks, a SLIP-based stride reward, and a residual. Walking is preserved by construction, though the abstract does not quantify forgetting. — [arXiv 2607.12114](https://arxiv.org/abs/2607.12114)
- [PREPRINT] Cassie 100 m: training speeds sampled 0–5 m/s with gait parameters (frequency, swing ratio) sampled around a hand-tuned speed mapping; no speed curriculum described. — [arXiv 2508.03070](https://arxiv.org/html/2508.03070v1)

### Inferences
- With a reference library, the reference does part of a curriculum's job (each speed has a feasible target). But SPRINT's no-curriculum drop (5.96 → 4.48 m/s) happened even with priors, so a curriculum is still worth having for the 2.5–3.5 m/s band.
- A practical recipe consistent with these sources: a reward-gated grid over (vx, ωz) that keeps unlocked jog and turn bins in the sampling mix. Add a fixed floor of standing and slow-jog environments (cf. 2% stand; 40% walking-reference starts), and start sprint episodes from lower-speed library states so acceleration is learned.
- Policy collapse or standing still at high commands is most often reported as reward conflict (stillness vs velocity) or too-wide initial ranges, not as an inherent speed limit.

### Gaps
- None of the retrieved papers quantitatively measures slow-speed forgetting as the speed range widens; this remains anecdotal.
- No exact "standing still" collapse statistics for humanoids under uniform full-range sampling were found.

## Q4. Sim-to-real factors specific to fast gaits (actuators, limits, latency, contact) and G1 motor specs

### Takeaway
At top speed, every system that reports it is **torque-limited at hip and knee**: Cassie's hip-roll saturates above ~3 m/s; the Noetix robots run hip-pitch and knee at ~99% of torque and ankle at ~90%. Friction randomization, PD-gain randomization, actuator delay (0–20 ms), armature, and measured torque-speed (T–n) limits are the factors papers credit. On the G1, the 7520-22 knee/hip-roll motor (139 Nm, 20 rad/s in mjlab's model) is the binding actuator. **mjlab's default G1 config does not enforce a velocity limit or torque-speed curve**, and overheating has been reported for G1 running.

### Cited Findings
- [PREPRINT] Cassie: hip-roll motors reach maximum torque above ~3 m/s, which the authors hypothesize caps effective ground reaction force. Effective GRF in Cassie tapers after ~3 m/s, whereas in humans it rises near-linearly with speed. — [arXiv 2508.03070](https://arxiv.org/html/2508.03070v1)
- [PREPRINT] Noetix: at the highest tested speed, **hip-pitch and knee-pitch ~99% of torque limit, ankle-pitch ~90%**. Both robots hit torque limits at top speed. At 6.5 m/s the robot recovered from only 56% of backward and 72% of lateral pushes (vs 100% at 5.5 m/s), attributed to the torque bottleneck. Authors list as limitations: no energy-limiting penalty for motor capacity, and gaits drifting to hardware-exploiting postures without a reference. — [arXiv 2610.14840](https://arxiv.org/html/2610.14840)
- [PREPRINT] Noetix actuator model: delayed implicit actuator with **random delay 0–4 sim steps (0–20 ms)**; leg PD natural frequency 2–5 Hz, damping ratio 0.5–0.707; action scale 0.25·τmax/kp. DR: friction 0.3–1.0, restitution 0–0.5, joint offset ±0.02 rad, CoM ±0.05 m, base mass ±5 kg, **kp/kd 0.8–1.2×, joint friction and armature 0.7–1.5×**, pushes ±1.0 m/s every 5–10 s. Joint velocity limits are only a soft penalty at a 0.9 ratio. — [arXiv 2610.14840](https://arxiv.org/html/2610.14840)
- [PREPRINT] Booster Lab (T1): **measured torque–speed (T–n) curves; their knee points set the simulated torque and velocity limits** (e.g. knee 60 Nm / 12.57 rad/s; hip pitch 45 Nm / 16.76 rad/s). Armature = N²·J_m. DR: friction, joint stiffness and damping, trunk mass, initial state, terrain roughness, pushes. — [arXiv 2606.27813](https://arxiv.org/html/2606.27813v1)
- [PREPRINT] Chasing Autonomy (G1): friction and PD-gain randomization were "very helpful" for high-speed sim-to-real. Motor properties and mass are also randomized, with root velocity pushes. PD gains over-damped (BeyondMimic gains). — [arXiv 2603.25902](https://arxiv.org/html/2603.25902)
- [PREPRINT] Chasing Stability (G1) DR: torso CoM, extra torso mass, PD gains, restitution, joint friction, **armature**, ground friction, velocity impulses; torque-limit penalty. — [arXiv 2509.19573](https://arxiv.org/html/2509.19573)
- [PREPRINT] SPRINT (G1) DR: mass, ground friction, **motor latency** (ranges not given). — [arXiv 2605.28549](https://arxiv.org/html/2605.28549)
- [PEER] Margolis uses online system identification; Ji et al. learn a state estimator (base velocity, contact probability) concurrently. Velocity estimation matters at speed because the actor usually does not observe base linear velocity. — [arXiv 2205.02824](https://arxiv.org/abs/2205.02824); [arXiv 2202.05481](https://arxiv.org/pdf/2202.05481)
- [PREPRINT] G1 thermal: earlier Caltech G1 running "had low time to overheat". — [arXiv 2603.25902](https://arxiv.org/html/2603.25902)
- [PREPRINT] Hardware changes matter at the extreme: Noetix's sprint used a 60 kg weight-reduced build vs a 70 kg nominal. — [arXiv 2610.14840](https://arxiv.org/html/2610.14840)

**G1 motor specs**
- [COMPANY] Official max knee torque **90 Nm (standard) / 120 Nm (EDU)**; motors are PMSM with dual encoders; hip range P ±154°, R −30…+170°, Y ±158°. Official pages give no joint velocity limits. — [Unitree G1](https://www.unitree.com/g1); [G1 brochure 2025-04](https://www.jecafair.jp/online2025/wp-content/uploads/2025/04/G1-Brochure-EN-2025.4.2.pdf)
- [LOCAL] mjlab `g1_constants.py` actuator models: 7520-22 (hip roll, knee) **139 Nm, 20 rad/s**; 7520-14 (hip pitch, hip yaw, waist yaw) **88 Nm, 32 rad/s**; 5020 (shoulders, elbow, wrist roll) 25 Nm, 37 rad/s; 4010 (wrist pitch/yaw) 5 Nm, 22 rad/s. Ankle pitch/roll and waist pitch/roll are 4-bar linkages modeled as 2× 5020 (**50 Nm**, 2× armature). PD gains come from armature with 10 Hz natural frequency and damping ratio 2.0 (over-damped). The `velocity_limit` values are defined on `ElectricActuator` but are **not passed to `BuiltinPositionActuatorCfg`** (only `effort_limit` and armature are), so the default sim has a flat torque clip with no speed-dependent derating. — `/home/amberlab/projects/mjlab/src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py` (lines ~105–190); [upstream](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py)
- [NEWS/blog] A 2026 review also lists G1 hip 88 Nm and knee 139 Nm peak, consistent with mjlab, but cites no primary source and conflicts with Unitree's 90/120 Nm knee spec. The difference is plausibly rated-at-joint vs motor peak, but this is unconfirmed. — [roboticscenter.ai review](https://admin.roboticscenter.ai/blog/unitree-g1-review)

### Inferences
- For a 3.5 m/s G1 sprint, the probable binding constraints are knee/hip-roll torque (7520-22) and, at high stride frequency, hip-pitch/knee speed, where a real PMSM loses torque near no-load speed. Because mjlab's default model clips torque flat, sim may be optimistic at sprint speed. Adding a torque–speed envelope (e.g. linear derating to zero at velocity_limit, as Booster Lab did from measured curves) plus a joint-velocity penalty or limit is likely the single most important fidelity upgrade for the sprint band.
- The ankle is modeled at 50 Nm combined. Noetix saw ankle-pitch at ~90% at top speed, so G1 ankle push-off may also saturate; this is worth logging per-joint torque/limit ratios across the speed sweep in sim.
- Thermal limits, plus the Caltech report of "low time to overheat", argue for an energy/power penalty (Noetix used motor power² at −5e-3) and short sprint bouts in hardware testing.

### Gaps
- No authoritative G1 joint velocity limits or torque–speed curves from Unitree were found; the mjlab values are the best available, and their provenance is unverified here.
- No paper quantified latency, contact, or foot-geometry sensitivity specifically at sprint speeds on the G1.
- Battery/power limits during sprints were not documented for G1.

## Q5. Reward and observation tricks for running (flight, clearance, energy, symmetry, gait clocks)

### Takeaway
High-speed papers use **speed-gated contact rewards** (penalize double flight at low speed, penalize double stance at high speed), clock or phase observations with gait-parameter inputs, flight-phase contact penalties with their own curriculum, holonomic (no-slip) stance terms, energy/power penalties, and symmetry augmentation. Human and Cassie data suggest top speed comes from greater ground force and longer aerial strides rather than faster leg cycling.

### Cited Findings
- [PREPRINT] SPRINT reward table (G1): **Low-speed air penalty −6.0 (v<1.5 m/s, both feet airborne)**; **High-speed ground penalty −3.0 (v>3.0 m/s, both feet on ground)**; feet air height +5.0; close-feet −100; feet slide −10; feet-ground-parallel +1; torque −5e-6; joint accel −2e-8; joint vel −5e-4; joint limits −10; roll/pitch ang-vel −0.5; base height −30; plus prior-guidance/velocity terms (weights unstated). Residual action q = q_ref + α·a on a 10-DoF subset (hip pitch, knee, ankle pitch, shoulder pitch, elbow). Actor obs: ang vel, gravity, commands, joints, last action; critic adds lin vel and reference. — [arXiv 2605.28549](https://arxiv.org/html/2605.28549)
- [PREPRINT] Chasing Stability (G1): CLF tracking exp(−V) (10.0), CLF decay penalty (1.0), holonomic stance-foot position (4.0) and velocity (2.0), **flight contact penalty via tanh of foot forces (3.0) with a weight curriculum**, torque 1e-5, action rate 1e-3, joint and torque limits 1.0. Rewards switch by hybrid domain. — [arXiv 2509.19573](https://arxiv.org/html/2509.19573)
- [PREPRINT] Chasing Autonomy (G1): CLF-RL beats generic mimic rewards for every reference type. Goal-conditioning on velocity (lin xy 1.0, yaw 1.0) improves tracking over kinematically adjusting the reference. Actor obs include sin/cos gait phase but **not the reference** (simplifies deployment); critic gets reference and contacts. — [arXiv 2603.25902](https://arxiv.org/html/2603.25902)
- [PREPRINT] Noetix: XY velocity exp 5.0, heading alignment 3.0, **forward speed bonus 2.0**, angular momentum 1.0, min foot separation 0.2 m (−5.0), **motor power² (−5e-3)**, action rate −0.1; **symmetry augmentation enabled**; 10-frame causal attention history encoder; actor does not observe base linear velocity. No explicit flight or stride reward was needed. — [arXiv 2610.14840](https://arxiv.org/html/2610.14840)
- [PREPRINT] Gait-conditioned (G1): running terms push-off (1.0), short ground contact (0.2), forward velocity (0.2); feet swing-height penalty −25; arm-swing symmetry 2.0; gait mask routes terms per mode. — [arXiv 2505.20619](https://arxiv.org/html/2505.20619)
- [PREPRINT] Cassie: dense periodic clock reward (zero foot velocity in stance, etc.); inputs include clock, **stride frequency and swing ratio**. Optimized fast gaits favor **low stride frequency and high swing ratio** (long, infrequent steps with significant aerial phase). Human aerial time ≈ 2/3 of swing time. Cites "Faster top running speeds are achieved with greater ground forces not more rapid leg movements." — [arXiv 2508.03070](https://arxiv.org/html/2508.03070v1)
- [PREPRINT] GaitSpan: a SLIP-dynamics-based stride-shaping reward for higher speeds. — [arXiv 2607.12114](https://arxiv.org/abs/2607.12114)
- [PREPRINT] KSLC: coordinated arm motion to counter destabilizing forces at speed; cycle time shortens with commanded speed. — [arXiv 2409.16611](https://arxiv.org/abs/2409.16611)

### Inferences
- The user's library-based setup could add SPRINT-style speed-gated contact terms aligned with the band boundaries (penalize flight in jog <1.5 m/s; penalize double support in sprint >2.5–3.0 m/s) without hand-tuning per-gait rewards.
- Given Cassie and human evidence, sprint references with longer flight and higher GRF are likely more torque-feasible than ones with high cadence. But GRF is exactly what saturates (Cassie hip-roll, Noetix knee), so torque-limit penalties should be in place before pushing the sprint band.
- Symmetry augmentation (Noetix) and mirrored library strides (Caltech) are cheap ways to get left/right-balanced gaits, which matters for straight-line tracking at 3.5 m/s and for symmetric ±ωz turning.

### Gaps
- No ablation was found isolating the effect of symmetry augmentation, energy penalties, or flight rewards on achievable top speed for humanoids.
- Foot-clearance scaling with speed is not explicitly specified in any retrieved humanoid sprint paper (SPRINT uses a constant feet-air-height reward).
