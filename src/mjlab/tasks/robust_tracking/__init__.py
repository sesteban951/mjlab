"""CLF-guided G1 tracking hardened for sim2real.

``G1-Tracking-Control`` re-tuned for transfer rather than for the cleanest possible CLF
ablation. It keeps that task's guide-only TVLQR -- the gain schedule is evaluated alongside
the policy every physics substep and feeds two shaping rewards, but ``apply_actions`` always
sends the POLICY's target to the sim -- and drops everything that only made sense as an
experiment: no blend weight, no ``lam`` curriculum, no environment-variable switches. Every
knob is a module constant in ``config/g1/env_cfgs.py``.

Three deliberate departures from ``G1-Tracking-Control``, all in service of transfer:

* the STOCK 7-CAPSULE SOLE is kept. The control task swaps in mj-nlp's 4-sphere sole because
  that is the plant its TVLQR was designed on; here the hardware plant wins. Guide-only is
  what makes that affordable -- the law is never in the loop, so a sole it was not designed
  for cannot destabilize anything, it only makes ``V`` a noisier shaping signal.
* the robot STARTS STANDING, on ``G1-Standing-DiffDrive``'s idle pose, and the action offset
  rides along with it, so a zero action is the stand rather than the gait's mean posture.
* the full custom DR base and the randomized actuator delay come along from
  ``G1-Tracking-Custom``, as does the limb/waist action-rate split.

See ``config/g1/env_cfgs.py`` for the constants and the reasoning behind each.
"""
