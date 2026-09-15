"""CLF-guided G1 sideroll tracking hardened for sim2real.

``G1-Tracking-Control`` re-tuned for transfer rather than for a clean CLF ablation. It keeps
that task's guide-only TVLQR -- the schedule feeds two shaping rewards but ``apply_actions``
always sends the POLICY's target to the sim -- and drops the blend weight, the ``lam``
curriculum and the ``MJLAB_PRIOR_*`` switches. Every knob is a constant in
``config/g1/env_cfgs.py``.

Departures from ``G1-Tracking-Control``: the stock capsule sole is kept (guide-only makes a
mismatched plant cost shaping, not stability), the action offset is the clip's own first
frame, the episode ends when the clip does, and contact ``solref`` plus a gentle ground tilt
are randomized on top of the inherited custom DR base.
"""
