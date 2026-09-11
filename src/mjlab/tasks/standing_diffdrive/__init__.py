"""Differential-drive (tank-style) UPRIGHT walking over the mj-nlp walk gait library.

The upright sibling of ``G1-Crawling-DiffDrive``: the same twist-indexed gait library command with
the differential-drive sampler -- each env is either walking straight ``[vx, 0, 0]`` (forward or
backward) OR turning in place ``[0, 0, wz]``, never a blended arc and never lateral -- with blended
clip transitions, a reference-free actor and a standing idle stop. What changes is the plant side:
the env builds on the upright ``G1-Tracking-Custom`` base (feet-only frictional contact) instead of
the contact-rich crawl base, starts on a standing pose, and reads its heading from the pelvis yaw.

The library is the upright walk grid from mj-nlp's ``examples/g1_mimic_periodic/library``
(walk_fwd, walk_bck, walk_turn_pos, walk_turn_neg; all T = 1.4 s -> 70 tracking frames) plus a
standing idle clip. See ``config/g1/env_cfgs.py``.
"""
