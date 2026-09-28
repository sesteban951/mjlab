"""Unitree G1 unicycle locomotion over a TWO-STRIDE gait library: walk + jog in one env.

``G1-Walk-Unicycle`` and ``G1-Jog-Unicycle`` each track a single-period library -- the
walk at T = 1.4 s (70 frames) and the jog at T = 0.86 s (43 frames). This task tracks
BOTH at once, so the commanded twist selects not just a speed but a gait: a 0.9 m/s walk
covers 1.26 m per stride, while the 1.0 m/s jog just across the cut covers 0.86 m.

Two things make that possible. The library (``LIBRARY_SPECS["walkjog_unicycle"]``)
carves the two grids into DISJOINT twist regions so a nearest-twist snap is never
ambiguous -- see that spec for the cut and why each boundary sits where it does. And the
loader/command handle a RAGGED library: clips are padded to the longest and each one
wraps at its OWN frame count, so both gaits keep their native period off a single
frame-per-step clock (the env steps at 50 Hz, the clips' own fps). A mid-episode twist
change rescales the frame index to preserve PHASE, so the transition blends between two
cycles rather than teleporting."""
