"""Unicycle WALKING over an mj-nlp walk gait library.

The walking sibling of ``G1-Jog-Unicycle``, and the same machinery throughout -- twist-indexed
``LibraryMotionCommand``, blended clip transitions, a reference-free actor (proprioception +
projected gravity + commanded twist + phase clock; the critic keeps the full reference), egocentric
path/heading rewards so looping and turning clips accumulate net motion, and an idle-pose stop.

It is ``G1-Walking-DiffDrive`` with the jog's command set. The differential drive is translate XOR
rotate; a unicycle does both at once. Each env is either arcing ``[±vx, 0, wz]`` (walk forward or
backward while turning, with ``wz = 0`` the straight case), pivoting ``[0, 0, wz]``, or standing.
Lateral motion is still never commanded.

The sampler is the jog's ``UnicycleMotionCommand``, imported rather than copied: nothing in it is
gait-specific. What differs is the library and the commanded ranges (``config/g1/env_cfgs.py``).

Library: ``crawling_common.library.LIBRARY_SPECS["walk_unicycle"]``; build/refresh it with
``uv run python -m mjlab.scripts.build_library walk_unicycle``.
"""
