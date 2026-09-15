"""Unicycle JOGGING over an mj-nlp jog gait library.

The running sibling of ``G1-Standing-DiffDrive``, and the same machinery throughout -- twist-indexed
``LibraryMotionCommand``, blended clip transitions, a reference-free actor (proprioception +
projected gravity + commanded twist + phase clock; the critic keeps the full reference), egocentric
path/heading rewards so looping and turning clips accumulate net motion, and an idle-pose stop.

ONE THING CHANGES: the command set. The differential drive is translate XOR rotate; a unicycle does
both at once. Each env is either arcing ``[±vx, 0, wz]`` (jog forward or backward while turning,
with ``wz = 0`` the straight case), pivoting ``[0, 0, wz]``, or standing. Lateral motion is still
never commanded.

That makes the library a genuine 2-D ``(vx, wz)`` grid instead of two 1-D columns, which is why the
nearest-clip metric weights are derived from the commanded spans here (see ``mdp/commands.py``)
rather than left equal.

Library: ``crawling_common.library.LIBRARY_SPECS["jog_unicycle"]``; build/refresh it with
``uv run python -m mjlab.scripts.build_library jog_unicycle``. See ``config/g1/env_cfgs.py``.
"""
