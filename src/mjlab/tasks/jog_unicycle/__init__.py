"""Unicycle JOGGING over an mj-nlp jog gait library.

Twist-indexed ``LibraryMotionCommand`` with blended clip transitions, a reference-free actor
(proprioception + gravity + commanded twist + phase clock), egocentric path/heading rewards and an
idle-pose stop. Each env arcs ``[+-vx, 0, wz]`` (``wz = 0`` is straight), pivots ``[0, 0, wz]`` or
stands; lateral motion is never commanded. The library is a 2-D ``(vx, wz)`` grid, so the
nearest-clip metric weights come from the commanded spans (``mdp/commands.py``).

Library: ``LIBRARY_SPECS["jog_unicycle"]``; ``uv run python -m mjlab.scripts.build_library jog_unicycle``.
"""
