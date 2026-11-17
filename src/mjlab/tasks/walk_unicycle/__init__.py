"""Unicycle WALKING over an mj-nlp walk gait library.

The walking sibling of ``G1-Jog-Unicycle``, same machinery throughout: twist-indexed
``LibraryMotionCommand``, blended clip transitions, a reference-free actor, egocentric path/heading
rewards and an idle-pose stop. Each env arcs ``[+-vx, 0, wz]``, pivots ``[0, 0, wz]`` or stands.
The sampler is the jog's ``UnicycleMotionCommand``, imported rather than copied; what differs is
the library and the commanded ranges (``config/g1/env_cfgs.py``).

Library: ``LIBRARY_SPECS["walk_unicycle"]``; ``uv run python -m mjlab.scripts.build_library walk_unicycle``.
"""
