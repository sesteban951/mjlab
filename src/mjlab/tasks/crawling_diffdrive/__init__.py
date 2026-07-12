"""Differential-drive (tank-style) crawl gait-library tracking, with blended clip transitions.

Same machinery as ``G1-Crawling-Omni`` (twist-indexed gait library, reference-free actor, idle-pose
stop, blended transitions), but the commanded twist is restricted to a DIFFERENTIAL-DRIVE command
space: each env is either driving straight ``[vx, 0, 0]`` OR turning in place ``[0, 0, wz]`` -- never
a blended arc, and never lateral (``vy`` is always 0). A per-resample mode draw picks translate vs
rotate (vs a small idle fraction); only that one axis is sampled, the others are pinned to 0.

The library therefore mixes two motion families -- the pure-forward clips (``vy=wz=0``) from the
forward crawl grid and the pure in-place turn clips (``vx=vy=0``) from the turning grid -- plus a
zero-twist idle clip. All gaits were generated at a common period, so they stack into one
``LibraryMotionLoader`` (shared frame count ``T``). See ``config/g1/env_cfgs.py``.
"""
