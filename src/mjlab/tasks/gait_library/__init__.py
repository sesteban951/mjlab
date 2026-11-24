"""Twist-conditioned gait-library tracking, shared by every library task.

A library is a set of periodic one-stride clips, each labelled with a planar twist
``[vx, vy, wz]``. The command snaps a commanded twist to the nearest clip, blends
across clip switches and serves the reference unrolled in the world; the actor is
reference-free. Nothing here assumes a posture or a gait: the crawl, walk, jog and
sprint tasks differ only in their library, twist sampler and task-specific rewards.
"""
