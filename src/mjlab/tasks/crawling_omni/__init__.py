"""Twist-conditioned OMNIDIRECTIONAL crawl gait-library tracking.

The shared gait-library machinery (twist-indexed library, reference-free actor, idle-pose
stop, blended clip transitions) over the FULL 3-D twist library: the commanded twist
``[vx, vy, wz]`` varies on all three axes, so one policy learns forward crawling with
lateral and turning steering. The library is mj-nlp's ``crawl_fwd`` 5x5x5 grid plus a
zero-twist idle clip; see ``config/g1/env_cfgs.py``.
"""
