"""Twist-conditioned crawl gait-library tracking task, with an idle-pose stop.

Tracks a library of periodic crawl gaits, each labelled with a planar twist ``[vx, vy, wz]``,
reusing mjlab's BeyondMimic per-frame reference tracking with a reference-free actor. The command
samples a target twist on a timer and snaps to the nearest library clip; a fraction of resamples
command a zero twist and snap to the static idle clip, so the robot holds the idle pose. Tracking
the static idle clip *is* the rest -- no phase-freeze or rest-reward machinery.
"""
