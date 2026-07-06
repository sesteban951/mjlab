"""Velocity-conditioned crawl gait-library tracking task.

Tracks a library of periodic crawl clips (one per forward speed) reusing mjlab's BeyondMimic
per-frame reference tracking. A commanded forward speed selects the nearest library clip per env.
"""
