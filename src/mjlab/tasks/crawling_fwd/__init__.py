"""Velocity-conditioned crawl gait-library tracking task, with an idle-pose stop.

Same as ``G1-Crawling-NoStopping`` (speed-indexed gait-library tracking, reference-free actor), but
the commanded speed range includes 0: speeds below a stop threshold are pinned to 0 and snap to the
static idle clip, so the robot holds the idle pose instead of crawling. Tracking the static idle
clip *is* the rest -- no phase-freeze or rest-reward machinery.
"""
