"""Differential-drive crawl on ROUGH terrain (grass/brick-like bumps + mild inclines).

Same task as ``G1-Crawling-DiffDrive`` (tank-style translate-XOR-rotate crawl over the gait library,
reference-free actor, blended transitions), but trained on a generated rough-terrain grid instead of
a flat plane. The crawl library is flat, so its absolute torso height/orientation fight non-flat
ground; this task reconciles them with a HYBRID scheme: the reference height is made terrain-relative
(offset by the local ground height under the torso, a privileged sim-only query) while absolute torso
roll/pitch tracking is relaxed. The actor stays blind (no terrain sensor in its observation), so it
deploys exactly like DiffDrive; only the training signal is terrain-aware.
"""
