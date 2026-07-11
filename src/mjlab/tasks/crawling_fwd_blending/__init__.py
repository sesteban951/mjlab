"""Twist-conditioned crawl gait-library tracking, with blended clip transitions.

Same as ``G1-Crawling-Fwd`` (twist-indexed gait library, reference-free actor, idle-pose stop) but
mid-episode twist resamples do NOT hard-switch the tracked reference. Instead the reference is
interpolated from the outgoing clip to the incoming clip over a short window, so the imitation
target is a smooth deceleration/acceleration rather than a teleport across the library's velocity
gap. The commanded-twist observation still steps at the resample, so the policy learns an intrinsic
graceful transition (settling via proprioceptive feedback) rather than merely following a ramped
command. RSI at episode reset is unaffected (instant, no blend).
"""
