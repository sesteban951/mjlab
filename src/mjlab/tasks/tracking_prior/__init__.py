"""Motion tracking with a full-state control prior evaluated alongside the action.

Identical to ``tracking`` except that the joint-position action also evaluates a
prior computed from privileged state -- the reference motion's joint angles (see
``mdp/priors.py``), which the actor does not observe. The prior never drives the
robot; it is a reward signal (``mdp/rewards.py``).
"""
