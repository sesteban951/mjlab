"""Motion tracking with a full-state control prior blended into the action.

Identical to ``tracking`` except that the joint-position action is a convex
combination of the policy target and a prior computed from privileged state::

    u = (1 - lam) * pi(o) + lam * u_prior(s)

The prior here is the reference motion's joint angles (see ``mdp/priors.py``),
which the actor does not observe. ``lam`` is annealed to zero by a curriculum,
so training starts near the reference and ends as a pure policy.
"""
