"""Exact solvers and evaluation: value/policy iteration, finite-horizon policy evaluation, CVaR."""

from riskladder.oracle.evaluate import evaluate, evaluate_mixture
from riskladder.oracle.oracle import Oracle
from riskladder.oracle.rollout import CVaR, Rollouts, cvar, rollout
from riskladder.oracle.solve import (
    distance_to_termination,
    is_proper,
    live_mask,
    policy_iteration,
    proper_policy,
    solve_policy,
    value_iteration,
)
from riskladder.oracle.types import (
    Evaluation,
    Mdp,
    Mixture,
    Policy,
    PolicyResult,
    Solution,
    as_mixture,
)

__all__ = [
    "CVaR",
    "Evaluation",
    "Mdp",
    "Mixture",
    "Oracle",
    "Policy",
    "PolicyResult",
    "Rollouts",
    "Solution",
    "as_mixture",
    "cvar",
    "distance_to_termination",
    "evaluate",
    "evaluate_mixture",
    "is_proper",
    "live_mask",
    "policy_iteration",
    "proper_policy",
    "rollout",
    "solve_policy",
    "value_iteration",
]
