"""Constrained MDP: minimise expected cost subject to P(fail) <= delta.

The constraint is handled by Lagrangian relaxation. For a multiplier `lam >= 0`, the policy that
minimises `cost + lam * P(fail)` is the penalty planner with `extra = lam`. As `lam` grows the
failure probability of that policy falls in jumps, because only finitely many deterministic policies
exist. The constrained optimum sits at a jump: two deterministic policies tie in the Lagrangian,
one just above the bound and one just below, and the optimum is a mixture of the two (Altman, 1999,
"Constrained Markov Decision Processes"). Choosing the policy once at the start of an episode with
the right probability makes `P(fail) = delta` exactly, because the failure probability of a
mixture is the weighted average of its parts.

Weak duality gives a certificate. For any `lam >= 0` and any policy with `P(fail) <= delta`,

    cost >= V_lam(start) - lam * delta,

where `V_lam` is the optimal Lagrangian value. The gap between the mixture's cost and this bound
shows how close to optimal the solution is, without trusting any other implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskladder.agents.planners import PenalizedMDP
from riskladder.oracle.evaluate import evaluate_mixture
from riskladder.oracle.solve import value_iteration
from riskladder.oracle.types import Mdp, Mixture, Policy, PolicyResult

MAX_MULTIPLIER = 1e9  # beyond this the Lagrangian policy no longer changes in any useful way
FEASIBILITY_SLACK = 1e-12


@dataclass(frozen=True)
class CmdpSolution:
    """Solution of the constrained problem for one bound `delta`."""

    delta: float
    mixture: Mixture  # policies and their probabilities, chosen once per episode
    result: PolicyResult  # exact outcome of the mixture from the start state
    multiplier: float  # the Lagrange multiplier at the optimum (0 if the bound does not bind)
    lower_bound: float  # dual bound on the cost of any policy with P(fail) <= delta
    feasible: bool  # False if no policy reaches P(fail) <= delta
    penalty_policy: Policy  # deterministic policy meeting the bound at the smallest penalty

    @property
    def gap(self) -> float:
        """Cost of the mixture above the dual bound; 0 means provably optimal."""
        return self.result.cost - self.lower_bound


def _lagrangian(mdp: Mdp, lam: float, tol: float) -> tuple[Policy, float, float]:
    """Optimal policy of the Lagrangian, its failure probability and its value at the start."""
    solution = value_iteration(PenalizedMDP(mdp, lam), tol=tol)
    result = evaluate_mixture(mdp, solution.policy)
    return solution.policy, result.p_fail, float(solution.V[mdp.start])


def solve_cmdp(mdp: Mdp, delta: float, tol: float = 1e-10, max_iter: int = 200) -> CmdpSolution:
    """Cheapest way to keep the failure probability at most `delta`, mixing at most two policies."""
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be in [0, 1]")

    def meets(fail: float) -> bool:
        return fail <= delta + FEASIBILITY_SLACK

    policy, fail, value = _lagrangian(mdp, 0.0, tol)
    if meets(fail):  # the bound does not bind
        result = evaluate_mixture(mdp, policy)
        return CmdpSolution(delta, ((1.0, policy),), result, 0.0, result.cost, True, policy)
    lam_lo, lo_policy, lo_fail = 0.0, policy, fail  # largest multiplier known to be infeasible

    lam_hi = 1.0  # grow until the Lagrangian policy meets the bound
    while True:
        hi_policy, hi_fail, hi_value = _lagrangian(mdp, lam_hi, tol)
        if meets(hi_fail):
            break
        lam_lo, lo_policy, lo_fail = lam_hi, hi_policy, hi_fail
        lam_hi *= 4.0
        if (
            lam_hi > MAX_MULTIPLIER
        ):  # no policy reaches the bound (e.g. a hazard that cannot be avoided)
            result = evaluate_mixture(mdp, hi_policy)
            return CmdpSolution(
                delta, ((1.0, hi_policy),), result, lam_lo, -np.inf, False, hi_policy
            )

    for _ in range(
        max_iter
    ):  # bisect between the last infeasible and the first feasible multiplier
        if lam_hi - lam_lo <= 1e-13 * max(1.0, lam_hi):
            break
        mid = 0.5 * (lam_lo + lam_hi)
        policy, fail, value = _lagrangian(mdp, mid, tol)
        if meets(fail):
            lam_hi, hi_policy, hi_fail, hi_value = mid, policy, fail, value
        else:
            lam_lo, lo_policy, lo_fail = mid, policy, fail

    mixture: Mixture = ((1.0, hi_policy),)
    if lo_fail - hi_fail > FEASIBILITY_SLACK:
        weight_lo = min(1.0, max(0.0, (delta - hi_fail) / (lo_fail - hi_fail)))
        parts = ((weight_lo, lo_policy), (1.0 - weight_lo, hi_policy))
        kept = [part for part in parts if part[0] > 1e-9]  # drop negligible parts, then renormalise
        total = sum(weight for weight, _ in kept)
        mixture = tuple((weight / total, policy) for weight, policy in kept)
    result = evaluate_mixture(mdp, mixture)
    return CmdpSolution(delta, mixture, result, lam_hi, hi_value - lam_hi * delta, True, hi_policy)


def frontier(mdp: Mdp, deltas: list[float], **kwargs) -> list[CmdpSolution]:
    """The cost/risk frontier: one solution per bound."""
    return [solve_cmdp(mdp, d, **kwargs) for d in deltas]
