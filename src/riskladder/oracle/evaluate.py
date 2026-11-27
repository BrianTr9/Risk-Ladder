"""Exact evaluation of any policy over the episode horizon (docs/env_spec.md section 6.4).

For a deterministic policy with transition matrix `P` among decision states and one-step vector
`b = [expected cost, P(success), P(fail)]`, the horizon-`H` outcome is the finite series

    X_H = sum_{k=0}^{H-1} P^k b

which is exact, never singular (so it also handles policies that loop forever), and linear in `b`.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csgraph

from riskladder.oracle.solve import policy_matrix, policy_vectors
from riskladder.oracle.types import (
    Evaluation,
    Mdp,
    Mixture,
    Policy,
    PolicyResult,
    as_mixture,
)


def evaluate(
    mdp: Mdp,
    policy: Policy,
    horizon: int | None = None,
    tol: float = 1e-12,
    start: int | None = None,
) -> Evaluation:
    """Expected cost, P(success), P(fail) (and P(timeout) = 1 - the others) within the horizon.

    `tol` allows stopping the series early. Every term is bounded by the previous one in max-norm
    (P is substochastic), so after summing terms up to `k` the remaining error is at most
    `(H - 1 - k) * max|term_{k+1}|`; the series stops once that bound is below `tol`. Use `tol=0`
    to always sum all `H` terms. Policies that never terminate keep large terms and run all `H`.

    If `start` is given, only the states reachable from it under the policy are evaluated (other
    entries are NaN). This is much faster on large maps and is what regret computations need.
    """
    n = mdp.n_states
    horizon = mdp.horizon if horizon is None else horizon
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    P = policy_matrix(mdp, policy)
    b = np.column_stack(policy_vectors(mdp, policy))

    keep = None
    if start is not None:
        keep = np.sort(
            csgraph.breadth_first_order(P, start, directed=True, return_predecessors=False)
        )
        P = P[keep][:, keep]
        b = b[keep]

    total = b.copy()
    term = b
    steps = 1  # terms summed so far
    while steps < horizon:
        term = P @ term
        if tol > 0 and (horizon - steps) * np.abs(term).max() <= tol:
            break
        total += term
        steps += 1

    if keep is None:
        full = total
    else:
        full = np.full((n, 3), np.nan)
        full[keep] = total
    return Evaluation(full[:, 0], full[:, 1], full[:, 2], horizon, steps)


def evaluate_mixture(
    mdp: Mdp,
    mixture: Policy | Mixture,
    horizon: int | None = None,
    tol: float = 1e-12,
    start: int | None = None,
) -> PolicyResult:
    """Outcome from `start` (default: the map's start) of a policy drawn once per episode.

    Cost and the outcome probabilities are linear in the mixture weights, so the result is the
    weighted average of each component policy's exact result.
    """
    start = mdp.start if start is None else start
    parts = as_mixture(mixture)
    results = [evaluate(mdp, pi, horizon, tol, start).at(start) for _, pi in parts]
    return PolicyResult.combine([w for w, _ in parts], results)
