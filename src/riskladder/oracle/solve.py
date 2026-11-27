"""Exact solvers for the stochastic shortest path problem (docs/env_spec.md section 6.2).

Costs are strictly positive and there is no discounting, so a policy that never terminates has
infinite cost. Value iteration started from zero then increases monotonically to the optimal value,
and policy iteration started from a *proper* policy (one that terminates with probability 1) keeps
every iterate proper.

Some cells can be trapped: a pocket of ground sealed off from every goal and hazard. No policy
terminates from there, so their value is infinite. They are detected once and excluded.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu

from riskladder.oracle.types import Mdp, Policy, Solution

_TOL_REL = 1e-10  # a policy-iteration improvement must exceed this to change the action


def _terminal_probability(mdp: Mdp) -> np.ndarray:
    return mdp.p_success + mdp.p_fail


def _steps_to_terminate(adjacency: sparse.spmatrix, terminating: np.ndarray) -> np.ndarray:
    """Fewest steps to reach SUCCESS/FAIL following edges of `adjacency`; inf if impossible.

    `terminating[s]` marks states with an edge into an absorbing state (distance 1). A virtual node
    stands for "terminated"; a single-source BFS on the reversed graph gives all distances at once.
    """
    n = adjacency.shape[0]
    edges = adjacency.tocoo()
    sources = np.flatnonzero(terminating)
    rows = np.concatenate([edges.col, np.full(len(sources), n)])  # reversed: t -> s, virtual -> s
    cols = np.concatenate([edges.row, sources])
    reverse = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n + 1, n + 1))
    dist = csgraph.dijkstra(reverse, directed=True, indices=n, unweighted=True)
    return dist[:n]


def _action_union(mdp: Mdp) -> sparse.csr_matrix:
    n, a = mdp.n_states, mdp.n_actions
    union = mdp.P_stack[:n]
    for k in range(1, a):
        union = union + mdp.P_stack[k * n : (k + 1) * n]
    return union.tocsr()


def distance_to_termination(mdp: Mdp) -> np.ndarray:
    """Fewest steps to SUCCESS/FAIL over all action choices, with positive probability."""
    terminating = _terminal_probability(mdp).max(axis=1) > 0
    return _steps_to_terminate(_action_union(mdp), terminating)


def live_mask(mdp: Mdp) -> np.ndarray:
    """States from which at least one policy terminates; the others are trapped."""
    return np.isfinite(distance_to_termination(mdp))


def proper_policy(mdp: Mdp) -> Policy:
    """A policy that terminates with probability 1 from every live state.

    Each live state picks the action with the largest probability of moving one layer closer to
    termination, where layers are fewest-step distances. Progress has positive probability at
    every step, so termination is certain (geometric tail).
    """
    n, a = mdp.n_states, mdp.n_actions
    layer = distance_to_termination(mdp)
    coo = mdp.P_stack.tocoo()
    state = coo.row % n
    closer = np.isfinite(layer[coo.col]) & (layer[coo.col] < layer[state])
    progress = np.bincount(coo.row, weights=coo.data * closer, minlength=a * n).reshape(a, n).T
    progress = progress + _terminal_probability(mdp)
    policy = np.argmax(progress, axis=1)
    policy[~np.isfinite(layer)] = 0
    return policy


def policy_matrix(mdp: Mdp, policy: Policy) -> sparse.csr_matrix:
    """n x n transition matrix among decision states under a deterministic policy."""
    n = mdp.n_states
    return mdp.P_stack[np.asarray(policy) * n + np.arange(n)]


def policy_vectors(mdp: Mdp, policy: Policy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-step expected cost, success probability and failure probability under a policy."""
    idx = (np.arange(mdp.n_states), np.asarray(policy))
    return mdp.cost[idx], mdp.p_success[idx], mdp.p_fail[idx]


def is_proper(mdp: Mdp, policy: Policy) -> bool:
    """True if `policy` terminates with probability 1 from every live state."""
    live = live_mask(mdp)
    _, p_success, p_fail = policy_vectors(mdp, policy)
    dist = _steps_to_terminate(policy_matrix(mdp, policy), (p_success + p_fail) > 0)
    return bool(np.all(np.isfinite(dist[live])))


def solve_policy(mdp: Mdp, policy: Policy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unbounded-horizon `(V, P(success), P(fail))` of a proper policy by one sparse factorisation.

    Values are 0 for trapped states, which no live state can reach.
    """
    live = live_mask(mdp)
    if not is_proper(mdp, policy):
        raise ValueError("policy is improper: it does not terminate from every live state")
    n = mdp.n_states
    cost, p_success, p_fail = policy_vectors(mdp, policy)
    idx = np.flatnonzero(live)
    P = policy_matrix(mdp, policy)[idx][:, idx]
    system = (sparse.identity(len(idx), format="csc") - P).tocsc()
    rhs = np.column_stack([cost[idx], p_success[idx], p_fail[idx]])
    solved = splu(system).solve(rhs)
    out = np.zeros((n, 3))
    out[idx] = solved
    return out[:, 0], out[:, 1], out[:, 2]


def _greedy(mdp: Mdp, V: np.ndarray) -> np.ndarray:
    """Q-values `cost + P V` as an (A, n) array."""
    a, n = mdp.n_actions, mdp.n_states
    flat_cost = np.ascontiguousarray(mdp.cost.T).ravel()
    return (flat_cost + mdp.P_stack @ V).reshape(a, n)


def value_iteration(mdp: Mdp, tol: float = 1e-9, max_iter: int = 1_000_000) -> Solution:
    """Jacobi value iteration from V = 0; stops when the largest Bellman residual is below `tol`.

    The residual is not a bound on the error (undiscounted problems need roughly `residual` times
    the expected number of steps), so tests compare against the exact policy-iteration values.
    """
    n = mdp.n_states
    live = live_mask(mdp)
    V = np.zeros(n)
    residual = np.inf
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        new = _greedy(mdp, V).min(axis=0)
        new[~live] = 0.0
        residual = float(np.abs(new - V).max())
        V = new
        if residual < tol:
            break
    policy = _greedy(mdp, V).argmin(axis=0)
    policy[~live] = 0
    V = np.where(live, V, np.inf)
    return Solution(V, policy, iterations, residual, residual < tol, live)


def policy_iteration(mdp: Mdp, init: Policy | None = None, max_iter: int = 1000) -> Solution:
    """Policy iteration with exact (sparse direct) evaluation, started from a proper policy.

    Ties keep the current action, so iteration cannot cycle between equally good actions.
    """
    n = mdp.n_states
    live = live_mask(mdp)
    policy = proper_policy(mdp) if init is None else np.asarray(init).copy()
    arange = np.arange(n)
    converged = False
    V = np.zeros(n)
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        V, _, _ = solve_policy(mdp, policy)
        q = _greedy(mdp, V)
        best = q.min(axis=0)
        better = q[policy, arange] - best > _TOL_REL * np.maximum(1.0, np.abs(best))
        new = np.where(better, q.argmin(axis=0), policy)
        new[~live] = 0
        if np.array_equal(new, policy):
            converged = True
            break
        policy = new
    return Solution(np.where(live, V, np.inf), policy, iterations, 0.0, converged, live)
