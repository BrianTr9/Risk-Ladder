"""Shared types of the oracle: the MDP protocol, policies, mixtures and results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import sparse

# A deterministic policy: action index for each decision state, shape (n_states,).
Policy = np.ndarray
# A mixture: one policy chosen once at the start of an episode, with the given probability.
Mixture = tuple[tuple[float, Policy], ...]


class Mdp(Protocol):
    """What the oracle needs from an MDP. `riskladder.env.SparseMDP` satisfies it.

    Row `a * n_states + s` of `P_stack` holds the transition probabilities among decision states of
    action `a` from state `s`; the remaining probability mass is absorption into SUCCESS or FAIL.
    """

    n_states: int
    n_actions: int
    start: int
    horizon: int
    P_stack: sparse.csr_matrix
    p_success: np.ndarray  # (n, A)
    p_fail: np.ndarray  # (n, A)
    cost: np.ndarray  # (n, A) expected one-step cost


@dataclass(frozen=True)
class Solution:
    """Result of value or policy iteration (stochastic shortest path, no discounting)."""

    V: np.ndarray  # optimal expected cost-to-go; inf in trapped states
    policy: Policy
    iterations: int
    residual: float  # last Bellman residual (value iteration) or 0.0 (policy iteration)
    converged: bool
    live: np.ndarray  # states from which some policy terminates; the rest are trapped


@dataclass(frozen=True)
class PolicyResult:
    """Exact outcome of running a policy from one state for at most the horizon."""

    cost: float  # expected total cost, including c_fail on failure
    p_success: float
    p_fail: float
    p_timeout: float

    @classmethod
    def combine(cls, weights: Sequence[float], results: Sequence[PolicyResult]) -> PolicyResult:
        """Weighted average; exact for a mixture because every quantity is linear in the weights."""
        return cls(
            *(
                float(sum(w * getattr(r, name) for w, r in zip(weights, results, strict=True)))
                for name in ("cost", "p_success", "p_fail", "p_timeout")
            )
        )


@dataclass(frozen=True)
class Evaluation:
    """`PolicyResult` for every state (NaN for states outside the evaluated set)."""

    cost: np.ndarray
    p_success: np.ndarray
    p_fail: np.ndarray
    horizon: int
    steps: int  # series terms actually summed (equals `horizon` unless stopped early)

    @property
    def p_timeout(self) -> np.ndarray:
        return np.clip(1.0 - self.p_success - self.p_fail, 0.0, 1.0)

    def at(self, state: int) -> PolicyResult:
        return PolicyResult(
            float(self.cost[state]),
            float(self.p_success[state]),
            float(self.p_fail[state]),
            float(self.p_timeout[state]),
        )


def as_mixture(policy: Policy | Mixture) -> Mixture:
    """Accept a single policy or a mixture and return a validated mixture."""
    if isinstance(policy, np.ndarray):
        return ((1.0, policy),)
    weights = np.array([w for w, _ in policy], dtype=float)
    if len(weights) == 0 or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("mixture weights must be non-negative and sum to 1")
    return tuple((float(w), np.asarray(p)) for w, p in policy)
