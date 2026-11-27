"""Vectorised Monte Carlo rollouts and CVaR.

Everything here is an *estimate*: use it for tail risk and for checking the exact evaluation.
Expected cost and failure probability come from `riskladder.oracle.evaluate`, which is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from riskladder.env.dynamics import FAIL, RUNNING, SUCCESS
from riskladder.env.model import NEXT_FAIL, NEXT_SUCCESS, SparseMDP
from riskladder.oracle.types import Mixture, Policy, as_mixture


@dataclass(frozen=True)
class Rollouts:
    """Outcomes of `n` simulated episodes. `status` is SUCCESS, FAIL, or RUNNING for a timeout."""

    cost: np.ndarray
    status: np.ndarray
    steps: np.ndarray

    @property
    def n(self) -> int:
        return len(self.cost)

    def rate(self, status: int) -> float:
        return float(np.mean(self.status == status))

    @property
    def success_costs(self) -> np.ndarray:
        """Mission cost of the episodes that reached a goal (the input of CVaR)."""
        return self.cost[self.status == SUCCESS]


def rollout(
    mdp: SparseMDP,
    policy: Policy | Mixture,
    n: int,
    rng: np.random.Generator | int | None = None,
    start: int | None = None,
    horizon: int | None = None,
) -> Rollouts:
    """Simulate `n` episodes at once; one time step is a few vectorised operations over all of them.

    The sampling rule is the one used by `RoverEnv.step`, so results agree with the exact model.
    A mixture picks one policy per episode at the start.
    """
    rng = np.random.default_rng(rng)
    parts = as_mixture(policy)
    table = np.stack([pi for _, pi in parts])
    choice = rng.choice(len(parts), size=n, p=[w for w, _ in parts])
    horizon = mdp.horizon if horizon is None else horizon
    cumulative = np.cumsum(mdp.branch_prob, axis=1)

    state = np.full(n, mdp.start if start is None else start)
    cost = np.zeros(n)
    steps = np.zeros(n, dtype=np.int64)
    status = np.full(n, RUNNING, dtype=np.int8)
    active = np.arange(n)
    for _ in range(horizon):
        if active.size == 0:
            break
        s = state[active]
        a = table[choice[active], s]
        u = rng.random(active.size)
        branch = np.minimum((u[:, None] >= cumulative[a]).sum(axis=1), cumulative.shape[1] - 1)
        nxt = mdp.branch_next[s, a, branch]
        cost[active] += mdp.branch_cost[s, a, branch]
        steps[active] += 1
        status[active[nxt == NEXT_SUCCESS]] = SUCCESS
        status[active[nxt == NEXT_FAIL]] = FAIL
        running = nxt >= 0
        state[active[running]] = nxt[running]
        active = active[running]
    return Rollouts(cost, status, steps)


@dataclass(frozen=True)
class CVaR:
    """Mean of the worst `1 - alpha` fraction of costs, with a bootstrap confidence interval."""

    value: float
    ci_low: float
    ci_high: float
    alpha: float
    n: int


def _tail_mean(costs: np.ndarray, k: int) -> float:
    return float(np.partition(costs, len(costs) - k)[len(costs) - k :].mean())


def cvar(
    costs: np.ndarray,
    alpha: float = 0.95,
    n_boot: int = 200,
    confidence: float = 0.95,
    rng: np.random.Generator | int | None = None,
) -> CVaR:
    """Conditional value at risk: the mean of the largest `ceil((1 - alpha) * n)` costs."""
    costs = np.asarray(costs, dtype=float)
    n = len(costs)
    k = math.ceil((1.0 - alpha) * n - 1e-9)  # tolerance: (1 - 0.95) * 100 is 5.000000000000004
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n == 0 or k < 1:
        raise ValueError("need at least one tail sample")
    rng = np.random.default_rng(rng)
    boot = [_tail_mean(costs[rng.integers(0, n, n)], k) for _ in range(n_boot)]
    low, high = np.percentile(boot, [50 * (1 - confidence), 100 - 50 * (1 - confidence)])
    return CVaR(_tail_mean(costs, k), float(low), float(high), alpha, n)
