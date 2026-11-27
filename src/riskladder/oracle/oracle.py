"""`Oracle`: the single entry point that scores any policy against the exact optimum of a map."""

from __future__ import annotations

from functools import cached_property

from riskladder.oracle.evaluate import evaluate, evaluate_mixture
from riskladder.oracle.solve import value_iteration
from riskladder.oracle.types import Mdp, Mixture, Policy, PolicyResult, Solution


class Oracle:
    """Exact scoring of policies on one MDP.

    The optimum is solved once. `evaluate` returns the exact expected cost, P(success), P(fail) and
    P(timeout) of a policy (or of a mixture drawn once per episode) over the episode horizon, and
    `regret` is its expected cost minus the optimal policy's expected cost over the same horizon.
    """

    def __init__(self, mdp: Mdp, horizon: int | None = None, vi_tol: float = 1e-10) -> None:
        self.mdp = mdp
        self.horizon = mdp.horizon if horizon is None else horizon
        self._vi_tol = vi_tol
        self._optimal_results: dict[int, PolicyResult] = {}

    @cached_property
    def optimum(self) -> Solution:
        return value_iteration(self.mdp, tol=self._vi_tol)

    def evaluate(self, policy: Policy | Mixture, start: int | None = None) -> PolicyResult:
        start = self.mdp.start if start is None else start
        return evaluate_mixture(self.mdp, policy, self.horizon, start=start)

    def optimal(self, start: int | None = None) -> PolicyResult:
        start = self.mdp.start if start is None else start
        if start not in self._optimal_results:
            result = evaluate(self.mdp, self.optimum.policy, self.horizon, start=start)
            self._optimal_results[start] = result.at(start)
        return self._optimal_results[start]

    def regret(self, policy: Policy | Mixture, start: int | None = None) -> float:
        return self.evaluate(policy, start).cost - self.optimal(start).cost
