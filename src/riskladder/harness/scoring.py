"""Exact scoring of a learner's greedy policy against the oracle (docs/env_spec.md section 6)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from riskladder.env import RoverEnv
from riskladder.oracle import Mixture, Oracle, PolicyResult

if TYPE_CHECKING:
    from riskladder.harness.train import Learner


@dataclass(frozen=True)
class Score:
    """Exact outcome of the greedy policy from the start cell, over the episode horizon."""

    regret: float  # expected cost minus the optimum's, both over the horizon
    cost: float
    p_success: float
    p_fail: float
    p_timeout: float

    @classmethod
    def of(cls, result: PolicyResult, optimal_cost: float) -> Score:
        return cls(
            result.cost - optimal_cost, result.cost, result.p_success, result.p_fail,
            result.p_timeout,
        )  # fmt: skip


class Scorer:
    """Holds the exact model of one map. This is the harness's privilege, never the learner's:
    the learner only ever sees `SampleEnv`."""

    def __init__(self, env: RoverEnv, tau: float = 0.05) -> None:
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must be in (0, 1)")
        self.layout = env.layout
        self.mdp = env.model()
        self.oracle = Oracle(self.mdp)
        self.tau = tau

    @cached_property
    def optimal(self) -> PolicyResult:
        return self.oracle.optimal()

    @property
    def threshold(self) -> float:
        """A learner counts as having learned the map once its regret is at most this."""
        return self.tau * self.optimal.cost

    def deployed(self, learner: Learner) -> np.ndarray | Mixture:
        """What the learner would run: its greedy policy, or the mixture of policies (chosen once
        per episode) when it plans with the constrained solver."""
        if hasattr(learner, "mixture"):
            parts = learner.mixture(self.layout, self.mdp.cells)
            return tuple((float(w), self._checked(np.asarray(a, dtype=int))) for w, a in parts)
        return self.policy(learner)

    def _checked(self, actions: np.ndarray) -> np.ndarray:
        if actions.shape != (self.mdp.n_states,):
            raise ValueError("a learner must return one action per decision cell")
        return actions

    def policy(self, learner: Learner) -> np.ndarray:
        return self._checked(np.asarray(learner.greedy(self.layout, self.mdp.cells), dtype=int))

    def score(self, learner: Learner) -> Score:
        return Score.of(self.oracle.evaluate(self.deployed(learner)), self.optimal.cost)
