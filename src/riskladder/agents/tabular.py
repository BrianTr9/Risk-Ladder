"""Tabular learners: Q-learning and SARSA on the cell grid (docs/env_spec.md section 8).

Both learn the expected cost-to-go `Q[row, col, action]` from sampled transitions only. Costs are
positive, so a table of zeros is an optimistic start: an action nobody has tried looks cheap, which
pulls the learner towards it. A failure ends the episode and its sampled cost already contains the
failure cost, so the update target at a terminal step is the sampled cost itself. A cut at the
horizon is not a real end, so the target still bootstraps from the next cell.
"""

from __future__ import annotations

import numpy as np

from riskladder.env.dynamics import N_ACTIONS
from riskladder.env.layout import Layout


class EpsilonGreedy:
    """Shared machinery: the table, the step-size rule and the linear exploration schedule."""

    name = "tabular"

    def __init__(
        self,
        alpha: float = 0.3,
        alpha_decay: float = 0.0,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_fraction: float = 0.5,
        q_init: float = 0.0,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if alpha_decay < 0.0:
            raise ValueError("alpha_decay must be non-negative")
        if not 0.0 <= eps_end <= eps_start <= 1.0:
            raise ValueError("need 0 <= eps_end <= eps_start <= 1")
        if not 0.0 < eps_fraction <= 1.0:
            raise ValueError("eps_fraction must be in (0, 1]")
        self.alpha, self.alpha_decay = alpha, alpha_decay
        self.eps_start, self.eps_end, self.eps_fraction = eps_start, eps_end, eps_fraction
        self.q_init = q_init
        self.config = {
            "alpha": alpha, "alpha_decay": alpha_decay, "eps_start": eps_start,
            "eps_end": eps_end, "eps_fraction": eps_fraction, "q_init": q_init,
        }  # fmt: skip

    def begin(self, layout: Layout, budget_steps: int, rng: np.random.Generator) -> None:
        self.shape = (layout.rows, layout.cols)
        self.Q = np.full((*self.shape, N_ACTIONS), self.q_init)
        self.visits = np.zeros((*self.shape, N_ACTIONS), dtype=np.int64)
        self.rng = rng
        self.t = 0
        self._ramp = max(1.0, self.eps_fraction * budget_steps)

    def start_episode(self, layout: Layout) -> None:
        if (layout.rows, layout.cols) != self.shape:
            raise ValueError("a tabular learner is tied to the map it started on")

    def epsilon(self) -> float:
        frac = min(1.0, self.t / self._ramp)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def _choose(self, cell: tuple[int, int]) -> int:
        if self.rng.random() < self.epsilon():
            return int(self.rng.integers(N_ACTIONS))
        return int(np.argmin(self.Q[cell]))

    def _update(self, cell: tuple[int, int], action: int, target: float) -> None:
        key = (*cell, action)
        step = self.alpha / (1.0 + self.alpha_decay * self.visits[key])
        self.visits[key] += 1
        self.Q[key] += step * (target - self.Q[key])

    def greedy(self, layout: Layout, cells: np.ndarray) -> np.ndarray:
        return np.argmin(self.Q[cells[:, 0], cells[:, 1]], axis=1)


class QLearning(EpsilonGreedy):
    """Off-policy: the target uses the best next action, whatever the learner does next."""

    name = "q"

    def act(self, cell: tuple[int, int]) -> int:
        return self._choose(cell)

    def observe(
        self,
        cell: tuple[int, int],
        action: int,
        cost: float,
        next_cell: tuple[int, int],
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.t += 1
        target = cost if terminated else cost + float(self.Q[next_cell].min())
        self._update(cell, action, target)


class SARSA(EpsilonGreedy):
    """On-policy: the target uses the action the exploring learner will really take next."""

    name = "sarsa"

    def begin(self, layout: Layout, budget_steps: int, rng: np.random.Generator) -> None:
        super().begin(layout, budget_steps, rng)
        self._next: int | None = None

    def start_episode(self, layout: Layout) -> None:
        super().start_episode(layout)
        self._next = None

    def act(self, cell: tuple[int, int]) -> int:
        if self._next is not None:
            action, self._next = self._next, None
            return action
        return self._choose(cell)

    def observe(
        self,
        cell: tuple[int, int],
        action: int,
        cost: float,
        next_cell: tuple[int, int],
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.t += 1
        if terminated:
            self._update(cell, action, cost)
            return
        following = self._choose(next_cell)
        self._update(cell, action, cost + float(self.Q[(*next_cell, following)]))
        self._next = None if truncated else following
