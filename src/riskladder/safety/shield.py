"""Safety shield: block actions that could lose the rover in one step (docs/learning.md).

The shield sees the map and an upper bound `p_max` on the slip probability, nothing else (no costs,
no true noise; docs/env_spec.md section 8). For every cell and action it computes the largest
one-step probability of entering a hazard that is still possible under `p_max` and blocks the action
when that exceeds the threshold `theta`.

What it guarantees, and what it does not. It bounds the risk of each *step*, so the risk of a whole
episode is bounded only by the number of steps taken near hazards times `theta`; it is not a
constraint on P(fail) (that is the constrained solver's job). And a shield can paint the rover into
a corner: if the blocked actions leave no route to the goal from a cell, the threshold is raised
for that cell just enough to keep one (the cell is reported in `Shield.relaxed`), so some action and
a route to the goal always remain.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from riskladder.env.dynamics import FAIL, N_ACTIONS, SUCCESS, branch_table
from riskladder.env.layout import PASSABLE, Layout

_EPS = 1e-12


@dataclass(frozen=True)
class Shield:
    p_max: float = 0.4  # assumed upper bound on the slip probability of any single action
    theta: float = 0.05  # largest one-step failure probability the shield tolerates

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_max <= 1.0:
            raise ValueError("p_max must be in [0, 1]")
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError("theta must be in [0, 1]")

    def risk(self, layout: Layout) -> np.ndarray:
        """`risk[row, col, action]`: worst-case probability of entering a hazard in one step."""
        return _branches(layout, self.p_max)[0]

    def analyse(self, layout: Layout) -> ShieldTable:
        risk, landing = _branches(layout, self.p_max)
        rows, cols = layout.rows, layout.cols
        route = _minimax_route(layout, risk, landing)  # least worst step on any route to the goal
        passable = layout.mask(PASSABLE)
        allowed = np.ones((rows, cols, N_ACTIONS), dtype=bool)
        relaxed = np.zeros((rows, cols), dtype=bool)
        for r, c in np.argwhere(passable):
            bar = max(self.theta, route[r, c])  # the cell's own threshold
            if not np.isfinite(bar):  # no route at all: keep the least risky actions
                allowed[r, c] = risk[r, c] <= risk[r, c].min() + _EPS
                relaxed[r, c] = True
                continue
            ok = risk[r, c] <= bar + _EPS
            for a in range(N_ACTIONS):
                lr, lc = landing[r, c, a]
                if ok[a] and lr >= 0:
                    ok[a] = route[lr, lc] <= bar + _EPS
                elif ok[a]:
                    ok[a] = lr == _GOAL  # a hazard landing is never a route
            allowed[r, c] = ok
            relaxed[r, c] = route[r, c] > self.theta + _EPS
        allowed[~passable] = True
        return ShieldTable(allowed, relaxed, risk)

    def table(self, layout: Layout) -> np.ndarray:
        """Boolean `allowed[row, col, action]`; at least one action is allowed in every cell."""
        return self.analyse(layout).allowed

    @property
    def config(self) -> dict:
        return {"p_max": self.p_max, "theta": self.theta}


@dataclass(frozen=True)
class ShieldTable:
    allowed: np.ndarray  # (rows, cols, A) bool
    relaxed: np.ndarray  # (rows, cols) bool: cells whose threshold had to be raised
    risk: np.ndarray  # (rows, cols, A) worst-case one-step failure probability


_GOAL, _HAZARD = -1, -2  # `landing` markers for SUCCESS and FAIL


def _branches(layout: Layout, p_max: float) -> tuple[np.ndarray, np.ndarray]:
    """One-step failure probability under `p_max`, and where each action lands when it goes as
    intended: (row, col), or (_GOAL, _GOAL) / (_HAZARD, _HAZARD)."""
    b = branch_table(layout)
    fail = b.status == FAIL
    risk = (1.0 - p_max) * fail[..., 0] + 0.5 * p_max * fail[..., 1:].sum(axis=-1)
    landing = np.stack([b.rows[..., 0], b.cols[..., 0]], axis=-1)
    landing[b.status[..., 0] == SUCCESS] = _GOAL
    landing[fail[..., 0]] = _HAZARD
    return risk, landing


def _minimax_route(layout: Layout, risk: np.ndarray, landing: np.ndarray) -> np.ndarray:
    """`route[r, c]`: the smallest, over routes to the goal that follow intended outcomes, of the
    largest one-step risk along the route (a bottleneck shortest path, found backwards from the
    goal). Infinite where no such route exists."""
    rows, cols = layout.rows, layout.cols
    route = np.full((rows, cols), np.inf)
    incoming: dict[tuple[int, int] | int, list[tuple[tuple[int, int], float]]] = {}
    for r, c in np.argwhere(layout.mask(PASSABLE)):
        for a in range(N_ACTIONS):
            lr, lc = landing[r, c, a]
            if lr == _HAZARD:
                continue
            key = _GOAL if lr == _GOAL else (int(lr), int(lc))
            incoming.setdefault(key, []).append(((int(r), int(c)), float(risk[r, c, a])))
    heap: list[tuple[float, int, tuple[int, int] | int]] = [(0.0, 0, _GOAL)]
    done: set = set()
    tie = 0
    while heap:
        value, _, node = heapq.heappop(heap)
        if node in done:
            continue
        done.add(node)
        for cell, weight in incoming.get(node, []):
            worst = max(weight, value)
            if worst < route[cell]:
                route[cell] = worst
                tie += 1
                heapq.heappush(heap, (worst, tie, cell))
    return route
