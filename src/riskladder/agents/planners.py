"""Planners that use the model: A* on the noise-free map and the penalty planner.

A* is rung 1 of the ladder (deterministic dynamics). The penalty planner is the baseline of rung 2:
it turns a risk preference into a number (an extra failure penalty) and solves the resulting MDP.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from riskladder.env.distances import steps_to_goal
from riskladder.env.dynamics import FAIL, N_ACTIONS, SUCCESS, outcomes
from riskladder.env.spec import MapSpec, Noise
from riskladder.oracle.solve import value_iteration
from riskladder.oracle.types import Mdp, Solution

# --- A* on the noise-free map ---------------------------------------------------------------


@dataclass(frozen=True)
class AStarResult:
    actions: tuple[int, ...]  # action indices from the start to the goal
    cost: float
    expanded: int  # states taken off the frontier
    found: bool


def cost_per_cell(spec: MapSpec) -> float:
    """Lower bound on the cost per covered cell: a move covers one, a dash at most two."""
    return min(spec.costs.move, spec.costs.dash / 2.0)


def heuristic_table(spec: MapSpec) -> np.ndarray:
    """`h(cell) = steps_to_goal * cost_per_cell` for every cell (inf if no goal is reachable).

    Admissible: every action costs at least `cost_per_cell` for each cell it covers (a collision
    covers none but costs more than a move), and a path of cost c covers at least `steps_to_goal`
    cells. This is with respect to the problem A* solves, a hazard-free route from a cell the
    search can reach; a cell with no such route has `h = inf`.

    Consistent: moving between two cells changes `steps_to_goal` by at most the number of cells
    covered, so `h(s) <= cost(s, a) + h(s')`. Both properties are checked by tests.
    """
    return steps_to_goal(spec.layout) * cost_per_cell(spec)


GOAL_NODE = (-1, -1)  # a virtual node: reaching any goal cell ends in it


def astar(spec: MapSpec, use_heuristic: bool = True) -> AStarResult:
    """Cheapest hazard-free route to a goal on the noise-free version of the map.

    With `use_heuristic=False` the search is Dijkstra, which makes the saving of the heuristic
    measurable. Because the failure cost exceeds any hazard-free route, the cheapest episode never
    ends in a hazard, so the result equals the optimal value of the noise-free MDP.

    The goal test happens when a node is taken off the frontier, not when it is generated: a goal
    can be reached early by an expensive route, and only the cheapest entry may end the search.
    """
    exact = spec.with_noise(Noise(p_slip=0.0, dash_slip_mult=0.0))
    h = heuristic_table(spec) if use_heuristic else None
    start = spec.layout.start
    best = {start: 0.0}
    parent: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    frontier = [(0.0 if h is None else float(h[start]), 0.0, start)]
    expanded = 0
    while frontier:
        _, g, cell = heapq.heappop(frontier)
        if g > best.get(cell, np.inf):
            continue  # stale entry
        if cell == GOAL_NODE:
            return AStarResult(_trace(parent, cell), g, expanded, True)
        expanded += 1
        for action in range(N_ACTIONS):
            (move,) = outcomes(exact, cell, action)  # one outcome: the dynamics are deterministic
            if move.status == FAIL:
                continue
            target = GOAL_NODE if move.status == SUCCESS else move.cell
            cost = g + move.cost
            if cost < best.get(target, np.inf):
                best[target] = cost
                parent[target] = (cell, action)
                estimate = 0.0 if h is None or target == GOAL_NODE else float(h[target])
                heapq.heappush(frontier, (cost + estimate, cost, target))
    return AStarResult((), float("inf"), expanded, False)


def _trace(parent: dict, cell: tuple[int, int]) -> tuple[int, ...]:
    actions: list[int] = []
    while cell in parent:
        cell, action = parent[cell]
        actions.append(action)
    return tuple(reversed(actions))


# --- penalty planner ----------------------------------------------------------------------


@dataclass(frozen=True)
class PenalizedMDP:
    """A model with an extra penalty on failure: `cost + extra * P(fail)`.

    The environment cost already contains `c_fail` for a failure, so the total price of a failure
    is `c_fail + extra`. `extra = 0` is the environment's own optimisation problem.
    """

    base: Mdp
    extra: float
    cost: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.extra < 0:
            raise ValueError("the extra penalty must be non-negative")
        object.__setattr__(self, "cost", self.base.cost + self.extra * self.base.p_fail)

    @property
    def n_states(self) -> int:
        return self.base.n_states

    @property
    def n_actions(self) -> int:
        return self.base.n_actions

    @property
    def start(self) -> int:
        return self.base.start

    @property
    def horizon(self) -> int:
        return self.base.horizon

    @property
    def P_stack(self) -> sparse.csr_matrix:
        return self.base.P_stack

    @property
    def p_success(self) -> np.ndarray:
        return self.base.p_success

    @property
    def p_fail(self) -> np.ndarray:
        return self.base.p_fail


def penalty_planner(mdp: Mdp, extra: float, tol: float = 1e-10) -> Solution:
    """Optimal policy when a failure costs `extra` more than the environment already charges."""
    return value_iteration(PenalizedMDP(mdp, extra), tol=tol)


__all__ = [
    "GOAL_NODE",
    "AStarResult",
    "PenalizedMDP",
    "astar",
    "cost_per_cell",
    "heuristic_table",
    "penalty_planner",
]
