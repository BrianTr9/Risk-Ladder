"""Planning as a service: the logic behind the API, free of any web framework.

`plan` answers "given this map, what should the rover do and how risky is it?" and `evaluate` scores
a policy someone else wrote. Both accept the `.map` text format, enforce size limits (so one
request cannot tie up the server) and report every problem as a `ServiceError` carrying an
HTTP-style status, which the API layer turns into a response.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from riskladder.agents import penalty_planner, solve_cmdp
from riskladder.env import MapError, MapSpec, Noise, build_model, parse_map
from riskladder.env.dynamics import ACTIONS, N_ACTIONS
from riskladder.env.model import SparseMDP
from riskladder.oracle import PolicyResult, evaluate_mixture

MAX_STATES = 2_500  # decision cells a request may have; set from the latency measurements
MAX_MAP_CHARS = 40_000  # a 100 x 100 grid plus its header
NO_ACTION = (
    -1
)  # entry of a policy grid at a cell where the rover never decides (rock, goal, hazard)


class ServiceError(Exception):
    """A request the service cannot or will not handle. `status` follows HTTP conventions."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class Component:
    """One deterministic policy of a mixture, as a grid of action indices."""

    weight: float
    actions: list[list[int]]  # rows x cols, NO_ACTION where there is no decision


@dataclass(frozen=True)
class PlanResult:
    solver: str  # "environment" (no risk bound), "penalty" or "cmdp"
    components: list[Component]
    outcome: PolicyResult
    multiplier: float | None  # Lagrange multiplier (cmdp) or the extra penalty (penalty)
    optimality_gap: float | None  # distance to the dual bound (cmdp); 0 means provably optimal
    feasible: bool
    states: int
    seconds: float


def load_model(text: str, p_slip: float | None = None) -> tuple[MapSpec, SparseMDP]:
    """Parse a map, apply the limits and build its exact model."""
    if not isinstance(text, str) or len(text) > MAX_MAP_CHARS:
        raise ServiceError(413, f"map text is longer than {MAX_MAP_CHARS:,} characters")
    try:
        spec = parse_map(text)
        if p_slip is not None:
            spec = spec.with_noise(Noise(p_slip=p_slip, dash_slip_mult=spec.noise.dash_slip_mult))
    except MapError as error:
        raise ServiceError(422, str(error)) from None
    cells = len(spec.layout.passable_cells)
    if cells > MAX_STATES:
        raise ServiceError(413, f"map has {cells:,} decision cells; the limit is {MAX_STATES:,}")
    return spec, build_model(spec)


def to_grid(mdp: SparseMDP, policy: np.ndarray) -> list[list[int]]:
    grid = mdp.to_grid(np.asarray(policy, dtype=int), fill=NO_ACTION)
    return grid.astype(int).tolist()


def from_grid(mdp: SparseMDP, grid: list[list[int]]) -> np.ndarray:
    """The policy array for a grid of action indices; rejects grids that do not fit the map."""
    rows, cols = mdp.index.shape
    if len(grid) != rows or any(len(row) != cols for row in grid):
        raise ServiceError(422, f"policy grid must be {rows} x {cols}")
    policy = np.empty(mdp.n_states, dtype=int)
    for s, (r, c) in enumerate(mdp.cells):
        action = grid[r][c]
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < N_ACTIONS:
            raise ServiceError(422, f"cell ({r}, {c}) needs an action index in [0, {N_ACTIONS})")
        policy[s] = action
    return policy


def plan(
    text: str,
    delta: float | None = None,
    extra_penalty: float | None = None,
    p_slip: float | None = None,
) -> PlanResult:
    """Plan for a map. With `delta` the failure probability is bounded by it (constrained solver);
    with `extra_penalty` a failure is priced that much higher (penalty planner); with neither the
    environment's own failure cost decides."""
    if delta is not None and extra_penalty is not None:
        raise ServiceError(422, "give either delta or extra_penalty, not both")
    started = time.perf_counter()
    mdp = load_model(text, p_slip)[1]
    try:
        if delta is not None:
            solution = solve_cmdp(mdp, delta)
            mixture, multiplier, gap, feasible = (
                solution.mixture, solution.multiplier, solution.gap, solution.feasible
            )  # fmt: skip
            solver = "cmdp"
        else:
            price = 0.0 if extra_penalty is None else extra_penalty
            mixture = ((1.0, penalty_planner(mdp, price).policy),)
            multiplier, gap, feasible = price, None, True
            solver = "environment" if extra_penalty is None else "penalty"
    except ValueError as error:
        raise ServiceError(422, str(error)) from None
    outcome = evaluate_mixture(mdp, mixture)
    components = [Component(float(w), to_grid(mdp, pi)) for w, pi in mixture]
    gap = None if gap is None or not np.isfinite(gap) else float(gap)
    return PlanResult(
        solver, components, outcome, multiplier, gap, feasible, mdp.n_states,
        time.perf_counter() - started,
    )  # fmt: skip


def evaluate(
    text: str, components: list[tuple[float, list[list[int]]]], p_slip: float | None = None
) -> tuple[PolicyResult, float]:
    """Exact outcome of a mixture of policies given as action grids, and the time it took."""
    started = time.perf_counter()
    _, mdp = load_model(text, p_slip)
    if not components:
        raise ServiceError(422, "give at least one policy")
    try:
        mixture = tuple((float(w), from_grid(mdp, grid)) for w, grid in components)
        outcome = evaluate_mixture(mdp, mixture)
    except ValueError as error:
        raise ServiceError(422, str(error)) from None
    return outcome, time.perf_counter() - started


def action_names() -> list[str]:
    return [a.name for a in ACTIONS]


__all__ = [
    "MAX_MAP_CHARS",
    "MAX_STATES",
    "NO_ACTION",
    "Component",
    "PlanResult",
    "ServiceError",
    "action_names",
    "evaluate",
    "load_model",
    "plan",
]
