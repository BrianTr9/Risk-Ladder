"""Model-based learners: estimate a model from samples, then plan on it (docs/learning.md).

`MBCount` knows nothing about the map. It counts transitions per (cell, action) and plans on the
empirical model. Its only exploration mechanism is optimism (R-max, Brafman and Tennenholtz 2002):
an action tried fewer than `known_after` times is assumed to end the episode at no cost, so the
plan is drawn to try it. Replans are frequent (at most 1,000 steps apart) because each newly
known action can change what to try next. The plan that is scored and deployed is not optimistic:
an untried action there ends the episode at a prohibitive cost. Without the frequent replans
optimism left most actions untried, and exploring with epsilon-greedy under a pessimistic model
never left a bad route on some maps (see docs/learning.md).

`MBStruct` knows the geometry (where each of an action's three branches leads) and learns only what
is hidden: the slip probability of a move and of a dash, and the five cost parameters (move, dash,
rough, collision, failure). Every step informs every cell at once, so a few thousand steps pin the
model down and it needs no exploration at all (its epsilon defaults to 0). With `cmdp=True` it
plans with the constrained solver under an upper confidence bound on the slip probabilities, so
the failure probability of the plan is bounded by `delta` while the estimate is still uncertain
(the link to experiment E2).

Both replan on a geometric schedule (every `growth` times as many steps as the last plan), so the
planning cost stays small however long training runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from riskladder.agents.cmdp import solve_cmdp
from riskladder.env import Costs, MapSpec, Noise, build_model
from riskladder.env.dynamics import ACTIONS, FAIL, N_ACTIONS, PERPENDICULAR, RUNNING, walk_many
from riskladder.env.layout import Layout
from riskladder.oracle.solve import value_iteration


class Replanner:
    """Plans at step `first`, then again once the step count has grown by `growth` (but never more
    than `longest` steps apart, so a plan that turns out wrong is not followed for long)."""

    def __init__(
        self, first: int = 200, growth: float = 1.1, gap: int = 100, longest: int | None = None
    ) -> None:
        if first < 1 or growth <= 1.0 or gap < 1 or (longest is not None and longest < gap):
            raise ValueError("need first >= 1, growth > 1, gap >= 1 and longest >= gap")
        self.first, self.growth, self.gap, self.longest = first, growth, gap, longest
        self.next = first

    def due(self, t: int) -> bool:
        return t >= self.next

    def planned(self, t: int) -> None:
        self.next = max(t + self.gap, int(t * self.growth))
        if self.longest is not None:
            self.next = min(self.next, t + self.longest)


class _ModelBased:
    """Shared loop: collect statistics in `observe`, plan on a schedule, act on the latest plan."""

    name = "model_based"

    def __init__(
        self,
        eps_start: float = 0.0,
        eps_end: float = 0.0,
        eps_fraction: float = 0.5,
        first_plan: int = 200,
        growth: float = 1.1,
        longest: int | None = 5_000,
    ) -> None:
        if not 0.0 <= eps_end <= eps_start <= 1.0 or not 0.0 < eps_fraction <= 1.0:
            raise ValueError("need 0 <= eps_end <= eps_start <= 1 and eps_fraction in (0, 1]")
        self.eps_start, self.eps_end, self.eps_fraction = eps_start, eps_end, eps_fraction
        self._schedule = (first_plan, growth, 100, longest)
        Replanner(*self._schedule)  # validates the schedule
        self.config = {
            "eps_start": eps_start, "eps_end": eps_end, "eps_fraction": eps_fraction,
            "first_plan": first_plan, "growth": growth, "longest": longest,
        }  # fmt: skip

    def begin(self, layout: Layout, budget_steps: int, rng: np.random.Generator) -> None:
        self.layout, self.rng, self.t = layout, rng, 0
        self._ramp = max(1.0, self.eps_fraction * budget_steps)
        self.replanner = Replanner(*self._schedule)
        self._parts: list[tuple[float, np.ndarray]] | None = None  # (weight, action grid)
        self._chosen = 0
        self._current = False  # does the plan use all the data so far?
        self._deployed: list[tuple[float, np.ndarray]] | None = None
        self._deployed_current = False
        self._setup(layout)

    def epsilon(self) -> float:
        frac = min(1.0, self.t / self._ramp)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def start_episode(self, layout: Layout) -> None:
        if layout is not self.layout:
            raise ValueError("a model-based learner is tied to the map it started on")
        self._chosen = 0
        if self._parts is not None and len(self._parts) > 1:
            weights = np.array([w for w, _ in self._parts])
            self._chosen = int(self.rng.choice(len(weights), p=weights / weights.sum()))

    def _acting_plan(self) -> list[tuple[float, np.ndarray]]:
        if self._parts is None or not self._current:
            self._parts = self._plan(training=True)
            self._current = True
            self._chosen = min(self._chosen, len(self._parts) - 1)
        return self._parts

    def act(self, cell: tuple[int, int]) -> int:
        if self._parts is None or (self.replanner.due(self.t) and not self._current):
            self._acting_plan()
            self.replanner.planned(self.t)
        if self.rng.random() < self.epsilon():
            return int(self.rng.integers(N_ACTIONS))
        return int(self._parts[self._chosen][1][cell])

    def observe(self, cell, action, cost, next_cell, terminated, truncated) -> None:
        self.t += 1
        self._current = False
        self._deployed_current = False
        self._record(cell, action, cost, next_cell, terminated)

    def mixture(self, layout: Layout, cells: np.ndarray) -> list[tuple[float, np.ndarray]]:
        """The plan as it would be deployed: (weight, action per cell) parts."""
        if self._deployed is None or not self._deployed_current:
            self._deployed = self._plan(training=False)
            self._deployed_current = True
        return [(w, grid[cells[:, 0], cells[:, 1]]) for w, grid in self._deployed]

    def greedy(self, layout: Layout, cells: np.ndarray) -> np.ndarray:
        return max(self.mixture(layout, cells), key=lambda part: part[0])[1]

    def _setup(self, layout: Layout) -> None:
        raise NotImplementedError

    def _record(self, cell, action, cost, next_cell, terminated) -> None:
        raise NotImplementedError

    def _plan(self, training: bool) -> list[tuple[float, np.ndarray]]:
        """The plan used to act while training, or the one that would be deployed."""
        raise NotImplementedError


@dataclass(frozen=True)
class EmpiricalMDP:
    """A model estimated from counts, in the form the solvers need."""

    n_states: int
    n_actions: int
    start: int
    horizon: int
    P_stack: sparse.csr_matrix
    p_success: np.ndarray
    p_fail: np.ndarray
    cost: np.ndarray


class MBCount(_ModelBased):
    name = "mb_count"

    def __init__(
        self,
        known_after: int = 3,
        first_plan: int = 200,
        growth: float = 1.1,
        longest: int | None = 1_000,
    ) -> None:
        super().__init__(0.0, 0.0, 0.5, first_plan, growth, longest)
        if known_after < 1:
            raise ValueError("known_after must be at least 1")
        self.known_after = known_after
        self.config["known_after"] = known_after

    def _setup(self, layout: Layout) -> None:
        self.rows, self.cols = layout.rows, layout.cols
        n = self.rows * self.cols
        self.visits = np.zeros((n, N_ACTIONS), dtype=np.int64)
        self.ended = np.zeros((n, N_ACTIONS), dtype=np.int64)  # episodes that ended here
        self.cost_sum = np.zeros((n, N_ACTIONS))
        self.moved = np.zeros((n, N_ACTIONS, n), dtype=np.int32)  # (cell, action) -> next cell
        self._values: np.ndarray | None = None

    def _record(self, cell, action, cost, next_cell, terminated) -> None:
        s = cell[0] * self.cols + cell[1]
        self.visits[s, action] += 1
        self.cost_sum[s, action] += cost
        if terminated:
            self.ended[s, action] += 1
        else:
            self.moved[s, action, next_cell[0] * self.cols + next_cell[1]] += 1

    def model(self, optimistic: bool = True) -> EmpiricalMDP:
        """The empirical model. An action tried fewer than `known_after` times ends the episode at
        no cost when `optimistic` (so planning is drawn to try it) and at ten times the dearest
        average seen otherwise (so a deployed plan avoids what it has not learned)."""
        n = self.rows * self.cols
        known = self.visits >= self.known_after
        seen = np.maximum(self.visits, 1)
        blocks = []
        for a in range(N_ACTIONS):
            probs = self.moved[:, a, :] / seen[:, a : a + 1]
            probs[~known[:, a]] = 0.0
            blocks.append(sparse.csr_matrix(probs))
        end = np.where(known, self.ended / seen, 1.0)
        average = self.cost_sum / seen
        unknown_cost = 0.0 if optimistic else 10.0 * float(average[known].max(initial=1.0))
        cost = np.where(known, average, unknown_cost)
        return EmpiricalMDP(
            n, N_ACTIONS, 0, 0, sparse.vstack(blocks, format="csr"), end, np.zeros_like(end), cost
        )

    def _plan(self, training: bool) -> list[tuple[float, np.ndarray]]:
        solution = value_iteration(
            self.model(optimistic=training), tol=1e-7, init=self._values if training else None
        )
        if training:
            self._values = np.where(np.isfinite(solution.V), solution.V, 0.0)
        return [(1.0, solution.policy.reshape(self.rows, self.cols))]


@dataclass(frozen=True)
class _Branches:
    """Where each (cell, action, branch) leads, from the layout alone. Branch 0 is the intended
    direction, 1 and 2 the sideways slips (the same order as the exact model)."""

    row: np.ndarray
    col: np.ndarray
    ended: np.ndarray  # the branch ends the episode (goal or hazard)
    features: np.ndarray  # (..., 3): rough tiles crossed, bumped into a rock, entered a hazard


def _branches(layout: Layout) -> _Branches:
    rows, cols = layout.rows, layout.cols
    r, c = (v.ravel() for v in np.indices((rows, cols)))
    shape = (rows * cols, N_ACTIONS, 3)
    row, col = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)
    ended = np.zeros(shape, dtype=bool)
    features = np.zeros((*shape, 3))
    for a, action in enumerate(ACTIONS):
        sides = (action.direction, *PERPENDICULAR[action.direction])
        for b, direction in enumerate(sides):
            walk = walk_many(layout, r, c, direction, action.cells)
            row[:, a, b], col[:, a, b] = walk.rows, walk.cols
            ended[:, a, b] = walk.status != RUNNING
            features[:, a, b] = np.column_stack([walk.rough, walk.collided, walk.status == FAIL])
    unflatten = (rows, cols, N_ACTIONS, 3)
    return _Branches(
        row.reshape(unflatten), col.reshape(unflatten), ended.reshape(unflatten),
        features.reshape(*unflatten, 3),
    )  # fmt: skip


def _wilson_upper(slips: float, trials: float, z: float) -> float:
    if trials <= 0:
        return 1.0
    p = slips / trials
    centre = p + z * z / (2 * trials)
    spread = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return float(min(1.0, (centre + spread) / (1 + z * z / trials)))


class MBStruct(_ModelBased):
    """Knows the geometry; learns slip probabilities and costs from the samples."""

    name = "mb_struct"
    PRIOR_COSTS = np.array([1.0, 1.0, 0.0, 0.0, 0.0])  # move, dash, rough, collision, failure

    def __init__(
        self,
        cmdp: bool = False,
        delta: float = 0.01,
        p_max: float = 0.4,
        z: float = 1.64,
        ridge: float = 1e-2,
        eps_start: float = 0.0,
        eps_end: float = 0.0,
        eps_fraction: float = 0.5,
        first_plan: int = 200,
        growth: float = 1.1,
        longest: int | None = 5_000,
    ) -> None:
        super().__init__(eps_start, eps_end, eps_fraction, first_plan, growth, longest)
        if not 0.0 <= delta <= 1.0 or not 0.0 < p_max <= 1.0 or z < 0.0 or ridge <= 0.0:
            raise ValueError("need delta in [0, 1], p_max in (0, 1], z >= 0 and ridge > 0")
        self.cmdp, self.delta, self.p_max, self.z, self.ridge = cmdp, delta, p_max, z, ridge
        self.config.update(cmdp=cmdp, delta=delta, p_max=p_max, z=z, ridge=ridge)
        self.name = "mb_struct_cmdp" if cmdp else "mb_struct"

    def _setup(self, layout: Layout) -> None:
        self.branch = _branches(layout)
        self.trials = np.zeros(2)  # moves, dashes
        self.slips = np.zeros(2)
        self.gram = np.zeros((5, 5))  # normal equations for the cost parameters
        self.moment = np.zeros(5)
        self._values: np.ndarray | None = None

    def slip_estimate(self) -> np.ndarray:
        """Maximum-likelihood slip probability of a move and of a dash (0 before any sample)."""
        return np.divide(self.slips, self.trials, out=np.zeros(2), where=self.trials > 0)

    def slip_bound(self) -> np.ndarray:
        """Upper confidence bound on each slip probability, capped at the assumed `p_max`."""
        return np.array([min(self.p_max, _wilson_upper(k, n, self.z)) for k, n in
                         zip(self.slips, self.trials, strict=True)])  # fmt: skip

    def cost_estimate(self) -> np.ndarray:
        """Ridge regression of sampled costs on (move, dash, rough, collision, failure), pulled
        towards a prior of one cost unit per step and nothing for the rest."""
        system = self.gram + self.ridge * np.eye(5)
        theta = np.linalg.solve(system, self.moment + self.ridge * self.PRIOR_COSTS)
        return np.maximum(theta, np.array([1e-3, 1e-3, 0.0, 0.0, 0.0]))

    def _record(self, cell, action, cost, next_cell, terminated) -> None:
        r, c = cell
        b = self.branch
        match = (
            (b.row[r, c, action] == next_cell[0])
            & (b.col[r, c, action] == next_cell[1])
            & (b.ended[r, c, action] == terminated)
        )
        if not match.any():
            return
        kind = int(ACTIONS[action].cells == 2)
        p = self.slip_estimate()[kind]
        weight = match * np.array([1.0 - p, p / 2, p / 2])
        weight = weight / weight.sum() if weight.sum() > 0 else match / match.sum()
        self.trials[kind] += 1
        self.slips[kind] += weight[1] + weight[2]
        design = np.zeros(5)  # branches that cannot be told apart share the observation
        design[kind] = 1.0
        design[2:] = weight @ b.features[r, c, action]
        self.gram += np.outer(design, design)
        self.moment += design * cost

    def estimated_spec(self, upper: bool) -> MapSpec:
        slip = self.slip_bound() if upper else self.slip_estimate()
        move, dash, rough, collision, failure = self.cost_estimate()
        p_move = float(np.clip(slip[0], 1e-6, 1.0))
        p_dash = float(np.clip(slip[1], 0.0, 1.0))
        return MapSpec(
            self.layout,
            noise=Noise(p_slip=p_move, dash_slip_mult=p_dash / p_move),
            costs=Costs(move=move, dash=dash, rough_extra=rough, collision=collision),
            c_fail_override=failure,
        )

    def _plan(self, training: bool) -> list[tuple[float, np.ndarray]]:
        mdp = build_model(self.estimated_spec(upper=self.cmdp))
        if self.cmdp:
            solution = solve_cmdp(mdp, self.delta, tol=1e-8)
            return [(w, mdp.to_grid(p, fill=0)) for w, p in solution.mixture]
        solution = value_iteration(mdp, tol=1e-8, init=self._values)
        self._values = np.where(np.isfinite(solution.V), solution.V, 0.0)
        return [(1.0, mdp.to_grid(solution.policy, fill=0))]
