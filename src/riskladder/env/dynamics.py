"""Dynamics: the single source of truth for what an action does (docs/env_spec.md section 4).

`walk_many` is the only implementation of motion (geometry) and `move_many` prices it. Both are
vectorised over cells, so the exact model (every cell at once) and `outcomes` (one cell) run the
same code.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from riskladder.env.layout import C_GOAL, C_HAZARD, C_ROCK, C_ROUGH, PAD, Layout
from riskladder.env.spec import MapSpec

# Episode status after a transition.
RUNNING, SUCCESS, FAIL = 0, 1, 2

DIRECTIONS = ("N", "E", "S", "W")
DELTA = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
# Sideways slip directions, clockwise first.
PERPENDICULAR = {"N": ("E", "W"), "E": ("S", "N"), "S": ("W", "E"), "W": ("N", "S")}


class Action(NamedTuple):
    name: str
    direction: str
    cells: int  # 1 = move, 2 = dash


ACTIONS: tuple[Action, ...] = tuple(
    Action(f"{kind}_{d}", d, cells)
    for kind, cells in (("move", 1), ("dash", 2))
    for d in DIRECTIONS
)
N_ACTIONS = len(ACTIONS)
N_BRANCHES = 3  # intended direction, clockwise slip, counter-clockwise slip


class Outcome(NamedTuple):
    prob: float
    cell: tuple[int, int]  # cell the rover ends in (for SUCCESS/FAIL: the goal/hazard it entered)
    cost: float
    status: int


def branches(spec: MapSpec, action: int) -> tuple[tuple[str, float], ...]:
    """The three (direction, probability) branches of an action, in a fixed order."""
    act = ACTIONS[action]
    p = spec.noise.slip(act.cells)
    cw, ccw = PERPENDICULAR[act.direction]
    return (act.direction, 1.0 - p), (cw, p / 2), (ccw, p / 2)


class Walk(NamedTuple):
    """Where a rover ends and what it touched; geometry only, so no costs are involved."""

    rows: np.ndarray
    cols: np.ndarray
    status: np.ndarray  # RUNNING, SUCCESS or FAIL
    rough: np.ndarray  # rough tiles entered
    collided: np.ndarray  # bool: bumped into a rock


def walk_many(
    layout: Layout, rows: np.ndarray, cols: np.ndarray, direction: str, cells: int
) -> Walk:
    """Advance rovers at (rows, cols) up to `cells` cells in `direction` over the bare layout.

    A rock stops the rover where it stands; a hazard or goal ends the episode on entry.
    """
    codes = layout.codes
    dr, dc = DELTA[direction]
    r = np.asarray(rows, dtype=np.int64) + PAD
    c = np.asarray(cols, dtype=np.int64) + PAD
    rough = np.zeros(r.shape, dtype=np.int64)
    collided = np.zeros(r.shape, dtype=bool)
    status = np.full(r.shape, RUNNING, dtype=np.int8)
    active = np.ones(r.shape, dtype=bool)
    for _ in range(cells):
        nr, nc = r + dr, c + dc
        tile = codes[nr, nc]
        blocked = active & (tile == C_ROCK)
        collided |= blocked
        active &= ~blocked
        r = np.where(active, nr, r)
        c = np.where(active, nc, c)
        hazard = active & (tile == C_HAZARD)
        goal = active & (tile == C_GOAL)
        rough += active & (tile == C_ROUGH)
        status[hazard] = FAIL
        status[goal] = SUCCESS
        active &= ~(hazard | goal)
    return Walk(r - PAD, c - PAD, status, rough, collided)


class Branches(NamedTuple):
    """Where every (cell, action, branch) leads over the bare layout, as (rows, cols, A, 3) arrays.
    Branch 0 is the intended direction, 1 and 2 the sideways slips (the order of the exact model).
    Rows and columns are where the rover ends (the goal or hazard tile it entered, for those)."""

    rows: np.ndarray
    cols: np.ndarray
    status: np.ndarray  # RUNNING, SUCCESS or FAIL
    rough: np.ndarray  # rough tiles entered
    collided: np.ndarray  # bumped into a rock


def branch_table(layout: Layout) -> Branches:
    """The outcome geometry of every action from every cell: the one place the learners' features,
    the shield and the structured model learner read it from."""
    r, c = (v.ravel() for v in np.indices((layout.rows, layout.cols)))
    walks = [[], [], [], [], []]
    for action in ACTIONS:
        parts = [
            walk_many(layout, r, c, direction, action.cells)
            for direction in (action.direction, *PERPENDICULAR[action.direction])
        ]
        for field_, values in zip(walks, zip(*parts, strict=True), strict=True):
            field_.append(np.stack(values, axis=-1))  # (cells, 3)
    shape = (layout.rows, layout.cols, N_ACTIONS, 3)
    return Branches(*(np.stack(field_, axis=1).reshape(shape) for field_ in walks))


def move_many(
    spec: MapSpec, rows: np.ndarray, cols: np.ndarray, direction: str, cells: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance rovers at (rows, cols) up to `cells` cells in `direction`.

    Returns (rows, cols, extra_cost, status). `extra_cost` excludes the base action cost.
    """
    walk = walk_many(spec.layout, rows, cols, direction, cells)
    costs = spec.costs
    extra = (
        walk.collided * costs.collision
        + walk.rough * costs.rough_extra
        + (walk.status == FAIL) * spec.c_fail
    )
    return walk.rows, walk.cols, extra, walk.status


def base_cost(spec: MapSpec, action: int) -> float:
    return spec.costs.move if ACTIONS[action].cells == 1 else spec.costs.dash


def outcomes(spec: MapSpec, cell: tuple[int, int], action: int) -> tuple[Outcome, ...]:
    """Every distinct outcome of `action` from `cell`, with probability. Zero-probability branches
    are dropped and identical outcomes are merged."""
    act = ACTIONS[action]
    merged: dict[tuple[tuple[int, int], float, int], float] = {}
    for direction, prob in branches(spec, action):
        if prob == 0.0:
            continue
        r, c, extra, status = move_many(
            spec, np.array([cell[0]]), np.array([cell[1]]), direction, act.cells
        )
        key = ((int(r[0]), int(c[0])), base_cost(spec, action) + float(extra[0]), int(status[0]))
        merged[key] = merged.get(key, 0.0) + prob
    return tuple(Outcome(p, cell_, cost, status) for (cell_, cost, status), p in merged.items())
