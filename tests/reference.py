"""Independent reference implementation, written directly from docs/env_spec.md.

It deliberately shares no code with `riskladder`: plain Python loops and dictionaries, its own BFS,
its own value iteration and its own forward evaluation. It is slow and only used by tests to check
that the vectorised implementation does what the specification says (not merely that it agrees with
itself).
"""

from __future__ import annotations

from collections import deque
from functools import cache

DIRS = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
SIDEWAYS = {"N": ("E", "W"), "E": ("N", "S"), "S": ("E", "W"), "W": ("N", "S")}
# Spec section 3: indices 0-3 are 1-cell moves N, E, S, W; 4-7 are 2-cell dashes in the same order.
ACTION_LIST = [("move", d) for d in "NESW"] + [("dash", d) for d in "NESW"]
PASSABLE = ".S~"


def cell_of(grid, symbol):
    return [(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == symbol]


def tile(grid, r, c):
    if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
        return grid[r][c]
    return "#"  # spec 2.1: outside the grid behaves like rock


def bfs(grid, sources):
    """Step distances from `sources` moving through passable cells (sources are distance 0)."""
    dist = {s: 0 for s in sources}
    queue = deque(sources)
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS.values():
            nxt = (r + dr, c + dc)
            if nxt not in dist and tile(grid, *nxt) in PASSABLE:
                dist[nxt] = dist[(r, c)] + 1
                queue.append(nxt)
    return dist


@cache
def c_fail(spec):
    """Spec section 5: fail_multiplier * D_max * move cost, unless overridden."""
    if spec.c_fail_override is not None:
        return spec.c_fail_override
    grid = spec.layout.grid
    reachable = bfs(grid, cell_of(grid, "S"))
    to_goal = bfs(grid, cell_of(grid, "G"))
    d_max = max(to_goal[cell] for cell in reachable)
    return spec.fail_multiplier * d_max * spec.costs.move


def outcomes(spec, cell, action):
    """Spec section 4, step by step. Returns {(end_cell, cost, status): probability}."""
    grid = spec.layout.grid
    kind, direction = ACTION_LIST[action]
    length = 1 if kind == "move" else 2
    base = spec.costs.move if kind == "move" else spec.costs.dash
    slip = spec.noise.p_slip * (1.0 if kind == "move" else spec.noise.dash_slip_mult)
    branches = [(direction, 1.0 - slip)] + [(side, slip / 2.0) for side in SIDEWAYS[direction]]

    result = {}
    for effective, prob in branches:
        if prob == 0.0:
            continue
        r, c = cell
        cost, status = base, "running"
        for _ in range(length):
            dr, dc = DIRS[effective]
            kind_of_next = tile(grid, r + dr, c + dc)
            if kind_of_next == "#":
                cost += spec.costs.collision
                break
            r, c = r + dr, c + dc
            if kind_of_next == "X":
                cost += c_fail(spec)
                status = "fail"
                break
            if kind_of_next == "G":
                status = "success"
                break
            if kind_of_next == "~":
                cost += spec.costs.rough_extra
        key = ((r, c), round(cost, 9), status)
        result[key] = result.get(key, 0.0) + prob
    return result


def decision_cells(spec):
    grid = spec.layout.grid
    return [(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch in PASSABLE]


def optimal_values(spec, tol=1e-12, max_sweeps=200_000):
    """Gauss-Seidel value iteration on the SSP (no discounting), from V = 0."""
    cells = decision_cells(spec)
    table = {(cell, a): outcomes(spec, cell, a) for cell in cells for a in range(8)}
    value = dict.fromkeys(cells, 0.0)
    for _ in range(max_sweeps):
        change = 0.0
        for cell in cells:
            best = min(
                sum(
                    p * (cost + (value[end] if status == "running" else 0.0))
                    for (end, cost, status), p in table[(cell, a)].items()
                )
                for a in range(8)
            )
            change = max(change, abs(best - value[cell]))
            value[cell] = best
        if change < tol:
            return value
    raise RuntimeError("reference value iteration did not converge")


def evaluate(spec, policy, start, horizon):
    """Forward propagation of the state distribution for `horizon` steps.

    `policy` maps a decision cell to an action index. Returns (expected cost, P(success), P(fail),
    P(timeout)) from `start`.
    """
    table = {}
    dist = {start: 1.0}
    cost = success = fail = 0.0
    for _ in range(horizon):
        nxt = {}
        for cell, mass in dist.items():
            key = (cell, policy[cell])
            if key not in table:
                table[key] = outcomes(spec, *key)
            for (end, step_cost, status), p in table[key].items():
                cost += mass * p * step_cost
                if status == "success":
                    success += mass * p
                elif status == "fail":
                    fail += mass * p
                else:
                    nxt[end] = nxt.get(end, 0.0) + mass * p
        dist = nxt
    timeout = sum(dist.values())
    return cost, success, fail, timeout
