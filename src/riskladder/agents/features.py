"""Map-only features of a (cell, action) pair for the linear learner (docs/env_spec.md section 8).

Everything here comes from the layout: where each of an action's three branches (the intended
direction and the two sideways slips) would lead, how far the landing cell is from the goal in
hazard-free steps, and what the shortest route from there still has in store (how many of its
cells lie next to a hazard, how many are rough). The slip *probabilities* and every cost stay
hidden; the learner has to find out from samples how much each feature matters.
"""

from __future__ import annotations

import numpy as np

from riskladder.env.distances import hazard_adjacent, steps_to_goal
from riskladder.env.dynamics import ACTIONS, FAIL, N_ACTIONS, RUNNING, branch_table
from riskladder.env.layout import NEIGHBOURS, PASSABLE, ROUGH, Layout

FEATURES = (
    "bias",
    "goal_distance_after",  # hazard-free steps from the intended landing cell to the goal
    "dash",
    "intended_hazard",  # the intended path enters a hazard
    "slip_hazard",  # share of the two slips that enter a hazard
    "slip_goal_distance",  # mean goal distance after the two slips
    "route_exposure_after",  # hazard-adjacent cells left on the shortest route from the landing
    "route_rough_after",  # rough tiles left on that route
    "collision",  # the intended path bumps into a rock
    "rough_tiles",  # rough tiles on the intended path
)
N_FEATURES = len(FEATURES)


def _route_totals(layout: Layout, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hazard-adjacent and rough cells still ahead on the shortest route from every cell.

    The route steps to the neighbour one step closer to the goal, preferring the one with less
    exposure left; cells that cannot reach the goal without a hazard get the largest finite value.
    """
    exposed = hazard_adjacent(layout).astype(float)
    rough = layout.mask([ROUGH]).astype(float)
    exposure, rough_left = exposed.copy(), rough.copy()
    reachable = np.isfinite(dist)
    for r, c in sorted(zip(*np.nonzero(reachable), strict=True), key=lambda rc: dist[rc]):
        if dist[r, c] == 0:
            continue
        ahead = [
            (r + dr, c + dc)
            for dr, dc in NEIGHBOURS
            if 0 <= r + dr < layout.rows
            and 0 <= c + dc < layout.cols
            and dist[r + dr, c + dc] == dist[r, c] - 1
        ]
        best = min(ahead, key=lambda cell: exposure[cell])  # ties go to the first neighbour
        exposure[r, c] += exposure[best]
        rough_left[r, c] += rough_left[best]
    for arr in (exposure, rough_left):
        arr[~reachable] = arr[reachable].max() if reachable.any() else 0.0
    return exposure, rough_left


def feature_table(layout: Layout) -> np.ndarray:
    """`phi[row, col, action, feature]`, scaled to roughly [0, 1.5]; zeros where the rover never
    decides."""
    dist = steps_to_goal(layout)
    finite = dist[np.isfinite(dist)]
    scale = max(1.0, float(finite.max()))
    h = np.where(np.isfinite(dist), dist, 1.5 * scale) / scale
    exposure, rough_left = _route_totals(layout, dist)
    exposure, rough_left = exposure / scale, rough_left / scale

    b = branch_table(layout)
    running = b.status == RUNNING
    h_after = np.where(running, h[b.rows, b.cols], 0.0)  # (rows, cols, A, 3)
    intended = (..., 0)
    slips = (..., slice(1, None))
    phi = np.zeros((layout.rows, layout.cols, N_ACTIONS, N_FEATURES))
    phi[..., 0] = 1.0
    phi[..., 1] = h_after[intended]
    phi[..., 2] = [float(a.cells == 2) for a in ACTIONS]
    phi[..., 3] = b.status[intended] == FAIL
    phi[..., 4] = (b.status[slips] == FAIL).mean(axis=-1)
    phi[..., 5] = h_after[slips].mean(axis=-1)
    phi[..., 6] = np.where(running[intended], exposure[b.rows[intended], b.cols[intended]], 0.0)
    phi[..., 7] = np.where(running[intended], rough_left[b.rows[intended], b.cols[intended]], 0.0)
    phi[..., 8] = b.collided[intended]
    phi[..., 9] = b.rough[intended]
    phi[~layout.mask(PASSABLE)] = 0.0
    return phi
