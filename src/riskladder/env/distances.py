"""Deterministic step distances on the map.

These use only the layout (no noise, no costs), so learners, the shield and reward shaping may use
them without seeing hidden parameters. See docs/env_spec.md sections 5 and 8.
"""

from __future__ import annotations

import numpy as np

from riskladder.env.layout import HAZARD, PASSABLE, Layout, grid_bfs


def steps_to_goal(layout: Layout, avoid_hazard_adjacent: bool = False) -> np.ndarray:
    """Fewest 1-cell moves from every cell to the nearest goal without entering a hazard.

    With `avoid_hazard_adjacent`, cells next to a hazard are also forbidden, which gives the length
    of the *safe* route: a 1-cell move along it can never slip into a hazard (a 2-cell dash still
    can). Goals are 0, unreachable cells inf.
    """
    walkable = layout.mask(PASSABLE)
    if avoid_hazard_adjacent:
        walkable &= ~hazard_adjacent(layout)
    return grid_bfs(walkable, layout.goals)


def max_reachable_distance(layout: Layout) -> int:
    """`D_max`: the largest hazard-free step distance to a goal over cells reachable from start."""
    reachable = grid_bfs(layout.mask(PASSABLE), [layout.start]) < np.inf
    dist = steps_to_goal(layout)
    return int(dist[reachable].max())


def hazard_adjacent(layout: Layout) -> np.ndarray:
    """Cells with at least one 4-neighbour that is a hazard."""
    padded = np.pad(layout.mask([HAZARD]), 1)
    return padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]
