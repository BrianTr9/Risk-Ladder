"""Seeded map generators.

- `dilemma`: a short corridor beside a hazard strip versus a long, hazard-free detour. The choice
  between them is the cost-versus-risk trade-off that the headline experiments study.
- `field`: open terrain with clustered hazards, rocks and rough ground, for generalisation tests.

Generation is deterministic in `seed`. An invalid draw is retried with the next internal draw.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from riskladder.env.distances import steps_to_goal
from riskladder.env.layout import GOAL, GROUND, HAZARD, ROCK, ROUGH, START, Layout, MapError
from riskladder.env.spec import Costs, MapSpec, Noise

Grid = list[list[str]]
MAX_ATTEMPTS = 200


def generate(
    template: str,
    rows: int = 30,
    cols: int | None = None,
    seed: int = 0,
    noise: Noise | None = None,
    costs: Costs | None = None,
    **params: float,
) -> MapSpec:
    """Generate a valid map. Extra `params` are passed to the template (e.g. `hazard_density`)."""
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}; choose from {sorted(TEMPLATES)}")
    cols = rows if cols is None else cols
    rng = np.random.default_rng(seed)
    for _ in range(MAX_ATTEMPTS):
        grid = TEMPLATES[template](rows, cols, rng, **params)
        try:
            layout = Layout(tuple("".join(line) for line in grid))
        except MapError:
            continue
        if template == "dilemma" and not has_safe_route(layout):
            continue  # scattered rocks closed the detour: the map would not be a dilemma
        return MapSpec(
            layout=layout,
            noise=noise or Noise(),
            costs=costs or Costs(),
            name=f"{template}-{rows}x{cols}-s{seed}",
            seed=seed,
        )
    raise MapError(f"could not generate a valid {template} map in {MAX_ATTEMPTS} attempts")


def _dilemma(rows: int, cols: int, rng: np.random.Generator, detour: float = 0.8) -> Grid:
    """Corridor on row `rc` from S (left) to G (right); a hazard strip runs along its south side.

    The detour climbs a lane on each side through a rock wall of depth `d` and crosses an open
    region at the top, so it is about `2 * d` cells longer than the corridor.
    `detour` sets `2 * d` as a fraction of the corridor length.
    """
    if rows < 10 or cols < 10:
        raise ValueError("dilemma maps need at least 10 x 10 cells")
    g = [[ROCK] * cols for _ in range(rows)]
    corridor = cols - 3  # steps from S to G along the corridor
    rc = rows - 3 - int(rng.integers(0, rows // 10 + 1))  # corridor near the bottom edge
    depth = int(np.clip(round(detour * corridor / 2 * rng.uniform(0.8, 1.2)), 2, rc - 4))
    top = rc - depth  # rows 1 .. top-1 are the open region

    for r in range(1, top):
        for c in range(1, cols - 1):
            g[r][c] = GROUND
    for r in range(1, rc + 1):  # lanes
        g[r][1] = GROUND
        g[r][cols - 2] = GROUND
    for c in range(1, cols - 1):
        g[rc][c] = GROUND

    strip = int(rng.integers(max(3, corridor // 4), max(4, corridor // 2) + 1))
    first = int(rng.integers(2, cols - 2 - strip))
    for c in range(first, first + strip):
        g[rc + 1][c] = HAZARD

    _scatter(g, rng, 1, top, ROUGH, fraction=0.15)
    _scatter(g, rng, 1, top, ROCK, fraction=0.06)
    g[rc][1] = START
    g[rc][cols - 2] = GOAL
    return g


def _field(
    rows: int,
    cols: int,
    rng: np.random.Generator,
    hazard_density: float = 0.10,
    rock_density: float = 0.08,
    rough_density: float = 0.10,
) -> Grid:
    """Open terrain with clusters; S in the left quarter, G in the right quarter."""
    g = [[ROCK] * cols for _ in range(rows)]
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            g[r][c] = GROUND
    _scatter(g, rng, 1, rows - 1, HAZARD, fraction=hazard_density)
    _scatter(g, rng, 1, rows - 1, ROCK, fraction=rock_density)
    _scatter(g, rng, 1, rows - 1, ROUGH, fraction=rough_density)
    quarter = max(1, (cols - 2) // 4)
    sr, sc = int(rng.integers(1, rows - 1)), int(rng.integers(1, 1 + quarter))
    gr, gc = int(rng.integers(1, rows - 1)), int(rng.integers(cols - 1 - quarter, cols - 1))
    g[sr][sc] = START
    g[gr][gc] = GOAL
    return g


def _scatter(
    g: Grid, rng: np.random.Generator, r0: int, r1: int, tile: str, fraction: float
) -> None:
    """Grow random clusters of `tile` on ground cells in rows [r0, r1) to cover `fraction`."""
    cols = len(g[0])
    n_ground = sum(row[1 : cols - 1].count(GROUND) for row in g[r0:r1])
    target = int(fraction * n_ground)
    placed, attempts = 0, 0
    while placed < target and attempts < 50 * target + 100:
        attempts += 1
        r, c = int(rng.integers(r0, r1)), int(rng.integers(1, cols - 1))
        if g[r][c] != GROUND:
            continue
        for _ in range(int(rng.integers(2, 7))):  # short random walk = one cluster
            if g[r][c] == GROUND and placed < target:
                g[r][c] = tile
                placed += 1
            dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[int(rng.integers(4))]
            if r0 <= r + dr < r1 and 1 <= c + dc < cols - 1:
                r, c = r + dr, c + dc


def has_safe_route(layout: Layout) -> bool:
    """True if S can reach a goal using only cells that are not next to a hazard."""
    r, c = layout.start
    return bool(np.isfinite(steps_to_goal(layout, avoid_hazard_adjacent=True)[r, c]))


TEMPLATES: dict[str, Callable[..., Grid]] = {"dilemma": _dilemma, "field": _field}
