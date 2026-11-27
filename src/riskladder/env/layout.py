"""Map layout: tiles, validation and the geometry that every agent is allowed to see."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property

import numpy as np

GROUND, START, ROUGH, ROCK, HAZARD, GOAL = ".", "S", "~", "#", "X", "G"
TILES = frozenset({GROUND, START, ROUGH, ROCK, HAZARD, GOAL})
PASSABLE = frozenset({GROUND, START, ROUGH})  # cells where the rover can stand and decide

# Integer tile codes used by the vectorised dynamics.
C_GROUND, C_ROUGH, C_ROCK, C_HAZARD, C_GOAL = 0, 1, 2, 3, 4
_CODE = {
    GROUND: C_GROUND,
    START: C_GROUND,
    ROUGH: C_ROUGH,
    ROCK: C_ROCK,
    HAZARD: C_HAZARD,
    GOAL: C_GOAL,
}

NEIGHBOURS = ((-1, 0), (0, 1), (1, 0), (0, -1))
PAD = 2  # rock border added around the grid so that a 2-cell move never leaves the array


class MapError(ValueError):
    """Raised when a map violates the rules in docs/env_spec.md section 2."""


def grid_bfs(walkable: np.ndarray, sources: Iterable[tuple[int, int]]) -> np.ndarray:
    """4-connected BFS step distances from `sources` through `walkable` cells.

    Sources are always distance 0, even if they are not walkable themselves (e.g. goal tiles).
    Returns a float array with `inf` for unreached cells.
    """
    rows, cols = walkable.shape
    dist = np.full((rows, cols), np.inf)
    queue: deque[tuple[int, int]] = deque()
    for r, c in sources:
        if dist[r, c] != 0:
            dist[r, c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        d = dist[r, c] + 1
        for dr, dc in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and walkable[nr, nc] and dist[nr, nc] == np.inf:
                dist[nr, nc] = d
                queue.append((nr, nc))
    return dist


@dataclass(frozen=True)
class Layout:
    """An immutable, validated grid of tiles. Contains no dynamics, costs or noise."""

    grid: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate(self.grid)

    @classmethod
    def from_text(cls, text: str) -> Layout:
        return cls(tuple(line for line in text.strip("\n").split("\n")))

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def cols(self) -> int:
        return len(self.grid[0])

    def tile(self, r: int, c: int) -> str:
        """Tile at (r, c); cells outside the grid behave like rock."""
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.grid[r][c]
        return ROCK

    @cached_property
    def start(self) -> tuple[int, int]:
        return next(self.positions(START))

    @cached_property
    def goals(self) -> tuple[tuple[int, int], ...]:
        return tuple(self.positions(GOAL))

    @cached_property
    def passable_cells(self) -> tuple[tuple[int, int], ...]:
        """Decision cells in row-major order; this order defines the state index."""
        return tuple(
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r][c] in PASSABLE
        )

    @cached_property
    def codes(self) -> np.ndarray:
        """Tile codes padded with `PAD` cells of rock on every side (read-only)."""
        codes = np.full((self.rows + 2 * PAD, self.cols + 2 * PAD), C_ROCK, dtype=np.int8)
        for r, line in enumerate(self.grid):
            codes[PAD + r, PAD : PAD + self.cols] = [_CODE[ch] for ch in line]
        codes.flags.writeable = False
        return codes

    def mask(self, tiles: Iterable[str]) -> np.ndarray:
        wanted = set(tiles)
        return np.array([[ch in wanted for ch in line] for line in self.grid])

    def positions(self, tile: str) -> Iterable[tuple[int, int]]:
        for r, line in enumerate(self.grid):
            for c, ch in enumerate(line):
                if ch == tile:
                    yield r, c

    def __str__(self) -> str:
        return "\n".join(self.grid)


def _validate(grid: tuple[str, ...]) -> None:
    if not grid or not grid[0]:
        raise MapError("map is empty")
    if any(len(line) != len(grid[0]) for line in grid):
        raise MapError("map is not rectangular")
    bad = {ch for line in grid for ch in line} - TILES
    if bad:
        raise MapError(f"unknown tile symbols: {sorted(bad)}")
    n_start = sum(line.count(START) for line in grid)
    if n_start != 1:
        raise MapError(f"expected exactly one start 'S', found {n_start}")
    if not any(GOAL in line for line in grid):
        raise MapError("expected at least one goal 'G'")

    walkable = np.array([[ch in PASSABLE for ch in line] for line in grid])
    start = next((r, line.index(START)) for r, line in enumerate(grid) if START in line)
    reached = grid_bfs(walkable, [start]) < np.inf
    for r, line in enumerate(grid):
        for c, ch in enumerate(line):
            if ch == GOAL and any(
                0 <= r + dr < len(grid) and 0 <= c + dc < len(line) and reached[r + dr, c + dc]
                for dr, dc in NEIGHBOURS
            ):
                return
    raise MapError("no hazard-free route from 'S' to any 'G'")
