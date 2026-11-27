"""Prior knowledge a learner can switch on: the safety shield and shaping (docs/learning.md).

Both are computed from the layout alone and cached per layout, so one learner can be trained over
several maps. They are independent switches, which is what the ablation in E1 needs.
"""

from __future__ import annotations

import numpy as np

from riskladder.env.dynamics import N_ACTIONS
from riskladder.env.layout import Layout
from riskladder.safety.shaping import Shaper
from riskladder.safety.shield import Shield


class Priors:
    def __init__(
        self,
        shield: Shield | dict | None = None,
        shaping: float | None = None,
        shield_at_run: bool = True,
    ) -> None:
        self.shield = Shield(**shield) if isinstance(shield, dict) else shield
        self.shaper = None if not shaping else Shaper(float(shaping))
        self.shield_at_run = shield_at_run
        self._cache: dict[int, tuple[Layout, np.ndarray | None, np.ndarray | None]] = {}

    @property
    def config(self) -> dict:
        return {
            "shield": None if self.shield is None else self.shield.config,
            "shaping": None if self.shaper is None else self.shaper.scale,
            "shield_at_run": self.shield_at_run,
        }

    def _entry(self, layout: Layout) -> tuple[Layout, np.ndarray | None, np.ndarray | None]:
        entry = self._cache.get(id(layout))
        if entry is None:
            allowed = None if self.shield is None else self.shield.table(layout)
            potential = None if self.shaper is None else self.shaper.potential(layout)
            entry = (layout, allowed, potential)  # the layout is kept so its id stays valid
            self._cache[id(layout)] = entry
        return entry

    def allowed(self, layout: Layout) -> np.ndarray | None:
        return self._entry(layout)[1]

    def run_allowed(self, layout: Layout) -> np.ndarray | None:
        """The mask applied to the greedy policy that is deployed and scored."""
        return self.allowed(layout) if self.shield_at_run else None

    def potential(self, layout: Layout) -> np.ndarray | None:
        return self._entry(layout)[2]

    def shape(
        self,
        layout: Layout,
        cell: tuple[int, int],
        next_cell: tuple[int, int],
        cost: float,
        terminated: bool,
    ) -> float:
        potential = self.potential(layout)
        if potential is None:
            return cost
        return Shaper.cost(potential, cell, next_cell, cost, terminated)


def masked(values: np.ndarray, allowed: np.ndarray | None) -> np.ndarray:
    """Action values with the blocked actions set to infinity (so arg-min never picks them)."""
    return values if allowed is None else np.where(allowed, values, np.inf)


def explore(rng: np.random.Generator, allowed: np.ndarray | None) -> int:
    """A uniformly random action among the allowed ones."""
    if allowed is None:
        return int(rng.integers(N_ACTIONS))
    return int(rng.choice(np.flatnonzero(allowed)))
