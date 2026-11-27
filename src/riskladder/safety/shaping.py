"""Potential-based reward shaping from the deterministic distance to the goal (Ng et al., 1999).

The potential is the hazard-free step distance to the goal times a scale, in cost units, and is zero
in the absorbing outcomes. A transition from `s` to `s'` then costs `c + Phi(s') - Phi(s)`, which
telescopes along an episode to `sum(c) - Phi(s0)`: a constant that does not depend on the policy.
Every policy therefore has the same ranking under the shaped and the original cost, and the oracle
keeps scoring on the original cost.

The distance comes from the layout alone, so no hidden quantity is used (docs/env_spec.md section
8). The scale is arbitrary for correctness; it only decides how much of the cost-to-go the potential
already explains, so it is a tuned setting rather than a read-off of the (hidden) step cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskladder.env.distances import steps_to_goal
from riskladder.env.layout import Layout


@dataclass(frozen=True)
class Shaper:
    scale: float = 1.0  # cost units per step of distance

    def __post_init__(self) -> None:
        if not self.scale >= 0.0 or not np.isfinite(self.scale):
            raise ValueError("scale must be a finite non-negative number")

    def potential(self, layout: Layout) -> np.ndarray:
        """`Phi[row, col]` in cost units; cells cut off from the goal get the largest finite value
        plus one step. Absorbing outcomes are handled in `cost`, not stored here."""
        dist = steps_to_goal(layout)
        finite = dist[np.isfinite(dist)]
        cap = (finite.max() if finite.size else 0.0) + 1.0
        return self.scale * np.where(np.isfinite(dist), dist, cap)

    @staticmethod
    def cost(
        potential: np.ndarray,
        cell: tuple[int, int],
        next_cell: tuple[int, int],
        cost: float,
        terminated: bool,
    ) -> float:
        """Shaped cost of one transition; the potential of a terminated episode is zero."""
        after = 0.0 if terminated else float(potential[next_cell])
        return cost + after - float(potential[cell])
