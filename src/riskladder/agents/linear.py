"""Linear Q-learning on map-only features: one weight vector shared by every cell and every map.

`Q(s, a) = w . phi(s, a)` where `phi` comes from the layout alone (see `features.py`). Because the
weights do not depend on which cell or which map they were learned on, they carry over to maps the
learner has never seen; that is the point of the generalisation experiment.

The update is semi-gradient Q-learning with the step normalised by the feature norm, so a step
size in (0, 1] moves the estimate that fraction of the way to the target whatever the feature
scale. Costs are positive and the weights start at zero, which is the same optimism as the table.
"""

from __future__ import annotations

import numpy as np

from riskladder.agents.features import N_FEATURES, feature_table
from riskladder.agents.priors import Priors, explore, masked
from riskladder.env.layout import Layout
from riskladder.safety.shield import Shield


class LinearQ:
    name = "linear"

    def __init__(
        self,
        alpha: float = 0.1,
        alpha_decay: float = 0.0,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_fraction: float = 0.5,
        shield: Shield | dict | None = None,
        shaping: float | None = None,
        shield_at_run: bool = True,
        shaping_route: str = "free",
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if alpha_decay < 0.0:
            raise ValueError("alpha_decay must be non-negative")
        if not 0.0 <= eps_end <= eps_start <= 1.0:
            raise ValueError("need 0 <= eps_end <= eps_start <= 1")
        if not 0.0 < eps_fraction <= 1.0:
            raise ValueError("eps_fraction must be in (0, 1]")
        self.alpha, self.alpha_decay = alpha, alpha_decay
        self.eps_start, self.eps_end, self.eps_fraction = eps_start, eps_end, eps_fraction
        self.priors = Priors(shield, shaping, shield_at_run, shaping_route)
        self.config = {
            "alpha": alpha, "alpha_decay": alpha_decay, "eps_start": eps_start,
            "eps_end": eps_end, "eps_fraction": eps_fraction, **self.priors.config,
        }  # fmt: skip
        self.w = np.zeros(N_FEATURES)
        self._tables: dict[int, tuple[Layout, np.ndarray]] = {}
        self._phi: np.ndarray | None = None

    def table(self, layout: Layout) -> np.ndarray:
        """Feature table of a layout, computed once and kept (the layout is kept alive too)."""
        entry = self._tables.get(id(layout))
        if entry is None:
            entry = (layout, feature_table(layout))
            self._tables[id(layout)] = entry
        return entry[1]

    def begin(self, layout: Layout, budget_steps: int, rng: np.random.Generator) -> None:
        self.rng = rng
        self.t = 0
        self._ramp = max(1.0, self.eps_fraction * budget_steps)
        self.start_episode(layout)

    def start_episode(self, layout: Layout) -> None:
        self.layout = layout
        self._phi = self.table(layout)
        self._allowed = self.priors.allowed(layout)

    def _allowed_at(self, cell: tuple[int, int]) -> np.ndarray | None:
        return None if self._allowed is None else self._allowed[cell]

    def epsilon(self) -> float:
        frac = min(1.0, self.t / self._ramp)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def act(self, cell: tuple[int, int]) -> int:
        allowed = self._allowed_at(cell)
        if self.rng.random() < self.epsilon():
            return explore(self.rng, allowed)
        return int(np.argmin(masked(self._phi[cell] @ self.w, allowed)))

    def observe(
        self,
        cell: tuple[int, int],
        action: int,
        cost: float,
        next_cell: tuple[int, int],
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.t += 1
        phi = self._phi[(*cell, action)]
        cost = self.priors.shape(self.layout, cell, next_cell, cost, terminated)
        if terminated:
            target = cost
        else:
            # A cost-to-go is never below the lowest value the potential allows (zero without
            # shaping): the clamp keeps the bootstrap from chasing an extrapolation beneath it,
            # which makes the weights diverge.
            potential = self.priors.potential(self.layout)
            floor = 0.0 if potential is None else -float(potential[next_cell])
            q_next = masked(self._phi[next_cell] @ self.w, self._allowed_at(next_cell))
            target = cost + max(floor, float(q_next.min()))
        step = self.alpha / (1.0 + self.alpha_decay * self.t)
        self.w += step * (target - phi @ self.w) * phi / (phi @ phi + 1e-8)

    def greedy(self, layout: Layout, cells: np.ndarray) -> np.ndarray:
        q = self.table(layout)[cells[:, 0], cells[:, 1]] @ self.w
        allowed = self.priors.run_allowed(layout)
        if allowed is not None:
            q = masked(q, allowed[cells[:, 0], cells[:, 1]])
        return np.argmin(q, axis=1)
