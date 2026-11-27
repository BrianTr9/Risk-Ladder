"""The two interfaces to a map: full access for planners, samples only for learners."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from riskladder.env.dynamics import ACTIONS, FAIL, N_ACTIONS, RUNNING, SUCCESS
from riskladder.env.layout import Layout
from riskladder.env.model import NEXT_FAIL, NEXT_SUCCESS, SparseMDP, build_model
from riskladder.env.spec import MapSpec


@dataclass
class EnvStats:
    steps: int = 0
    episodes: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0


class StepResult(NamedTuple):
    cell: tuple[int, int]
    cost: float
    terminated: bool  # reached SUCCESS or FAIL
    truncated: bool  # hit the horizon H
    status: int  # RUNNING, SUCCESS or FAIL


class RoverEnv:
    """Full-access environment: the exact model and a seeded sampler built from the same arrays.

    Learners must not receive this object; give them `sample_interface()` instead.
    """

    def __init__(self, spec: MapSpec, seed: int | None = None) -> None:
        self.spec = spec
        self._mdp = build_model(spec)
        self._rng = np.random.default_rng(seed)
        self._cum_prob = np.cumsum(self._mdp.branch_prob, axis=1)
        self._state: int | None = None
        self._t = 0
        self.stats = EnvStats()

    @property
    def layout(self) -> Layout:
        return self.spec.layout

    @property
    def horizon(self) -> int:
        return self.spec.horizon

    def model(self) -> SparseMDP:
        return self._mdp

    def reset(self, start: tuple[int, int] | None = None) -> tuple[int, int]:
        r, c = self.layout.start if start is None else start
        if (
            not (0 <= r < self.layout.rows and 0 <= c < self.layout.cols)
            or self._mdp.index[r, c] < 0
        ):
            raise ValueError(f"start {(r, c)} is not a decision cell")
        self._state = int(self._mdp.index[r, c])
        self._t = 0
        self.stats.episodes += 1
        return (r, c)

    def step(self, action: int) -> StepResult:
        if self._state is None:
            raise RuntimeError("episode is over: call reset() first")
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"action must be in [0, {N_ACTIONS}), got {action}")
        s = self._state
        b = int(np.searchsorted(self._cum_prob[action], self._rng.random(), side="right"))
        b = min(b, self._cum_prob.shape[1] - 1)  # guard against cumulative rounding below 1.0
        nxt = int(self._mdp.branch_next[s, action, b])
        cell = (
            int(self._mdp.branch_cell[s, action, b, 0]),
            int(self._mdp.branch_cell[s, action, b, 1]),
        )
        cost = float(self._mdp.branch_cost[s, action, b])
        self._t += 1
        self.stats.steps += 1

        if nxt == NEXT_SUCCESS:
            status, terminated = SUCCESS, True
            self.stats.successes += 1
        elif nxt == NEXT_FAIL:
            status, terminated = FAIL, True
            self.stats.failures += 1
        else:
            status, terminated = RUNNING, False
        truncated = not terminated and self._t >= self.horizon
        if truncated:
            self.stats.timeouts += 1
        self._state = None if terminated or truncated else nxt
        return StepResult(cell, cost, terminated, truncated, status)

    def sample_interface(self) -> SampleEnv:
        return SampleEnv(self)


class SampleEnv:
    """What a learner receives: the map, the horizon, `reset` and `step`.

    It has no `model()` and no noise or cost attributes (docs/env_spec.md section 8). Python cannot
    make the wrapped environment truly private, so agent code must use only this public interface.
    """

    __slots__ = ("__env",)

    n_actions = N_ACTIONS
    action_names = tuple(a.name for a in ACTIONS)

    def __init__(self, env: RoverEnv) -> None:
        self.__env = env

    @property
    def layout(self) -> Layout:
        return self.__env.layout

    @property
    def horizon(self) -> int:
        return self.__env.horizon

    @property
    def stats(self) -> EnvStats:
        return replace(self.__env.stats)

    def reset(self, start: tuple[int, int] | None = None) -> tuple[int, int]:
        return self.__env.reset(start)

    def step(self, action: int) -> StepResult:
        return self.__env.step(action)
