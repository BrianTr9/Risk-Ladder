"""The training loop: a learner interacts with `SampleEnv`, and exact scores are taken on the side.

A learner needs five things: `begin`, `start_episode`, `act`, `observe` and `greedy`. The loop owns
episodes, the step budget and the bookkeeping, so every learner is run and measured the same way.
Scoring uses the exact model through a `Scorer`; it never reaches the learner.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from riskladder.env import FAIL, SUCCESS, SampleEnv
from riskladder.env.layout import Layout
from riskladder.harness.scoring import Score, Scorer


class Learner(Protocol):
    name: str

    def begin(self, layout: Layout, budget_steps: int, rng: np.random.Generator) -> None: ...

    def start_episode(self, layout: Layout) -> None: ...

    def act(self, cell: tuple[int, int]) -> int: ...

    def observe(
        self,
        cell: tuple[int, int],
        action: int,
        cost: float,
        next_cell: tuple[int, int],
        terminated: bool,
        truncated: bool,
    ) -> None: ...

    def greedy(self, layout: Layout, cells: np.ndarray) -> np.ndarray:
        """Greedy action for each (row, col) in `cells`, shape (m, 2) -> (m,)."""
        ...


@dataclass(frozen=True)
class Checkpoint:
    """State of training after `steps` environment steps."""

    steps: int
    episodes: int
    train_failures: int  # episodes that ended in a hazard so far
    train_successes: int
    train_cost: float  # cumulative sampled cost, including failure costs
    scores: tuple[Score, ...]  # one per scorer, of the greedy policy at this moment


def train(
    learner: Learner,
    envs: SampleEnv | Sequence[SampleEnv],
    budget_steps: int,
    rng: np.random.Generator,
    scorers: Sequence[Scorer] = (),
    n_checkpoints: int = 25,
) -> list[Checkpoint]:
    """Run `learner` for exactly `budget_steps` environment steps and return its checkpoints.

    With several environments each episode picks one at random, which trains across maps.
    Checkpoints come every `budget_steps / n_checkpoints` steps, plus one before any training, and
    each scores the greedy policy exactly. Episodes are cut at the environment's horizon.
    """
    if budget_steps < 1:
        raise ValueError("budget_steps must be at least 1")
    if n_checkpoints < 1:
        raise ValueError("n_checkpoints must be at least 1")
    pool = [envs] if isinstance(envs, SampleEnv) else list(envs)
    if not pool:
        raise ValueError("give at least one environment")
    every = max(1, budget_steps // n_checkpoints)

    learner.begin(pool[0].layout, budget_steps, rng)
    steps = episodes = failures = successes = 0
    cost = 0.0
    checkpoints: list[Checkpoint] = []

    def snapshot() -> None:
        scores = tuple(s.score(learner) for s in scorers)
        checkpoints.append(Checkpoint(steps, episodes, failures, successes, cost, scores))

    snapshot()
    while steps < budget_steps:
        env = pool[0] if len(pool) == 1 else pool[int(rng.integers(len(pool)))]
        cell = env.reset()
        learner.start_episode(env.layout)
        episodes += 1
        while steps < budget_steps:
            action = learner.act(cell)
            result = env.step(action)
            learner.observe(
                cell, action, result.cost, result.cell, result.terminated, result.truncated
            )
            steps += 1
            cost += result.cost
            failures += result.status == FAIL
            successes += result.status == SUCCESS
            if steps % every == 0 or steps == budget_steps:
                snapshot()
            if result.terminated or result.truncated:
                break
            cell = result.cell
    return checkpoints
