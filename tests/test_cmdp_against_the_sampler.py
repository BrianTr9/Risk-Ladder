"""The constrained planner's exact numbers against episodes sampled from the environment.

The solver and the exact evaluation share linear algebra; the sampler does not. Playing the mixture
for many episodes through `RoverEnv.step` and comparing is an independent check of the whole chain
(model, constrained solve, evaluation) on real maps.
"""

from pathlib import Path

import numpy as np
import pytest

from riskladder.agents import solve_cmdp
from riskladder.env import RoverEnv, load_map

pytestmark = pytest.mark.slow

MAPS = sorted((Path(__file__).parents[1] / "experiments" / "maps" / "ridge").glob("*.map"))[:3]


def play(env, grids, weights, episodes, rng):
    costs, fails = np.empty(episodes), np.zeros(episodes, dtype=bool)
    for i in range(episodes):
        grid = grids[rng.choice(len(grids), p=weights)]
        cell, total = env.reset(), 0.0
        while True:
            result = env.step(int(grid[cell]))
            total += result.cost
            if result.terminated or result.truncated:
                break
            cell = result.cell
        costs[i], fails[i] = total, result.status == 2
    return costs, fails


@pytest.mark.parametrize("path", MAPS, ids=lambda p: p.stem)
@pytest.mark.parametrize("delta", [0.005, 0.02])
def test_sampled_episodes_agree_with_the_exact_failure_probability_and_cost(path, delta):
    env = RoverEnv(load_map(path), seed=7)
    mdp = env.model()
    solution = solve_cmdp(mdp, delta)
    grids = [mdp.to_grid(p, fill=0) for _, p in solution.mixture]
    weights = np.array([w for w, _ in solution.mixture])
    episodes = 30_000
    costs, fails = play(env, grids, weights, episodes, np.random.default_rng(1))
    exact = solution.result
    sigma = np.sqrt(exact.p_fail * (1 - exact.p_fail) / episodes)
    assert abs(fails.mean() - exact.p_fail) <= 4 * sigma + 1e-9
    assert abs(costs.mean() - exact.cost) <= 4 * costs.std(ddof=1) / np.sqrt(episodes)
    assert exact.p_fail <= delta + 1e-9  # and the exact value respects the bound
