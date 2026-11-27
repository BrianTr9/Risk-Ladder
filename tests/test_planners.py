import itertools
from pathlib import Path

import numpy as np
import pytest

from riskladder.agents import PenalizedMDP, astar, cost_per_cell, heuristic_table, penalty_planner
from riskladder.env import (
    ACTIONS,
    Costs,
    Layout,
    MapSpec,
    Noise,
    RoverEnv,
    build_model,
    generate,
    load_map,
    outcomes,
)
from riskladder.env.dynamics import FAIL, SUCCESS
from riskladder.env.layout import PASSABLE, grid_bfs
from riskladder.oracle import evaluate_mixture, value_iteration

MAPS = Path(__file__).parents[1] / "experiments" / "maps"
RIDGES = sorted((MAPS / "ridge").glob("*.map"))
NOISE_FREE = Noise(p_slip=0.0, dash_slip_mult=0.0)

SPECS = [
    generate(template, seed=seed, **kwargs)
    for template, kwargs in (
        ("dilemma", {"rows": 20}),
        ("ridge", {"rows": 6, "cols": 30}),
        ("field", {"rows": 16}),
    )
    for seed in range(4)
]
IDS = [spec.name for spec in SPECS]


def noise_free_values(spec):
    mdp = build_model(spec.with_noise(NOISE_FREE))
    return mdp, value_iteration(mdp, tol=1e-12)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_astar_matches_value_iteration_without_noise(spec):
    mdp, solution = noise_free_values(spec)
    plan = astar(spec)
    assert plan.found
    assert plan.cost == pytest.approx(solution.V[mdp.start], abs=1e-9)
    assert astar(spec, use_heuristic=False).cost == pytest.approx(plan.cost, abs=1e-9)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_astar_plan_replays_in_the_environment(spec):
    exact = spec.with_noise(NOISE_FREE)
    plan = astar(spec)
    env = RoverEnv(exact, seed=0)
    env.reset()
    total, result = 0.0, None
    for action in plan.actions:
        result = env.step(action)
        total += result.cost
    assert result is not None and result.status == SUCCESS
    assert total == pytest.approx(plan.cost, abs=1e-9)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_heuristic_is_admissible_and_consistent(spec):
    """On every cell the search can visit (reachable from the start without touching a hazard)."""
    mdp, solution = noise_free_values(spec)
    h = heuristic_table(spec)
    exact = spec.with_noise(NOISE_FREE)
    reachable = grid_bfs(spec.layout.mask(PASSABLE), [spec.layout.start]) < np.inf
    checked = 0
    for s, cell in enumerate(map(tuple, mdp.cells)):
        if not reachable[cell]:
            continue
        checked += 1
        assert h[cell] <= solution.V[s] + 1e-9  # never above the true cost-to-go
        for action in range(8):
            (move,) = outcomes(exact, cell, action)
            if move.status == FAIL:
                continue
            successor = 0.0 if move.status == SUCCESS else h[move.cell]
            assert h[cell] <= move.cost + successor + 1e-9  # consistency
    assert checked > 10


def test_heuristic_saves_expansions_and_never_costs_more():
    for spec in SPECS:
        with_h, without_h = astar(spec), astar(spec, use_heuristic=False)
        assert with_h.expanded <= without_h.expanded
    field = generate("field", 30, seed=2)
    assert astar(field).expanded < astar(field, use_heuristic=False).expanded / 2


def test_astar_tests_for_the_goal_when_a_node_is_expanded_not_when_it_is_generated():
    """With a dear dash the goal is first reached by an expensive route; only the cheapest entry may
    end the search. Testing at generation would return 5.0 here, the optimum is 2.0."""
    spec = MapSpec(Layout.from_text("#S.G#\n#####"), costs=Costs(move=1.0, dash=5.0))
    assert astar(spec).cost == pytest.approx(2.0)
    assert [ACTIONS[a].name for a in astar(spec).actions] == ["move_E", "move_E"]


def test_astar_without_a_route_reports_failure():
    spec = MapSpec(Layout.from_text("#S.G#"))
    blocked = MapSpec(Layout.from_text("#S.G#\n#####"))
    assert astar(spec).found and astar(blocked).found
    assert cost_per_cell(spec) == min(spec.costs.move, spec.costs.dash / 2)


def test_penalty_planner_at_zero_is_the_environment_optimum():
    mdp = build_model(load_map(RIDGES[0]))
    plain = value_iteration(mdp, tol=1e-10)
    penalised = penalty_planner(mdp, 0.0)
    np.testing.assert_allclose(penalised.V, plain.V, atol=1e-9)
    np.testing.assert_array_equal(penalised.policy, plain.policy)


def test_penalized_mdp_adds_the_penalty_to_failures_only():
    mdp = build_model(load_map(RIDGES[0]))
    penalised = PenalizedMDP(mdp, 7.0)
    np.testing.assert_allclose(penalised.cost - mdp.cost, 7.0 * mdp.p_fail)
    assert penalised.n_states == mdp.n_states and penalised.start == mdp.start
    assert penalised.horizon == mdp.horizon and penalised.P_stack is mdp.P_stack
    with pytest.raises(ValueError):
        PenalizedMDP(mdp, -1.0)


@pytest.mark.parametrize("path", RIDGES, ids=lambda p: p.stem)
def test_penalty_sweep_trades_risk_for_cost_in_steps(path):
    """A larger penalty never raises the failure probability and never lowers the cost, and on the
    ridge maps the sweep passes through several distinct policies (a staircase, not one jump)."""
    mdp = build_model(load_map(path))
    points = []
    for extra in [0.0, *np.geomspace(1.0, 3000.0, 40)]:
        result = evaluate_mixture(mdp, penalty_planner(mdp, extra).policy)
        points.append((result.cost, result.p_fail))
    for (cost_a, fail_a), (cost_b, fail_b) in itertools.pairwise(points):
        assert fail_b <= fail_a + 1e-9
        assert cost_b >= cost_a - 1e-9
    distinct = {(round(c, 6), round(f, 8)) for c, f in points}
    assert len(distinct) >= 3, f"{path.stem} has only {len(distinct)} distinct policies"


def test_most_ridge_maps_have_a_rich_staircase():
    counts = []
    for path in RIDGES:
        mdp = build_model(load_map(path))
        seen = set()
        for extra in [0.0, *np.geomspace(1.0, 3000.0, 40)]:
            result = evaluate_mixture(mdp, penalty_planner(mdp, extra).policy)
            seen.add((round(result.cost, 6), round(result.p_fail, 8)))
        counts.append(len(seen))
    assert np.median(counts) >= 4
