import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from riskladder.agents import frontier, penalty_planner, solve_cmdp
from riskladder.env import Costs, Layout, MapSpec, Noise, build_model, load_map
from riskladder.oracle import evaluate, evaluate_mixture

MAPS = Path(__file__).parents[1] / "experiments" / "maps"
RIDGES = sorted((MAPS / "ridge").glob("*.map"))
DELTAS = [0.0, 1e-4, 1e-3, 3e-3, 0.01, 0.02, 0.03, 0.05, 0.5, 1.0]


class RiskyChain:
    """Six states in a row. Action 0 is safe (cost 4); action 1 is risky (cost 3, fails w.p. 0.1).

    A state's expected cost includes the failure cost, as in the environment. All 2^6 deterministic
    policies can be enumerated, which gives an independent answer to the constrained problem.
    """

    n_states, n_actions, start, horizon = 6, 2, 0, 200
    c_fail = 20.0

    def __init__(self):
        n = self.n_states
        forward = sparse.diags([np.ones(n - 1)], [1], shape=(n, n), format="csr")
        risky = sparse.diags([np.full(n - 1, 0.9)], [1], shape=(n, n), format="csr")
        self.P_stack = sparse.vstack([forward, risky], format="csr")
        self.p_success = np.zeros((n, 2))
        self.p_success[-1] = [1.0, 0.9]
        self.p_fail = np.zeros((n, 2))
        self.p_fail[:, 1] = 0.1
        self.cost = np.column_stack([np.full(n, 4.0), np.full(n, 1.0 + 0.1 * self.c_fail)])


def brute_force_constrained_cost(mdp, delta):
    """Cheapest mixture of two deterministic policies meeting the bound, over all policies."""
    points = []
    for bits in itertools.product([0, 1], repeat=mdp.n_states):
        result = evaluate(mdp, np.array(bits), start=mdp.start).at(mdp.start)
        points.append((result.p_fail, result.cost))
    best = min((cost for fail, cost in points if fail <= delta + 1e-12), default=np.inf)
    for (f1, c1), (f2, c2) in itertools.combinations(points, 2):
        if f1 < delta < f2 and f2 - f1 > 1e-12:
            weight = (delta - f1) / (f2 - f1)
            best = min(best, (1 - weight) * c1 + weight * c2)
    return best


@pytest.mark.parametrize("delta", [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9])
def test_cmdp_matches_brute_force_over_all_policies(delta):
    """Independent check: enumerate every deterministic policy and mix the best pair."""
    mdp = RiskyChain()
    solution = solve_cmdp(mdp, delta)
    assert solution.feasible
    assert solution.result.cost == pytest.approx(brute_force_constrained_cost(mdp, delta), abs=1e-7)
    assert solution.result.p_fail <= delta + 1e-9
    assert solution.gap == pytest.approx(0.0, abs=1e-7)  # the dual bound certifies optimality


@pytest.mark.parametrize("path", RIDGES[:4], ids=lambda p: p.stem)
def test_cmdp_holds_the_bound_exactly_with_a_zero_duality_gap(path):
    mdp = build_model(load_map(path))
    unconstrained = evaluate_mixture(mdp, penalty_planner(mdp, 0.0).policy)
    for solution in frontier(mdp, DELTAS):
        assert solution.feasible
        assert solution.result.p_fail <= solution.delta + 1e-9
        assert solution.gap == pytest.approx(0.0, abs=1e-6)
        assert len(solution.mixture) <= 2
        assert sum(w for w, _ in solution.mixture) == pytest.approx(1.0)
        if solution.delta < unconstrained.p_fail - 1e-9:  # the bound binds: it is met exactly
            assert solution.result.p_fail == pytest.approx(solution.delta, abs=1e-9)
            assert solution.multiplier > 0
        else:  # the bound is slack: nothing changes
            assert solution.multiplier == 0
            assert solution.result.cost == pytest.approx(unconstrained.cost)


@pytest.mark.parametrize("path", RIDGES[:4], ids=lambda p: p.stem)
def test_a_looser_bound_never_costs_more(path):
    """Monotonicity: the feasible set grows with delta, so the optimal cost cannot rise."""
    mdp = build_model(load_map(path))
    costs = [s.result.cost for s in frontier(mdp, DELTAS)]
    for tight, loose in itertools.pairwise(costs):
        assert loose <= tight + 1e-9


@pytest.mark.parametrize("path", RIDGES[:3], ids=lambda p: p.stem)
def test_cmdp_beats_every_penalty_policy_that_meets_the_bound(path):
    mdp = build_model(load_map(path))
    penalty_results = [
        evaluate_mixture(mdp, penalty_planner(mdp, extra).policy)
        for extra in [0.0, *np.geomspace(1.0, 3000.0, 60)]
    ]
    for delta in (0.001, 0.005, 0.01, 0.02):
        cmdp = solve_cmdp(mdp, delta)
        for other in penalty_results:
            if other.p_fail <= delta + 1e-12:
                assert cmdp.result.cost <= other.cost + 1e-9
        # the deterministic policy that meets the bound is itself one of the penalty policies
        deploy = evaluate_mixture(mdp, cmdp.penalty_policy)
        assert deploy.p_fail <= delta + 1e-9 and deploy.cost >= cmdp.result.cost - 1e-9


def test_scaling_costs_scales_the_frontier_but_not_the_risk():
    """Equivariance: with every cost multiplied by k > 0 the same mixture is optimal."""
    spec = load_map(RIDGES[0])
    c, k = spec.costs, 3.0
    scaled = replace(spec, costs=Costs(c.move * k, c.dash * k, c.rough_extra * k, c.collision * k))
    mdp, mdp2 = build_model(spec), build_model(scaled)
    for delta in (0.003, 0.02):
        a, b = solve_cmdp(mdp, delta), solve_cmdp(mdp2, delta)
        assert b.result.cost == pytest.approx(k * a.result.cost, rel=1e-8)
        assert b.result.p_fail == pytest.approx(a.result.p_fail, abs=1e-9)
        assert [round(w, 6) for w, _ in b.mixture] == [round(w, 6) for w, _ in a.mixture]


def test_an_unavoidable_hazard_makes_a_zero_bound_infeasible():
    """Hazards on both sides of every cell on the route: each step risks a slip into one of them."""
    layout = Layout.from_text("#XXX#\n#S.G#\n#XXX#")
    mdp = build_model(MapSpec(layout, noise=Noise(p_slip=0.1)))
    impossible = solve_cmdp(mdp, 0.0)
    assert not impossible.feasible and impossible.lower_bound == -np.inf
    assert solve_cmdp(mdp, 0.9).feasible
    smallest = solve_cmdp(mdp, 0.9).result.p_fail  # any reachable risk level is feasible above it
    assert smallest > 0


def test_invalid_bounds_are_rejected():
    mdp = RiskyChain()
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            solve_cmdp(mdp, bad)
