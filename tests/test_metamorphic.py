"""Metamorphic tests: relations between two runs that must hold without knowing the right answer.

The reference implementation in `reference.py` and the code were both written from the same spec,
so they could share a misreading. These relations come from domain knowledge instead (geometry,
units, monotonicity), so they check the solution from the outside. Each relation lists why it is
valid and what kind of fault it can expose.
"""

import itertools
from dataclasses import replace

import numpy as np
import pytest

from riskladder.env import Costs, Layout, MapSpec, Noise, build_model, generate
from riskladder.oracle import evaluate, value_iteration

SPECS = [
    generate(template, 12, seed=seed, noise=Noise(p_slip=0.12))
    for template in ("dilemma", "field")
    for seed in range(4)
]
IDS = [spec.name for spec in SPECS]


def solve(spec: MapSpec):
    mdp = build_model(spec)
    solution = value_iteration(mdp, tol=1e-12)
    at_start = evaluate(mdp, solution.policy, start=mdp.start).at(mdp.start)
    return mdp, solution, at_start


def mirror(grid):
    return tuple(row[::-1] for row in grid)


def rotate_clockwise(grid):
    rows, cols = len(grid), len(grid[0])
    return tuple("".join(grid[rows - 1 - c][r] for c in range(rows)) for r in range(cols))


def corresponding_states(mdp_from, mdp_to, cell_map):
    """For every state of `mdp_from`, the index of the matching state of `mdp_to`."""
    return np.array([mdp_to.index[cell_map(int(r), int(c))] for r, c in mdp_from.cells])


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_mirror_image_has_the_same_values(spec):
    """Equivariance. Valid: flipping left-right swaps E and W and nothing else; the slip model is
    symmetric. Exposes: a direction table, slip rule or collision rule that is not symmetric."""
    mdp, solution, result = solve(spec)
    flipped = replace(spec, layout=Layout(mirror(spec.layout.grid)))
    mdp2, solution2, result2 = solve(flipped)
    cols = spec.layout.cols
    to = corresponding_states(mdp, mdp2, lambda r, c: (r, cols - 1 - c))
    np.testing.assert_allclose(solution.V, solution2.V[to], atol=1e-7)
    assert result2.p_fail == pytest.approx(result.p_fail, abs=1e-9)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_rotated_map_has_the_same_values(spec):
    """Equivariance under a 90 degree turn (N -> E -> S -> W). Valid: the map is square and every
    rule treats the four directions alike. Exposes: a swapped row/column or axis in the dynamics."""
    mdp, solution, result = solve(spec)
    turned = replace(spec, layout=Layout(rotate_clockwise(spec.layout.grid)))
    mdp2, solution2, result2 = solve(turned)
    rows = spec.layout.rows
    to = corresponding_states(mdp, mdp2, lambda r, c: (c, rows - 1 - r))
    np.testing.assert_allclose(solution.V, solution2.V[to], atol=1e-7)
    assert result2.p_fail == pytest.approx(result.p_fail, abs=1e-9)


@pytest.mark.parametrize("k", [0.5, 2.5, 10.0])
@pytest.mark.parametrize("spec", SPECS[:4], ids=IDS[:4])
def test_scaling_all_costs_scales_values_and_keeps_risk(spec, k):
    """Equivariance. Valid for k > 0 only, and only if the failure cost scales too: it does, because
    it is proportional to the move cost (with an explicit `c_fail` it must be scaled by hand).
    Exposes: a cost that is not scaled with the others, or a stray constant in a cost."""
    _, solution, result = solve(spec)
    c = spec.costs
    scaled = replace(spec, costs=Costs(c.move * k, c.dash * k, c.rough_extra * k, c.collision * k))
    _, solution2, result2 = solve(scaled)
    np.testing.assert_allclose(solution2.V, k * solution.V, rtol=1e-7)
    assert result2.p_fail == pytest.approx(result.p_fail, abs=1e-9)
    assert result2.cost == pytest.approx(k * result.cost, rel=1e-7)


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_a_larger_failure_cost_never_lowers_cost_and_never_raises_risk(spec):
    """Monotonicity. Valid: for c1 < c2 with optimal policies p1, p2 the two optimality conditions
    add up to (c2 - c1)(F2 - F1) <= 0, whatever the ties. Exposes: failure cost applied twice or
    not at all, and sign errors in the failure bookkeeping."""
    rows = [solve(replace(spec, c_fail_override=c))[2] for c in (10.0, 40.0, 150.0, 600.0, 2400.0)]
    for low, high in itertools.pairwise(rows):
        assert high.cost >= low.cost - 1e-9
        assert high.p_fail <= low.p_fail + 1e-9


@pytest.mark.parametrize("parameter", ["move", "dash", "rough_extra", "collision"])
@pytest.mark.parametrize("spec", SPECS[:4], ids=IDS[:4])
def test_a_larger_cost_never_lowers_the_optimal_value(spec, parameter):
    """Monotonicity. Valid: raising one cost raises (or keeps) the cost of every policy, so the
    minimum cannot fall. Exposes: a cost that is read from the wrong field or ignored."""
    c = spec.costs
    values = []
    for factor in (1.0, 1.5, 3.0):
        costs = replace(c, **{parameter: getattr(c, parameter) * factor})
        values.append(solve(replace(spec, costs=costs))[1].V[build_model(spec).start])
    assert values[0] <= values[1] + 1e-9 <= values[2] + 2e-9


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_a_sealed_region_changes_nothing(spec):
    """Invariance. Valid: cells the rover can never reach cannot matter, including for the failure
    cost (D_max counts reachable cells only). Exposes: unreachable cells leaking into a result."""
    cols = spec.layout.cols
    sealed = ("#" * cols, "#" + "." * (cols - 3) + "G#", "#" * cols)
    bigger = replace(spec, layout=Layout(tuple(spec.layout.grid) + sealed))
    assert bigger.c_fail == spec.c_fail
    mdp, solution, result = solve(spec)
    mdp2, solution2, result2 = solve(bigger)
    to = corresponding_states(mdp, mdp2, lambda r, c: (r, c))
    np.testing.assert_allclose(solution.V, solution2.V[to], atol=1e-8)
    assert result2.p_fail == pytest.approx(result.p_fail, abs=1e-12)


def test_the_relation_that_looks_valid_but_is_not():
    """Documented trap: scaling the costs but keeping a fixed `c_fail` does NOT scale the values.

    'Multiply every cost by k, so every value is multiplied by k' is plausible but wrong when a
    cost (here the failure cost) is held fixed, so scaling tests must scale it as well.
    """
    spec = SPECS[0]
    fixed = replace(spec, c_fail_override=spec.c_fail)
    base = solve(fixed)[2]
    c = spec.costs
    scaled = replace(fixed, costs=Costs(c.move * 3, c.dash * 3, c.rough_extra * 3, c.collision * 3))
    assert solve(scaled)[2].cost != pytest.approx(3 * base.cost, rel=1e-3)
