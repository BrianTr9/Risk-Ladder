"""Pairwise (2-way) coverage of the configuration space, checked against the reference.

Exhaustively combining the nine inputs below would be 1,728 maps. Most faults are triggered by one
or two inputs together, so a 2-way covering array exercises every pair of values in 12 maps. Values
include the boundary points (p_slip 0 and 0.5 with a dash multiplier that makes the dash slip
probability exactly 1, zero extra costs, a dash dearer or cheaper than a move).
"""

import itertools

import numpy as np
import pytest
from allpairspy import AllPairs

import reference as ref
from helpers import STATUS, assert_same, merged, model_outcomes
from riskladder.env import Costs, MapSpec, Noise, build_model, generate, outcomes
from riskladder.oracle import evaluate, value_iteration

PARAMETERS = {
    "template": ["dilemma", "field"],
    "p_slip": [0.0, 0.05, 0.5],
    "dash_mult": [0.0, 1.0, 2.0],  # with p_slip 0.5 the largest value gives slip probability 1
    "move_dash": [(1.0, 1.5), (0.8, 0.8), (0.1, 5.0)],
    "rough_extra": [0.0, 2.5],
    "collision": [0.0, 2.0],
    "fail_multiplier": [0.5, 5.0],
    "size": [10, 13],
    "seed": [0, 1],
}
NAMES = list(PARAMETERS)
ROWS = [dict(zip(NAMES, row, strict=True)) for row in AllPairs(list(PARAMETERS.values()))]


def build(row) -> MapSpec:
    move, dash = row["move_dash"]
    base = generate(
        row["template"],
        row["size"],
        seed=row["seed"],
        noise=Noise(row["p_slip"], row["dash_mult"]),
        costs=Costs(move, dash, row["rough_extra"], row["collision"]),
    )
    return MapSpec(
        base.layout,
        noise=base.noise,
        costs=base.costs,
        fail_multiplier=row["fail_multiplier"],
        seed=row["seed"],
    )


def test_the_covering_array_covers_every_pair_of_values():
    """The array property itself: every value pair of every two inputs appears in some row."""
    for a, b in itertools.combinations(NAMES, 2):
        wanted = set(itertools.product(range(len(PARAMETERS[a])), range(len(PARAMETERS[b]))))
        seen = {(PARAMETERS[a].index(row[a]), PARAMETERS[b].index(row[b])) for row in ROWS}
        assert seen == wanted, (a, b)


def test_the_covering_array_is_small_but_not_below_the_lower_bound():
    largest_pair = max(
        len(PARAMETERS[a]) * len(PARAMETERS[b]) for a, b in itertools.combinations(NAMES, 2)
    )
    exhaustive = int(np.prod([len(v) for v in PARAMETERS.values()]))
    assert largest_pair <= len(ROWS) < exhaustive / 50


@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r['template']}-{r['size']}-p{r['p_slip']}")
def test_every_configuration_agrees_with_the_reference(row):
    spec = build(row)
    mdp = build_model(spec)
    for s, cell in enumerate(map(tuple, mdp.cells)):
        for action in range(8):
            expected = ref.outcomes(spec, cell, action)
            got = merged(
                (o.cell, o.cost, STATUS[o.status], o.prob) for o in outcomes(spec, cell, action)
            )
            assert_same(got, expected, (cell, action))
            assert_same(model_outcomes(mdp, s, action), expected, (cell, action))

    # the oracle's exact evaluation of the optimal policy against naive forward propagation
    policy = value_iteration(mdp, tol=1e-10).policy
    as_dict = {tuple(c): int(policy[i]) for i, c in enumerate(mdp.cells)}
    start = tuple(spec.layout.start)
    expected = ref.evaluate(spec, as_dict, start, 50)
    got = evaluate(mdp, policy, horizon=50, tol=0, start=mdp.start).at(mdp.start)
    np.testing.assert_allclose(
        [got.cost, got.p_success, got.p_fail, got.p_timeout], expected, atol=1e-8
    )
