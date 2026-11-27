import itertools

import numpy as np
import pytest

from riskladder.env import FAIL, N_ACTIONS, SUCCESS, build_model, generate, outcomes


@pytest.fixture(params=["hand", "dilemma", "field"])
def spec(request, hand_spec):
    if request.param == "hand":
        return hand_spec
    return generate(request.param, 12, seed=7)


def test_rows_are_distributions(spec):
    mdp = build_model(spec)
    for a in range(N_ACTIONS):
        total = np.asarray(mdp.P[a].sum(axis=1)).ravel() + mdp.p_success[:, a] + mdp.p_fail[:, a]
        np.testing.assert_allclose(total, 1.0)
        assert mdp.P[a].min() >= 0


def test_model_matches_scalar_outcomes(spec):
    """The batch model and the single-cell `outcomes` path must agree on every (s, a)."""
    mdp = build_model(spec)
    for (s, cell), a in itertools.product(enumerate(spec.layout.passable_cells), range(N_ACTIONS)):
        row = np.zeros(mdp.n_states)
        p_success = p_fail = expected_cost = 0.0
        for o in outcomes(spec, cell, a):
            expected_cost += o.prob * o.cost
            if o.status == SUCCESS:
                p_success += o.prob
            elif o.status == FAIL:
                p_fail += o.prob
            else:
                row[mdp.index[o.cell]] += o.prob
        np.testing.assert_allclose(mdp.P[a][s].toarray().ravel(), row, atol=1e-12)
        assert mdp.p_success[s, a] == pytest.approx(p_success)
        assert mdp.p_fail[s, a] == pytest.approx(p_fail)
        assert mdp.cost[s, a] == pytest.approx(expected_cost)


def test_indexing_and_grid_mapping(hand_spec):
    mdp = build_model(hand_spec)
    assert mdp.n_states == 8
    assert tuple(mdp.cells[mdp.start]) == hand_spec.layout.start
    for s, (r, c) in enumerate(mdp.cells):
        assert mdp.index[r, c] == s
    grid = mdp.to_grid(np.arange(mdp.n_states, dtype=float))
    assert grid[1, 1] == 0 and np.isnan(grid[0, 0]) and np.isnan(grid[1, 4])


def test_arrays_are_read_only(hand_spec):
    mdp = build_model(hand_spec)
    with pytest.raises(ValueError):
        mdp.cost[0, 0] = 0.0
