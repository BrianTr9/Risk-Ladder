"""Helpers shared by tests that compare the code with the reference implementation."""

import pytest

from riskladder.env.dynamics import FAIL, RUNNING, SUCCESS

STATUS = {RUNNING: "running", SUCCESS: "success", FAIL: "fail"}


def merged(items):
    """Merge (end cell, cost, status, probability) into {(cell, cost, status): probability}."""
    out = {}
    for end, cost, status, p in items:
        key = (tuple(int(v) for v in end), round(float(cost), 9), status)
        out[key] = out.get(key, 0.0) + float(p)
    return out


def assert_same(actual, expected, context=""):
    assert set(actual) == set(expected), context
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], abs=1e-12), (context, key)


def model_outcomes(mdp, s, action):
    """The outcomes of (state, action) as stored in the vectorised model's branch arrays."""
    items = []
    for b in range(3):
        nxt = int(mdp.branch_next[s, action, b])
        status = "success" if nxt == -1 else "fail" if nxt == -2 else "running"
        end = tuple(int(v) for v in mdp.branch_cell[s, action, b])
        probability = mdp.branch_prob[action, b]
        if probability > 0:
            items.append((end, mdp.branch_cost[s, action, b], status, probability))
    return merged(items)
