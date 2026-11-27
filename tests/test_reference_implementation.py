"""Check the vectorised implementation against `tests/reference.py` (written from the spec text)."""

from pathlib import Path

import numpy as np
import pytest

import reference as ref
from riskladder.env import Costs, Layout, MapSpec, Noise, build_model, generate, load_map, outcomes
from riskladder.env.dynamics import FAIL, RUNNING, SUCCESS
from riskladder.oracle import Oracle, evaluate, policy_iteration, value_iteration

MAPS = sorted((Path(__file__).parents[1] / "experiments" / "maps").rglob("*.map"))
STATUS = {RUNNING: "running", SUCCESS: "success", FAIL: "fail"}
OTHER_NOISE = Noise(p_slip=0.3, dash_slip_mult=1.5)
OTHER_COSTS = Costs(move=0.8, dash=1.1, rough_extra=2.5, collision=0.5)


def specs_under_test(hand_spec):
    yield "hand", hand_spec
    for template in ("dilemma", "field"):
        for seed in (0, 1, 2):
            base = generate(template, 12, seed=seed)
            yield f"{template}-{seed}", base
            yield (
                f"{template}-{seed}-other",
                MapSpec(base.layout, noise=OTHER_NOISE, costs=OTHER_COSTS, fail_multiplier=3.0),
            )


def merged(items):
    out = {}
    for end, cost, status, p in items:
        key = (tuple(int(v) for v in end), round(float(cost), 9), status)
        out[key] = out.get(key, 0.0) + float(p)
    return out


def assert_same(actual, expected):
    assert set(actual) == set(expected)
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], abs=1e-12), key


@pytest.mark.parametrize("path", MAPS, ids=lambda p: p.stem)
def test_failure_cost_matches_independent_bfs(path):
    spec = load_map(path)
    assert spec.c_fail == pytest.approx(ref.c_fail(spec))


def test_failure_cost_on_a_known_map(hand_spec):
    # hand map: farthest reachable cell from the goal is 6 steps away -> 5 * 6 * 1.0
    assert ref.c_fail(hand_spec) == 30.0 == hand_spec.c_fail


def test_outcomes_match_the_reference(hand_spec):
    for name, spec in specs_under_test(hand_spec):
        for cell in spec.layout.passable_cells:
            for action in range(8):
                got = merged(
                    (o.cell, o.cost, STATUS[o.status], o.prob) for o in outcomes(spec, cell, action)
                )
                assert_same(got, ref.outcomes(spec, cell, action)), (name, cell, action)


def test_vectorised_model_matches_the_reference(hand_spec):
    """`build_model` is checked on its own branch arrays, not through `outcomes`."""
    for name, spec in specs_under_test(hand_spec):
        mdp = build_model(spec)
        for s, cell in enumerate(map(tuple, mdp.cells)):
            for action in range(8):
                items = []
                for b in range(3):
                    nxt = int(mdp.branch_next[s, action, b])
                    end = tuple(mdp.branch_cell[s, action, b])
                    status = "success" if nxt == -1 else "fail" if nxt == -2 else "running"
                    items.append(
                        (end, mdp.branch_cost[s, action, b], status, mdp.branch_prob[action, b])
                    )
                items = [i for i in items if i[3] > 0]
                assert_same(merged(items), ref.outcomes(spec, cell, action)), (name, cell, action)


def small_spec():
    return generate("field", 8, seed=5, noise=Noise(p_slip=0.2))


def test_optimal_values_match_a_naive_solver():
    spec = small_spec()
    mdp = build_model(spec)
    expected = ref.optimal_values(spec)
    reference_v = np.array([expected[tuple(c)] for c in mdp.cells])
    np.testing.assert_allclose(value_iteration(mdp, tol=1e-12).V, reference_v, atol=1e-7)
    np.testing.assert_allclose(policy_iteration(mdp).V, reference_v, atol=1e-7)


def test_policy_evaluation_matches_naive_forward_propagation():
    spec = small_spec()
    mdp = build_model(spec)
    horizon = 80
    rng = np.random.default_rng(0)
    policies = {
        "optimal": value_iteration(mdp, tol=1e-12).policy,
        "random": rng.integers(0, 8, mdp.n_states),
        "all dash east": np.full(mdp.n_states, 5),
    }
    for name, policy in policies.items():
        as_dict = {tuple(c): int(policy[s]) for s, c in enumerate(mdp.cells)}
        expected = ref.evaluate(spec, as_dict, tuple(spec.layout.start), horizon)
        got = evaluate(mdp, policy, horizon=horizon, tol=0, start=mdp.start).at(mdp.start)
        np.testing.assert_allclose(
            [got.cost, got.p_success, got.p_fail, got.p_timeout], expected, atol=1e-9, err_msg=name
        )


def test_regret_matches_the_naive_solver():
    spec = small_spec()
    mdp = build_model(spec)
    oracle = Oracle(mdp, horizon=400)
    policy = np.random.default_rng(1).integers(0, 8, mdp.n_states)
    as_dict = {tuple(c): int(policy[s]) for s, c in enumerate(mdp.cells)}
    start = tuple(spec.layout.start)
    expected = ref.evaluate(spec, as_dict, start, 400)[0] - ref.optimal_values(spec)[start]
    assert oracle.regret(policy) == pytest.approx(expected, abs=1e-6)


def test_reference_agrees_with_a_hand_worked_corridor():
    """Anchor the reference itself to a number worked out on paper (see test_oracle_solve)."""
    spec = MapSpec(Layout.from_text("#S.G#\n#####"), noise=Noise(p_slip=0.2))
    values = ref.optimal_values(spec)
    assert values[(0, 1)] == pytest.approx(3.5) and values[(0, 2)] == pytest.approx(1.75)
