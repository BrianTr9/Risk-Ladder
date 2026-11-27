import warnings
from pathlib import Path

import numpy as np
import pytest

from riskladder.env import Noise, build_model, generate, load_map
from riskladder.oracle import (
    Oracle,
    evaluate,
    evaluate_mixture,
    solve_policy,
    value_iteration,
)

MAPS = sorted((Path(__file__).parents[1] / "experiments" / "maps").rglob("*.map"))
MOVE_N = 0


@pytest.mark.parametrize("path", MAPS, ids=lambda p: p.stem)
def test_optimal_policy_terminates_within_the_horizon(path):
    """env_spec 6.3: H is large enough that the optimal policy's timeout probability is ~0."""
    mdp = build_model(load_map(path))
    pi_star = value_iteration(mdp, tol=1e-11).policy
    result = evaluate(mdp, pi_star, start=mdp.start).at(mdp.start)
    assert result.p_timeout < 1e-6
    assert result.p_success + result.p_fail == pytest.approx(1.0, abs=1e-6)
    V, _, _ = solve_policy(mdp, pi_star)
    assert result.cost == pytest.approx(V[mdp.start], rel=1e-6)


@pytest.mark.parametrize("template", ["dilemma", "field"])
def test_noise_free_optimum_never_enters_a_hazard(template):
    """c_fail makes losing the rover worse than any detour, so with p_slip = 0 nothing fails."""
    for seed in range(5):
        spec = generate(template, 20, seed=seed).with_noise(Noise(p_slip=0.0))
        mdp = build_model(spec)
        solution = value_iteration(mdp)
        result = evaluate(mdp, solution.policy, start=mdp.start).at(mdp.start)
        assert result.p_fail == 0.0
        assert result.p_success == pytest.approx(1.0)
        assert result.cost == pytest.approx(solution.V[mdp.start])


def test_looping_policy_times_out_without_numerical_trouble(hand_spec_deterministic):
    mdp = build_model(hand_spec_deterministic)
    bump = np.full(mdp.n_states, MOVE_N)  # from the start, north is the border: stay and pay
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = evaluate(mdp, bump, horizon=50, start=mdp.start)
    at = result.at(mdp.start)
    assert (at.p_timeout, at.p_success, at.p_fail) == (1.0, 0.0, 0.0)
    assert at.cost == pytest.approx(50 * 3.0)  # move cost 1 + collision 2 each step
    assert result.steps == 50  # a non-terminating policy never stops early


def test_early_stop_matches_the_full_series():
    mdp = build_model(load_map(MAPS[0]))
    policy = value_iteration(mdp).policy
    full = evaluate(mdp, policy, tol=0)
    early = evaluate(mdp, policy, tol=1e-12)
    assert early.steps < full.steps == mdp.horizon
    for name in ("cost", "p_success", "p_fail"):
        np.testing.assert_allclose(getattr(early, name), getattr(full, name), atol=1e-11)


def test_start_restriction_matches_full_evaluation():
    mdp = build_model(load_map(MAPS[0]))
    policy = value_iteration(mdp).policy
    full = evaluate(mdp, policy)
    restricted = evaluate(mdp, policy, start=mdp.start)
    assert restricted.at(mdp.start).cost == pytest.approx(full.at(mdp.start).cost, abs=1e-10)
    assert np.isnan(restricted.cost).sum() > 0 or restricted.cost.size == full.cost.size


def test_probabilities_sum_to_one_for_a_random_policy():
    mdp = build_model(generate("field", 12, seed=3))
    policy = np.random.default_rng(0).integers(0, 8, mdp.n_states)
    ev = evaluate(mdp, policy)
    np.testing.assert_allclose(ev.p_success + ev.p_fail + ev.p_timeout, 1.0, atol=1e-12)
    assert (ev.p_timeout >= 0).all() and (ev.cost > 0).all()


def test_mixture_is_the_weighted_average(hand_spec):
    mdp = build_model(hand_spec)
    pi_a = value_iteration(mdp).policy
    pi_b = np.full(mdp.n_states, 5)  # always dash_E: risky near the hazard
    mix = evaluate_mixture(mdp, ((0.25, pi_a), (0.75, pi_b)))
    a = evaluate(mdp, pi_a, start=mdp.start).at(mdp.start)
    b = evaluate(mdp, pi_b, start=mdp.start).at(mdp.start)
    assert mix.cost == pytest.approx(0.25 * a.cost + 0.75 * b.cost)
    assert mix.p_fail == pytest.approx(0.25 * a.p_fail + 0.75 * b.p_fail)
    with pytest.raises(ValueError, match="sum to 1"):
        evaluate_mixture(mdp, ((0.5, pi_a), (0.2, pi_b)))
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_mixture(mdp, ((1.5, pi_a), (-0.5, pi_b)))


def test_oracle_regret():
    mdp = build_model(load_map(MAPS[0]))
    oracle = Oracle(mdp)
    pi_star = oracle.optimum.policy
    assert oracle.regret(pi_star) == pytest.approx(0.0, abs=1e-8)
    worse = np.random.default_rng(1).integers(0, 8, mdp.n_states)
    assert oracle.regret(worse) > 0
    mix = ((0.4, pi_star), (0.6, worse))
    assert oracle.regret(mix) == pytest.approx(0.6 * oracle.regret(worse), rel=1e-9)
    assert oracle.optimal().cost == pytest.approx(oracle.optimum.V[mdp.start], rel=1e-6)
