import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from riskladder.env import FAIL, RUNNING, SUCCESS, Noise, RoverEnv, build_model, generate
from riskladder.oracle import cvar, evaluate_mixture, rollout, value_iteration


def z_score(estimate: float, exact: float, std: float, n: int) -> float:
    return abs(estimate - exact) / max(std / np.sqrt(n), 1e-12)


def test_rollout_matches_exact_evaluation(hand_spec):
    mdp = build_model(hand_spec)
    policy = np.full(mdp.n_states, 5)  # dash_E everywhere: a real chance of failure
    exact = evaluate_mixture(mdp, policy)
    n = 100_000
    roll = rollout(mdp, policy, n, rng=7)
    assert 0.05 < exact.p_fail < 0.95
    assert z_score(roll.cost.mean(), exact.cost, roll.cost.std(), n) < 5
    p = exact.p_fail
    assert z_score(roll.rate(FAIL), p, np.sqrt(p * (1 - p)), n) < 5
    assert roll.rate(SUCCESS) + roll.rate(FAIL) + roll.rate(RUNNING) == pytest.approx(1.0)


def test_rollout_mixture_and_reproducibility(hand_spec):
    mdp = build_model(hand_spec)
    pi_a, pi_b = np.full(mdp.n_states, 1), np.full(mdp.n_states, 5)
    mix = ((0.3, pi_a), (0.7, pi_b))
    exact = evaluate_mixture(mdp, mix)
    n = 100_000
    roll = rollout(mdp, mix, n, rng=3)
    assert z_score(roll.cost.mean(), exact.cost, roll.cost.std(), n) < 5
    again = rollout(mdp, mix, n, rng=3)
    np.testing.assert_array_equal(roll.cost, again.cost)


def test_rollout_uses_the_same_sampling_rule_as_the_env(hand_spec):
    mdp = build_model(hand_spec)
    policy = value_iteration(mdp).policy
    env = RoverEnv(hand_spec, seed=11)
    costs = []
    for _ in range(20_000):
        cell = env.reset()
        total, done = 0.0, False
        while not done:
            result = env.step(int(policy[mdp.index[cell]]))
            total += result.cost
            cell, done = result.cell, result.terminated or result.truncated
        costs.append(total)
    roll = rollout(mdp, policy, 20_000, rng=12)
    assert z_score(np.mean(costs), roll.cost.mean(), roll.cost.std(), 20_000) < 5


def test_rollout_truncates_at_the_horizon(hand_spec_deterministic):
    mdp = build_model(hand_spec_deterministic)
    roll = rollout(mdp, np.full(mdp.n_states, 0), 10, rng=0, horizon=5)
    assert (roll.status == RUNNING).all() and (roll.steps == 5).all()
    np.testing.assert_allclose(roll.cost, 15.0)
    assert roll.success_costs.size == 0


def test_success_costs_exclude_failures(hand_spec):
    mdp = build_model(hand_spec)
    roll = rollout(mdp, np.full(mdp.n_states, 5), 5_000, rng=0)
    assert roll.success_costs.size == np.sum(roll.status == SUCCESS) < roll.n
    # a failure carries c_fail, so it dominates every successful mission cost here
    assert roll.cost[roll.status == FAIL].min() > roll.success_costs.mean()


def test_cvar_known_values_and_interval():
    costs = np.arange(1.0, 101.0)
    result = cvar(costs, alpha=0.9, rng=0)
    assert result.value == pytest.approx(95.5)  # mean of 91..100
    assert result.ci_low <= result.value <= result.ci_high
    assert cvar(costs, alpha=0.5, rng=0).value == pytest.approx(75.5)
    assert cvar(costs, rng=0).value == pytest.approx(np.mean(costs[-5:]))


def test_cvar_validation():
    with pytest.raises(ValueError):
        cvar(np.array([]), 0.9)
    with pytest.raises(ValueError):
        cvar(np.arange(10.0), 1.0)


@settings(max_examples=12, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    p_slip=st.floats(0.0, 0.3),
    template=st.sampled_from(["dilemma", "field"]),
)
def test_exact_evaluation_matches_rollouts_on_random_maps(seed, p_slip, template):
    """Property test: exact results agree with 10^4 simulated episodes within 5 standard errors."""
    spec = generate(template, 10, seed=seed).with_noise(Noise(p_slip=p_slip))
    mdp = build_model(spec)
    policy = np.random.default_rng(seed).integers(0, 8, mdp.n_states)
    exact = evaluate_mixture(mdp, policy)
    n = 10_000
    roll = rollout(mdp, policy, n, rng=seed)
    assert z_score(roll.cost.mean(), exact.cost, roll.cost.std(), n) < 5
    for status, p in ((FAIL, exact.p_fail), (SUCCESS, exact.p_success), (RUNNING, exact.p_timeout)):
        assert z_score(roll.rate(status), p, np.sqrt(max(p * (1 - p), 1e-9)), n) < 5
