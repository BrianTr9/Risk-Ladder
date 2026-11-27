"""Property-based tests (round trip, invariant, idempotence, symmetry, differential).

Each property is stated from the specification, so a failing example points at the code under test
and not at the test. Examples are drawn by hypothesis (see the profiles in conftest.py).
"""

import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from riskladder.env import (
    Costs,
    Layout,
    MapSpec,
    Noise,
    build_model,
    format_map,
    generate,
    parse_map,
)
from riskladder.oracle import (
    evaluate,
    evaluate_mixture,
    policy_iteration,
    value_iteration,
)

positive = st.floats(min_value=1e-3, max_value=50.0, allow_nan=False)
non_negative = st.floats(min_value=0.0, max_value=50.0, allow_nan=False)


@st.composite
def specs(draw, sizes=(10, 14)):
    """A valid map with random noise, costs and optional overrides."""
    p_slip = draw(st.floats(min_value=0.0, max_value=0.5))
    mult = draw(st.floats(min_value=0.0, max_value=2.0))  # p_slip * mult <= 1 always holds
    base = generate(
        draw(st.sampled_from(["dilemma", "field"])),
        draw(st.integers(*sizes)),
        seed=draw(st.integers(0, 10_000)),
        noise=Noise(p_slip=p_slip, dash_slip_mult=mult),
        costs=Costs(draw(positive), draw(positive), draw(non_negative), draw(non_negative)),
    )
    return MapSpec(
        base.layout,
        noise=base.noise,
        costs=base.costs,
        fail_multiplier=draw(positive),
        c_fail_override=draw(st.none() | non_negative),
        horizon_override=draw(st.none() | st.integers(1, 5_000)),
        name=draw(st.text(max_size=12)),
        seed=base.seed,
    )


@given(spec=specs())
def test_round_trip_map_file(spec):
    """decode(encode(x)) == x: nothing about a map is lost by writing it to the `.map` format."""
    assert parse_map(format_map(spec)) == spec


@given(spec=specs())
def test_round_trip_layout_text(spec):
    assert Layout.from_text(str(spec.layout)) == spec.layout


@given(spec=specs(sizes=(10, 12)), seed=st.integers(0, 10_000))
def test_evaluation_probabilities_are_a_distribution(spec, seed):
    """Invariant: for any policy the three outcomes partition the episode."""
    mdp = build_model(spec)
    policy = np.random.default_rng(seed).integers(0, 8, mdp.n_states)
    ev = evaluate(mdp, policy, horizon=60, tol=0, start=mdp.start).at(mdp.start)
    for p in (ev.p_success, ev.p_fail, ev.p_timeout):
        assert -1e-12 <= p <= 1 + 1e-12
    assert ev.p_success + ev.p_fail + ev.p_timeout == pytest.approx(1.0, abs=1e-9)
    assert ev.cost >= 0


@given(spec=specs(sizes=(10, 12)), seed=st.integers(0, 10_000))
def test_a_longer_horizon_only_adds_outcomes(spec, seed):
    """Monotonicity: more steps can only add cost and settle more episodes."""
    mdp = build_model(spec)
    policy = np.random.default_rng(seed).integers(0, 8, mdp.n_states)
    results = [
        evaluate(mdp, policy, horizon=h, tol=0, start=mdp.start).at(mdp.start) for h in (5, 20, 80)
    ]
    for short, long in itertools.pairwise(results):
        assert long.cost >= short.cost - 1e-9
        assert long.p_success >= short.p_success - 1e-12
        assert long.p_fail >= short.p_fail - 1e-12
        assert long.p_timeout <= short.p_timeout + 1e-12


@given(spec=specs(sizes=(10, 12)))
def test_optimal_values_satisfy_the_bellman_equation(spec):
    """Invariant taken from the definition, not from another implementation: at the optimum no
    action does better than the value, and the chosen action attains it."""
    mdp = build_model(spec)
    solution = value_iteration(mdp, tol=1e-12)
    V = np.where(solution.live, solution.V, 0.0)
    q = mdp.cost + np.stack([mdp.P[a] @ V for a in range(8)], axis=1)  # (n, A)
    live = solution.live
    np.testing.assert_allclose(q.min(axis=1)[live], solution.V[live], rtol=1e-7, atol=1e-7)
    chosen = q[np.arange(mdp.n_states), solution.policy]
    np.testing.assert_allclose(chosen[live], solution.V[live], rtol=1e-7, atol=1e-7)


@given(spec=specs(sizes=(10, 12)))
def test_value_and_policy_iteration_agree_on_random_maps(spec):
    """Differential test between two algorithms (not two versions of the same code)."""
    mdp = build_model(spec)
    vi, pi = value_iteration(mdp, tol=1e-12), policy_iteration(mdp)
    assert pi.converged
    np.testing.assert_allclose(vi.V[vi.live], pi.V[pi.live], rtol=1e-6, atol=1e-6)
    again = policy_iteration(mdp, init=pi.policy)  # idempotence: the optimum is a fixed point
    assert again.iterations == 1
    np.testing.assert_array_equal(again.policy, pi.policy)


@given(spec=specs(sizes=(10, 12)), weight=st.floats(0.0, 1.0), seed=st.integers(0, 10_000))
def test_mixture_order_does_not_matter(spec, weight, seed):
    """Symmetry: f(a, b) == f(b, a) when the components are listed in the other order."""
    mdp = build_model(spec)
    rng = np.random.default_rng(seed)
    a, b = (rng.integers(0, 8, mdp.n_states) for _ in range(2))
    first = evaluate_mixture(mdp, ((weight, a), (1 - weight, b)), horizon=50)
    second = evaluate_mixture(mdp, ((1 - weight, b), (weight, a)), horizon=50)
    assert first.cost == pytest.approx(second.cost, abs=1e-9)
    assert first.p_fail == pytest.approx(second.p_fail, abs=1e-9)
