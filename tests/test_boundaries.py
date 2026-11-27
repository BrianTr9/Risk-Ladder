"""Boundary value analysis with the single-invalid-input rule.

For every numeric input the four points {just below, lower edge, upper edge, just above} are tried.
Each invalid case changes exactly ONE input of a valid baseline, so a rejection can only be caused
by that input (two invalid values at once would let the first check hide the second).
"""

import numpy as np
import pytest

from riskladder.env import (
    Costs,
    Layout,
    MapError,
    MapSpec,
    Noise,
    RoverEnv,
    build_model,
    generate,
)
from riskladder.env.spec import MAX_HORIZON
from riskladder.oracle import cvar, evaluate, rollout, value_iteration

EPS = 1e-9
LAYOUT = Layout.from_text("#S.G#\n#...#")
PASS, FAIL = "valid", "invalid"


def check(build, expectation):
    if expectation == PASS:
        assert build() is not None
    else:
        with pytest.raises(MapError):
            build()


@pytest.mark.parametrize(
    ("p_slip", "expectation"),
    [(-EPS, FAIL), (0.0, PASS), (1.0, PASS), (1.0 + EPS, FAIL)],
)
def test_slip_probability_boundaries(p_slip, expectation):
    check(lambda: Noise(p_slip=p_slip, dash_slip_mult=1.0), expectation)  # mult 1: p*mult == p


@pytest.mark.parametrize(
    ("mult", "expectation"),
    [(-EPS, FAIL), (0.0, PASS), (2.0, PASS), (2.0 + EPS, FAIL)],  # 0.5 * 2.0 == 1 exactly
)
def test_dash_slip_multiplier_boundaries(mult, expectation):
    check(lambda: Noise(p_slip=0.5, dash_slip_mult=mult), expectation)


@pytest.mark.parametrize("name", ["move", "dash"])
@pytest.mark.parametrize(("value", "expectation"), [(-EPS, FAIL), (0.0, FAIL), (EPS, PASS)])
def test_step_cost_boundaries(name, value, expectation):
    check(lambda: Costs(**{name: value}), expectation)


@pytest.mark.parametrize("name", ["rough_extra", "collision"])
@pytest.mark.parametrize(("value", "expectation"), [(-EPS, FAIL), (0.0, PASS), (EPS, PASS)])
def test_extra_cost_boundaries(name, value, expectation):
    check(lambda: Costs(**{name: value}), expectation)


@pytest.mark.parametrize(("value", "expectation"), [(0.0, FAIL), (EPS, PASS)])
def test_fail_multiplier_boundaries(value, expectation):
    check(lambda: MapSpec(LAYOUT, fail_multiplier=value), expectation)


@pytest.mark.parametrize(("value", "expectation"), [(-EPS, FAIL), (0.0, PASS), (EPS, PASS)])
def test_failure_cost_override_boundaries(value, expectation):
    check(lambda: MapSpec(LAYOUT, c_fail_override=value), expectation)


@pytest.mark.parametrize(
    ("horizon", "expectation"),
    [(0, FAIL), (1, PASS), (MAX_HORIZON, PASS), (MAX_HORIZON + 1, FAIL)],
)
def test_horizon_boundaries(horizon, expectation):
    check(lambda: MapSpec(LAYOUT, horizon_override=horizon), expectation)


@pytest.mark.parametrize(
    ("seed", "expectation"),
    [(-(2**63) - 1, FAIL), (-(2**63), PASS), (2**63 - 1, PASS), (2**63, FAIL)],
)
def test_seed_boundaries(seed, expectation):
    check(lambda: MapSpec(LAYOUT, seed=seed), expectation)


def test_layout_size_boundaries():
    assert Layout.from_text("SG").start == (0, 0)  # smallest map that can be solved
    for text in ("S", "G", "", "SGG\nS"):
        with pytest.raises(MapError):
            Layout.from_text(text)
    with pytest.raises(MapError):
        MapSpec("not a layout")  # type: ignore[arg-type]


@pytest.mark.parametrize(("size", "valid"), [(9, False), (10, True), (11, True)])
def test_dilemma_minimum_size(size, valid):
    if valid:
        assert generate("dilemma", size, seed=0).layout.rows == size
    else:
        with pytest.raises(ValueError, match="at least 10"):
            generate("dilemma", size, seed=0)


def test_start_cell_boundaries_in_the_environment(hand_spec):
    layout = hand_spec.layout
    env = RoverEnv(hand_spec, seed=0)
    last_row, last_col = layout.rows - 1, layout.cols - 1
    for cell in [
        (0, 0),
        (-1, 1),
        (layout.rows, 1),
        (1, -1),
        (1, layout.cols),
        (last_row, last_col),
    ]:
        with pytest.raises(ValueError):  # rock border, outside the grid, or the border corner
            env.reset(start=cell)
    for cell in [(1, 1), (layout.rows - 2, layout.cols - 2)]:  # innermost cells on each side
        assert env.reset(start=cell) == cell
    for action in (-1, 8):
        env.reset()
        with pytest.raises(ValueError):
            env.step(action)
    for action in (0, 7):
        env.reset()
        env.step(action)  # first and last action indices are accepted


def test_horizon_boundaries_in_evaluation_and_rollout(hand_spec):
    mdp = build_model(hand_spec)
    policy = value_iteration(mdp).policy
    for call in (
        lambda: evaluate(mdp, policy, horizon=0),
        lambda: rollout(mdp, policy, 5, horizon=0),
    ):
        with pytest.raises(ValueError, match="at least 1"):
            call()

    one = evaluate(mdp, policy, horizon=1, tol=0)  # one step: the exact one-step quantities
    s = np.arange(mdp.n_states)
    np.testing.assert_allclose(one.cost, mdp.cost[s, policy])
    np.testing.assert_allclose(one.p_success, mdp.p_success[s, policy])
    np.testing.assert_allclose(one.p_fail, mdp.p_fail[s, policy])
    assert one.steps == 1
    assert rollout(mdp, policy, 5, rng=0, horizon=1).steps.max() == 1


def test_cvar_boundaries():
    values = np.arange(1.0, 11.0)  # n = 10
    for alpha in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError):
            cvar(values, alpha)
    assert cvar(values, alpha=0.9, rng=0).value == pytest.approx(10.0)  # tail of exactly one sample
    assert cvar(values, alpha=0.8999, rng=0).value == pytest.approx(9.5)  # two samples
    assert cvar(np.array([7.0]), alpha=0.5, rng=0).value == 7.0  # n = 1 is enough for a tail of 1
    with pytest.raises(ValueError):
        cvar(np.array([]), alpha=0.5)


def test_rollout_sample_size_boundaries(hand_spec):
    mdp = build_model(hand_spec)
    policy = value_iteration(mdp).policy
    assert rollout(mdp, policy, 0, rng=0).n == 0
    assert rollout(mdp, policy, 1, rng=0).n == 1
