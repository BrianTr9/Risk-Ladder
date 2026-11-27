from pathlib import Path

import numpy as np
import pytest

from riskladder.env import Layout, MapSpec, Noise, build_model, load_map
from riskladder.oracle import (
    is_proper,
    live_mask,
    policy_iteration,
    proper_policy,
    solve_policy,
    value_iteration,
)

MAPS = sorted((Path(__file__).parents[1] / "experiments" / "maps").rglob("*.map"))
MOVE_N, MOVE_E = 0, 1


def corridor(p_slip: float) -> MapSpec:
    return MapSpec(Layout.from_text("#S.G#\n#####"), noise=Noise(p_slip=p_slip))


@pytest.mark.parametrize("solver", [value_iteration, policy_iteration])
def test_hand_computed_corridor(solver):
    """'#S.G#' with p_slip 0.2 (dash slip 0.4), collision cost 2: worked out by hand.

    V(.) = (1 + 2p) / (1 - p) = 1.75 with move_E, which beats dash_E (3.83).
    V(S) = 0.8 (1 + 1.75) + 0.2 (3 + V(S))  ->  3.5, which beats dash_E (3.83).
    """
    mdp = build_model(corridor(0.2))
    solution = solver(mdp)
    np.testing.assert_allclose(solution.V, [3.5, 1.75], atol=1e-8)
    assert list(solution.policy) == [MOVE_E, MOVE_E]
    assert solution.converged


def test_deterministic_corridor_prefers_the_cheaper_dash():
    mdp = build_model(corridor(0.0))
    solution = policy_iteration(mdp)
    # From S: dash_E passes '.' and stops on the goal for 1.5, cheaper than two moves (2.0).
    np.testing.assert_allclose(solution.V, [1.5, 1.0])


@pytest.mark.parametrize("path", MAPS, ids=lambda p: p.stem)
def test_value_and_policy_iteration_agree(path):
    mdp = build_model(load_map(path))
    vi, pi = value_iteration(mdp, tol=1e-11), policy_iteration(mdp)
    assert vi.converged and pi.converged
    np.testing.assert_allclose(vi.V[vi.live], pi.V[pi.live], atol=1e-6)
    assert is_proper(mdp, pi.policy)


def test_proper_policy_is_proper_and_bump_policy_is_not():
    mdp = build_model(load_map(MAPS[0]))
    assert is_proper(mdp, proper_policy(mdp))
    assert not is_proper(mdp, np.full(mdp.n_states, MOVE_N))  # keeps hitting the border
    with pytest.raises(ValueError, match="improper"):
        solve_policy(mdp, np.full(mdp.n_states, MOVE_N))


def test_trapped_pocket_is_excluded_and_solvers_still_converge():
    spec = MapSpec(Layout.from_text("#S.G#\n#####\n#..##\n#####"), noise=Noise(p_slip=0.1))
    mdp = build_model(spec)
    live = live_mask(mdp)
    assert live.tolist() == [True, True, False, False]
    for solution in (value_iteration(mdp), policy_iteration(mdp)):
        assert solution.converged
        assert np.isinf(solution.V[~live]).all() and np.isfinite(solution.V[live]).all()
        assert (solution.policy[~live] == 0).all()
    assert is_proper(mdp, proper_policy(mdp))


@pytest.mark.parametrize("q", [0.3, 0.5001, 0.65])
def test_gamblers_ruin_closed_form(chain, q):
    mdp = chain(n_total=12, q=q)
    policy = np.zeros(mdp.n_states, dtype=int)
    V, p_success, p_fail = solve_policy(mdp, policy)
    np.testing.assert_allclose(p_success, mdp.success_probability(), atol=1e-10)
    np.testing.assert_allclose(p_success + p_fail, 1.0, atol=1e-10)
    np.testing.assert_allclose(V, mdp.expected_duration(), rtol=1e-6)
    assert value_iteration(mdp, tol=1e-12).V[0] == pytest.approx(V[0], rel=1e-6)


def test_policy_iteration_from_a_given_start():
    mdp = build_model(load_map(MAPS[0]))
    reference = policy_iteration(mdp)
    again = policy_iteration(mdp, init=proper_policy(mdp))
    np.testing.assert_allclose(again.V, reference.V, atol=1e-9)


def test_policy_iteration_rejects_an_improper_initial_policy():
    mdp = build_model(load_map(MAPS[0]))
    with pytest.raises(ValueError, match="initial policy"):
        policy_iteration(mdp, init=np.full(mdp.n_states, MOVE_N))


def test_value_iteration_reports_non_convergence():
    mdp = build_model(load_map(MAPS[0]))
    stopped = value_iteration(mdp, max_iter=3)
    assert stopped.iterations == 3 and not stopped.converged and stopped.residual > 1e-9
    assert value_iteration(mdp, max_iter=0).iterations == 0  # no sweep at all is still well-formed


def test_policy_iteration_reports_non_convergence():
    mdp = build_model(load_map(MAPS[0]))
    stopped = policy_iteration(mdp, max_iter=1)  # the proper starting policy is not optimal
    assert stopped.iterations == 1 and not stopped.converged
    assert policy_iteration(mdp).converged


def test_policy_iteration_from_the_optimum_is_idempotent():
    """Started at pi*, one evaluation finds nothing to improve."""
    mdp = build_model(load_map(MAPS[0]))
    optimum = policy_iteration(mdp)
    again = policy_iteration(mdp, init=optimum.policy)
    assert again.iterations == 1 and again.converged
    np.testing.assert_array_equal(again.policy, optimum.policy)
