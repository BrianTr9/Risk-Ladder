"""Features of a (cell, action): map-only, bounded in number, and correct on hand cases."""

import tokenize
from pathlib import Path

import numpy as np
import pytest

from riskladder.agents.features import FEATURES, N_FEATURES, feature_table
from riskladder.env import Layout
from riskladder.env.dynamics import ACTIONS

#   row 1: S . . X G
#   row 2: . ~ . . .
HAND = Layout.from_text("#######\n#S..XG#\n#.~...#\n#######")
F = {name: i for i, name in enumerate(FEATURES)}
ACTION = {a.name: i for i, a in enumerate(ACTIONS)}


@pytest.fixture(scope="module")
def phi():
    return feature_table(HAND)


def test_there_are_at_most_ten_features(phi):
    assert N_FEATURES <= 10
    assert phi.shape == (4, 7, len(ACTIONS), N_FEATURES)


def test_cells_where_the_rover_never_decides_have_no_features(phi):
    assert not phi[0].any() and not phi[:, 0].any()  # the rock border
    assert not phi[1, 4].any() and not phi[1, 5].any()  # hazard and goal tiles


def test_the_bias_is_one_wherever_the_rover_decides(phi):
    assert phi[1, 1, :, F["bias"]].tolist() == [1.0] * len(ACTIONS)


def test_walking_into_the_hazard_is_flagged_and_slips_into_it_are_counted(phi):
    assert phi[1, 3, ACTION["move_E"], F["intended_hazard"]] == 1.0
    assert phi[1, 2, ACTION["dash_E"], F["intended_hazard"]] == 1.0  # the dash lands on the hazard
    assert phi[1, 3, ACTION["move_E"], F["slip_hazard"]] == 0.0  # slips go to rows 0 and 2
    # under the hazard at (2,4): going east or west can slip north into it, going south cannot
    assert phi[2, 4, ACTION["move_E"], F["slip_hazard"]] == 0.5
    assert phi[2, 4, ACTION["move_W"], F["slip_hazard"]] == 0.5
    assert phi[2, 4, ACTION["move_S"], F["slip_hazard"]] == 0.0
    assert phi[2, 4, ACTION["move_N"], F["intended_hazard"]] == 1.0


def test_goal_distance_falls_by_one_step_per_move_along_the_hazard_free_route(phi):
    after = lambda a: phi[2, 2, ACTION[a], F["goal_distance_after"]]  # noqa: E731
    assert after("move_E") < after("move_W")
    assert after("dash_E") < after("move_E")


def test_dash_and_rough_and_collision_flags(phi):
    assert phi[1, 1, ACTION["dash_E"], F["dash"]] == 1.0
    assert phi[1, 1, ACTION["move_E"], F["dash"]] == 0.0
    assert phi[1, 2, ACTION["move_S"], F["rough_tiles"]] == 1.0  # (2,2) is rough
    assert phi[1, 1, ACTION["move_S"], F["rough_tiles"]] == 0.0
    assert phi[1, 1, ACTION["move_N"], F["collision"]] == 1.0
    assert phi[1, 1, ACTION["move_E"], F["collision"]] == 0.0


def test_routes_ahead_of_a_cell_next_to_the_hazard_carry_exposure(phi):
    near = phi[1, 2, ACTION["move_S"], F["route_exposure_after"]]
    far = phi[2, 1, ACTION["move_N"], F["route_exposure_after"]]
    assert near >= 0 and far >= 0
    assert phi[2, 3, ACTION["move_N"], F["route_exposure_after"]] > 0  # lands beside the hazard


def test_features_are_scaled_for_a_stable_update(phi):
    assert np.isfinite(phi).all() and phi.min() >= 0.0 and phi.max() <= 3.0


def test_the_learners_never_touch_noise_costs_or_the_exact_model():
    """Interface contract (docs/env_spec.md section 8), checked on the code of the learners:
    comments and strings are dropped, what is left must not name any hidden quantity."""
    banned = {"noise", "p_slip", "build_model", "model", "Costs", "MapSpec", "c_fail", "costs"}
    for name in ("features.py", "tabular.py", "linear.py"):
        path = Path(__file__).parents[1] / "src" / "riskladder" / "agents" / name
        with path.open("rb") as handle:
            names = {
                t.string for t in tokenize.tokenize(handle.readline) if t.type == tokenize.NAME
            }
        assert not names & banned, (name, names & banned)
