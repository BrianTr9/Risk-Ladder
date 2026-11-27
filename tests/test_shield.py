"""The shield: what it blocks, and that it never blocks every way to the goal."""

import numpy as np
import pytest

from riskladder.env import Layout, generate
from riskladder.env.distances import steps_to_goal
from riskladder.env.dynamics import ACTIONS, walk_many
from riskladder.safety.shield import Shield

#   row 1: S . . X G
#   row 2: . ~ . . .
HAND = Layout.from_text("#######\n#S..XG#\n#.~...#\n#######")
ACTION = {a.name: i for i, a in enumerate(ACTIONS)}


def route_exists(layout, allowed):
    """From every decision cell, can the rover reach the goal using only allowed actions when
    every action goes as intended? (Backwards search over the allowed graph.)"""
    # cells with no hazard-free way to the goal (sealed pockets) cannot reach it by any policy
    dist = steps_to_goal(layout)
    cells = [tuple(c) for c in np.argwhere(layout.mask(".S~")) if np.isfinite(dist[tuple(c)])]
    reach = set()
    edges = []
    for r, c in cells:
        for a, act in enumerate(ACTIONS):
            if not allowed[r, c, a]:
                continue
            w = walk_many(layout, np.array([r]), np.array([c]), act.direction, act.cells)
            if w.status[0] == 1:
                reach.add((r, c))
            elif w.status[0] == 0:
                edges.append(((r, c), (int(w.rows[0]), int(w.cols[0]))))
    changed = True
    while changed:
        changed = False
        for src, dst in edges:
            if dst in reach and src not in reach:
                reach.add(src)
                changed = True
    return all(cell in reach for cell in cells), len(cells)


def test_risk_is_the_worst_case_one_step_failure_probability():
    risk = Shield(p_max=0.4).risk(HAND)
    assert risk[1, 3, ACTION["move_E"]] == pytest.approx(0.6)  # straight into the hazard
    assert risk[1, 2, ACTION["dash_E"]] == pytest.approx(0.6)  # the dash ends on it
    assert risk[2, 4, ACTION["move_E"]] == pytest.approx(0.2)  # only a slip north reaches it
    assert risk[2, 1, ACTION["move_E"]] == 0.0
    assert Shield(p_max=0.0).risk(HAND)[2, 4, ACTION["move_E"]] == 0.0


#   a corridor next to a hazard strip, and a longer detour that keeps clear of it
DETOUR = Layout.from_text("#########\n#S.....G#\n#..XXX..#\n#.......#\n#.......#\n#########")


def test_actions_that_can_enter_a_hazard_are_blocked_and_safe_ones_are_not():
    table = Shield(p_max=0.4, theta=0.05).analyse(DETOUR)
    assert not table.allowed[1, 2, ACTION["move_E"]]  # stepping into the corridor is blocked
    assert not table.allowed[1, 2, ACTION["dash_E"]]
    assert table.allowed[1, 1, ACTION["move_S"]] and table.allowed[4, 3, ACTION["move_E"]]
    assert not table.relaxed[1, 2] and not table.relaxed[4, 3]  # the detour needs no risk
    assert table.relaxed[1, 3:6].all()  # but a rover already above the strip has no safe exit


def test_a_lower_slip_bound_lets_the_rover_use_the_corridor():
    assert not Shield(p_max=0.4, theta=0.05).table(DETOUR)[1, 2, ACTION["move_E"]]
    assert Shield(p_max=0.05, theta=0.05).table(DETOUR)[1, 2, ACTION["move_E"]]  # risk 0.025


def test_no_threshold_ever_allows_walking_straight_into_a_hazard():
    allowed = Shield(p_max=0.4, theta=1.0).table(HAND)
    assert not allowed[1, 3, ACTION["move_E"]] and not allowed[1, 2, ACTION["dash_E"]]
    assert allowed[1, 3, ACTION["move_W"]]


def test_cells_that_are_not_decision_cells_allow_everything():
    allowed = Shield().table(HAND)
    assert allowed[0].all() and allowed[1, 4].all() and allowed[1, 5].all()


def test_larger_thresholds_only_allow_more():
    layout = generate("dilemma", seed=0, rows=20).layout
    previous = None
    for theta in (0.0, 0.01, 0.05, 0.3, 0.7, 1.0):
        allowed = Shield(theta=theta).table(layout)
        if previous is not None:
            assert (allowed >= previous).all()
        previous = allowed


@pytest.mark.parametrize("template", ["dilemma", "field", "ridge"])
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("shield", [Shield(0.4, 0.0), Shield(0.4, 0.05), Shield(0.1, 0.01)])
def test_a_route_to_the_goal_and_an_action_always_remain(template, seed, shield):
    layout = generate(template, seed=seed, rows=16 if template != "ridge" else 6,
                      cols=None if template != "ridge" else 24).layout  # fmt: skip
    table = shield.analyse(layout)
    decide = layout.mask(".S~")
    assert table.allowed[decide].any(axis=1).all()  # at least one action everywhere
    assert route_exists(layout, table.allowed)[0]


def test_the_threshold_is_raised_only_where_a_route_needs_it_and_that_is_reported():
    #   the only way to the goal passes beside the hazard
    layout = Layout.from_text("#######\n#S...G#\n#XXXXX#\n#######")
    table = Shield(p_max=0.4, theta=0.05).analyse(layout)
    assert table.relaxed[1, 1:5].all()
    assert route_exists(layout, table.allowed)[0]
    assert table.allowed[1, 1, ACTION["move_E"]]  # the corridor is open despite its risk
    assert not table.allowed[1, 1, ACTION["move_S"]]  # but stepping into the hazard is not


def test_a_cell_whose_every_exit_is_risky_is_relaxed_and_nothing_else_is():
    relaxed = Shield().analyse(HAND).relaxed  # (2,4) sits under the hazard next to the goal
    assert relaxed[2, 4] and relaxed.sum() == 1


def test_a_cell_cut_off_from_the_goal_keeps_its_least_risky_actions():
    layout = Layout.from_text("########\n#S..G#.#\n########")  # (1,6) is sealed off
    table = Shield().analyse(layout)
    assert table.relaxed[1, 6] and table.allowed[1, 6].any() and not table.relaxed[1, 1]


@pytest.mark.parametrize("bad", [{"p_max": -0.1}, {"p_max": 1.1}, {"theta": -1.0}, {"theta": 2.0}])
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        Shield(**bad)


def test_config_round_trips_through_the_constructor():
    shield = Shield(p_max=0.3, theta=0.02)
    assert Shield(**shield.config) == shield
