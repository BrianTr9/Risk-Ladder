import itertools

import pytest

from riskladder.env import ACTIONS, FAIL, N_ACTIONS, RUNNING, SUCCESS, generate, outcomes
from riskladder.env.layout import PASSABLE, ROCK

MOVE_E, MOVE_N = 1, 0
DASH_E, DASH_N = 5, 4


def as_set(results):
    return {(round(o.prob, 12), o.cell, round(o.cost, 12), o.status) for o in results}


def test_action_table():
    assert N_ACTIONS == 8
    assert [a.name for a in ACTIONS[:4]] == ["move_N", "move_E", "move_S", "move_W"]
    assert all(a.cells == 2 for a in ACTIONS[4:])


def test_move_with_slip_and_collision(hand_spec):
    # move_E from S: 0.8 forward; 0.1 slip south; 0.1 slip north into rock (stay, +collision)
    assert as_set(outcomes(hand_spec, (1, 1), MOVE_E)) == {
        (0.8, (1, 2), 1.0, RUNNING),
        (0.1, (2, 1), 1.0, RUNNING),
        (0.1, (1, 1), 3.0, RUNNING),
    }


def test_dash_into_hazard_rough_and_rock(hand_spec):
    # dash_E from (1,2): passes (1,3), enters hazard (1,4) -> FAIL with c_fail = 30
    # slip south: enters rough (2,2), then hits rock -> stays on (2,2): 1.5 + 1 + 2
    # slip north: rock at once -> stays on (1,2): 1.5 + 2
    assert as_set(outcomes(hand_spec, (1, 2), DASH_E)) == {
        (0.6, (1, 4), 31.5, FAIL),
        (0.2, (2, 2), 4.5, RUNNING),
        (0.2, (1, 2), 3.5, RUNNING),
    }


def test_dash_stops_on_goal(hand_spec):
    results = outcomes(hand_spec, (2, 5), DASH_N)
    assert (0.6, (1, 5), 1.5, SUCCESS) in as_set(results)


def test_identical_outcomes_are_merged(hand_spec_deterministic):
    results = outcomes(hand_spec_deterministic, (1, 1), MOVE_N)
    assert as_set(results) == {(1.0, (1, 1), 3.0, RUNNING)}


@pytest.mark.parametrize("template", ["dilemma", "field"])
def test_outcome_invariants(template):
    spec = generate(template, 12, seed=4)
    layout = spec.layout
    for cell, a in itertools.product(layout.passable_cells, range(N_ACTIONS)):
        results = outcomes(spec, cell, a)
        assert sum(o.prob for o in results) == pytest.approx(1.0)
        for o in results:
            tile = layout.tile(*o.cell)
            assert tile != ROCK
            assert (o.status == RUNNING) == (tile in PASSABLE)
            assert o.cost > 0
