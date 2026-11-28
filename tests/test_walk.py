"""`walk_many` is the geometry of motion; `move_many` only prices it."""

import numpy as np
import pytest

from riskladder.env import Costs, Layout, MapSpec
from riskladder.env.dynamics import (
    ACTIONS,
    FAIL,
    PERPENDICULAR,
    RUNNING,
    SUCCESS,
    branch_table,
    move_many,
    walk_many,
)

#   row 1: S ~ # . X G
#   row 2: . . . . . .
GRID = Layout.from_text("########\n#S~#.XG#\n#......#\n########")


def walk(cell, direction, cells):
    w = walk_many(GRID, np.array([cell[0]]), np.array([cell[1]]), direction, cells)
    return (int(w.rows[0]), int(w.cols[0]), int(w.status[0]), int(w.rough[0]), bool(w.collided[0]))


def test_a_move_counts_rough_tiles_it_enters():
    assert walk((1, 1), "E", 1) == (1, 2, RUNNING, 1, False)


def test_a_rock_stops_the_rover_where_it_stands_and_is_a_collision():
    assert walk((1, 2), "E", 1) == (1, 2, RUNNING, 0, True)


def test_a_dash_stops_at_a_rock_in_its_path():
    assert walk((2, 3), "N", 2) == (2, 3, RUNNING, 0, True)  # the rock is right above
    assert walk((1, 1), "E", 2) == (1, 2, RUNNING, 1, True)


def test_a_hazard_ends_the_episode_on_entry_and_nothing_after_it_counts():
    assert walk((1, 4), "E", 2) == (1, 5, FAIL, 0, False)


def test_a_goal_ends_the_episode_on_entry():
    assert walk((1, 5), "E", 1)[2] == SUCCESS


@pytest.mark.parametrize("direction", ["N", "E", "S", "W"])
@pytest.mark.parametrize("cells", [1, 2])
def test_pricing_the_walk_gives_move_many(direction, cells):
    spec = MapSpec(GRID, costs=Costs(move=1.0, dash=1.5, rough_extra=0.7, collision=2.3))
    rows, cols = np.nonzero(GRID.mask(".S~"))
    w = walk_many(GRID, rows, cols, direction, cells)
    r, c, extra, status = move_many(spec, rows, cols, direction, cells)
    assert (r == w.rows).all() and (c == w.cols).all() and (status == w.status).all()
    expected = 2.3 * w.collided + 0.7 * w.rough + spec.c_fail * (w.status == FAIL)
    assert extra == pytest.approx(expected)


def test_the_branch_table_is_each_action_walked_in_its_three_directions():
    table = branch_table(GRID)
    rows, cols = GRID.rows, GRID.cols
    assert table.rows.shape == (rows, cols, len(ACTIONS), 3)
    r, c = (v.ravel() for v in np.indices((rows, cols)))
    for a, action in enumerate(ACTIONS):
        for b, direction in enumerate((action.direction, *PERPENDICULAR[action.direction])):
            walk = walk_many(GRID, r, c, direction, action.cells)
            for name in ("rows", "cols", "status", "rough", "collided"):
                assert (getattr(table, name)[:, :, a, b].ravel() == getattr(walk, name)).all()
