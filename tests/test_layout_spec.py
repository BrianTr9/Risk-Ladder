import pytest

from riskladder.env import (
    Costs,
    Layout,
    MapError,
    MapSpec,
    Noise,
    format_map,
    load_map,
    parse_map,
    save_map,
)
from riskladder.env.layout import ROCK


def test_layout_basics(hand_spec):
    layout = hand_spec.layout
    assert (layout.rows, layout.cols) == (4, 7)
    assert layout.start == (1, 1)
    assert layout.goals == ((1, 5),)
    assert layout.tile(-1, 0) == ROCK and layout.tile(0, 99) == ROCK
    assert len(layout.passable_cells) == 8
    assert layout.passable_cells[0] == (1, 1)  # row-major order defines the state index


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("#S.G\n#..", "rectangular"),
        ("#S?G#", "unknown tile"),
        ("#..G#", "exactly one start"),
        ("#SS.G#", "exactly one start"),
        ("#S..#", "at least one goal"),
        ("#S.XG#", "no hazard-free route"),
        ("#S#G#", "no hazard-free route"),
    ],
)
def test_invalid_layouts(text, message):
    with pytest.raises(MapError, match=message):
        Layout(tuple(text.split("\n")) if text else ())


def test_noise_and_cost_validation():
    with pytest.raises(MapError):
        Noise(p_slip=-0.1)
    with pytest.raises(MapError):
        Noise(p_slip=0.6, dash_slip_mult=2.0)  # dash slip would be 1.2
    with pytest.raises(MapError):
        Costs(move=0.0)
    assert Noise(p_slip=0.1, dash_slip_mult=2.0).slip(2) == pytest.approx(0.2)


def test_failure_cost_and_horizon(hand_spec):
    # D_max = 6 (from S via the lower row), so c_fail = 5 * 6 * move
    assert hand_spec.c_fail == 30.0
    assert hand_spec.horizon == 10 * (4 + 7)
    custom = MapSpec(hand_spec.layout, c_fail_override=12.0, horizon_override=7)
    assert (custom.c_fail, custom.horizon) == (12.0, 7)


def test_map_format_round_trip(hand_spec):
    spec = MapSpec(
        hand_spec.layout, noise=Noise(0.05, 1.5), name="hand", seed=3, horizon_override=50
    )
    assert parse_map(format_map(spec)) == spec


def test_map_format_errors():
    with pytest.raises(MapError, match="missing"):
        parse_map("{}\n#S.G#")
    with pytest.raises(MapError, match="unknown header keys"):
        parse_map('{"speed": 3}\n---\n#S.G#')


def test_spec_validation(hand_spec):
    layout = hand_spec.layout
    with pytest.raises(MapError):
        Costs(rough_extra=-1.0)
    with pytest.raises(MapError):
        Costs(collision=-0.1)
    with pytest.raises(MapError):
        MapSpec(layout, fail_multiplier=0.0)
    with pytest.raises(MapError):
        MapSpec(layout, c_fail_override=-1.0)
    with pytest.raises(MapError):
        MapSpec(layout, horizon_override=0)


def test_failure_cost_ignores_regions_unreachable_from_start():
    """D_max is taken over cells reachable from S, not over every cell that can reach a goal."""
    layout = Layout.from_text("#S.G###\n#######\n#.....G")
    # S is 2 steps from its goal; the sealed corridor below has a cell 5 steps from its own goal.
    assert MapSpec(layout).c_fail == 5 * 2 * 1.0


def test_map_file_round_trip_with_overrides(tmp_path, hand_spec):
    spec = MapSpec(hand_spec.layout, c_fail_override=12.5, horizon_override=40, name="x", seed=1)
    path = tmp_path / "x.map"
    save_map(spec, path)
    assert load_map(path) == spec
