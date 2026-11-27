import pytest

from riskladder.env import Layout, MapSpec, Noise

# Small map whose outcomes can be worked out by hand.
#   row 1: S . . X G
#   row 2: . ~ . . .
HAND = """\
#######
#S..XG#
#.~...#
#######"""


@pytest.fixture
def hand_spec() -> MapSpec:
    # move slips with p = 0.2, dash with p = 0.4
    return MapSpec(Layout.from_text(HAND), noise=Noise(p_slip=0.2, dash_slip_mult=2.0))


@pytest.fixture
def hand_spec_deterministic() -> MapSpec:
    return MapSpec(Layout.from_text(HAND), noise=Noise(p_slip=0.0))
