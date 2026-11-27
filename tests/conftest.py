import numpy as np
import pytest
from hypothesis import settings

from riskladder.env import Layout, MapSpec, Noise

# Property tests: a small budget by default; `pytest --hypothesis-profile=thorough` digs deeper.
settings.register_profile("default", max_examples=12, deadline=None)
settings.register_profile("thorough", max_examples=200, deadline=None)
settings.load_profile("default")

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


class ChainMDP:
    """Random walk on states 1..N-1 with absorbing FAIL at 0 and SUCCESS at N (one action).

    From each state the walk moves up with probability `q` and down with `1 - q`; each step costs 1.
    Closed forms exist for the success probability and the expected duration (gambler's ruin).
    """

    n_actions = 1

    def __init__(self, n_total: int, q: float, horizon: int = 100_000) -> None:
        from scipy import sparse

        n = n_total - 1
        self.n_states, self.start, self.horizon = n, 0, horizon
        up = sparse.diags([np.full(n - 1, q)], [1], shape=(n, n))
        down = sparse.diags([np.full(n - 1, 1 - q)], [-1], shape=(n, n))
        self.P_stack = (up + down).tocsr()
        self.p_success = np.zeros((n, 1))
        self.p_fail = np.zeros((n, 1))
        self.p_success[-1, 0] = q
        self.p_fail[0, 0] = 1 - q
        self.cost = np.ones((n, 1))
        self.n_total, self.q = n_total, q

    def success_probability(self) -> np.ndarray:
        r = (1 - self.q) / self.q
        i = np.arange(1, self.n_total)
        return (1 - r**i) / (1 - r**self.n_total)

    def expected_duration(self) -> np.ndarray:
        i = np.arange(1, self.n_total)
        return i / (1 - 2 * self.q) - self.n_total / (1 - 2 * self.q) * self.success_probability()


@pytest.fixture
def chain():
    return ChainMDP
