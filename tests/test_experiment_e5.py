import itertools

import numpy as np
import pytest
from experiments import e5

from riskladder.env import load_map


@pytest.fixture(scope="module")
def quick():
    return e5.run(quick=True)


def test_a_star_and_value_iteration_agree_on_the_noise_free_problem():
    for path in e5.MAPS[:3]:
        policy, cost = e5.noise_free_policy(load_map(path))
        assert policy.ndim == 1 and cost > 0  # the assertion inside checks A* against the solver


def test_without_slip_the_a_star_plan_is_optimal_and_never_fails(quick):
    first = quick["curve"][0]
    assert first["p_slip"] == 0.0
    assert first["p_fail"]["max"] == pytest.approx(0.0, abs=1e-12)
    assert first["regret"]["max"] == pytest.approx(0.0, abs=1e-9)


def test_the_failure_probability_of_the_plan_grows_with_the_slip(quick):
    means = [c["p_fail"]["mean"] for c in quick["curve"]]
    assert all(b >= a - 1e-9 for a, b in itertools.pairwise(means))
    assert means[-1] > 0.5  # at a high slip the noise-free plan almost always loses the rover


def test_the_noise_free_plan_never_beats_the_optimum_for_the_true_slip(quick):
    for items in quick["rows"].values():
        assert all(r["regret"] >= -1e-9 for r in items)
        assert all(r["p_fail"] >= 0.0 and r["p_timeout"] >= 0.0 for r in items)


def test_the_optimum_keeps_the_rover_safe_where_the_plan_does_not(quick):
    risky = quick["curve"][-1]
    assert risky["optimal_p_fail"] < 0.1 < risky["p_fail"]["mean"]
    assert np.isfinite([c["regret"]["median"] for c in quick["curve"]]).all()


def test_main_writes_the_table_and_the_figure(tmp_path):
    e5.main(["--quick", "--out", str(tmp_path)])
    for name in ("e5.json", "e5.md", "figures/e5_astar.png"):
        assert (tmp_path / name).stat().st_size > 0
    assert e5.default_out(True) == e5.RESULTS / "smoke" and e5.default_out(False) == e5.RESULTS
