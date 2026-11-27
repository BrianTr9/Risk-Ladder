import pytest
from experiments import e0

from riskladder.env import load_map


def test_required_rollouts_matches_the_normal_approximation():
    # n = z^2 (1 - p) / (p * 0.1^2): about 3.5k, 38k and 384k rollouts for 10%, 1%, 0.1%
    assert e0.required_rollouts(0.1) == 3458
    assert e0.required_rollouts(0.01) == 38032
    assert e0.required_rollouts(0.001) == 383776


@pytest.mark.parametrize("target", [0.05, 0.002])
def test_slip_tuning_hits_the_target_failure_probability(target):
    spec = load_map(e0.MAP)
    policy = e0.shortest_route_policy(spec)
    p_slip = e0.tune_slip(spec, policy, target)
    assert e0.failure_probability(spec, policy, p_slip) == pytest.approx(target, rel=1e-6)


def test_quick_run_is_consistent():
    result = e0.run(replications=2, quick=True)
    assert [round(r["p_fail_exact"], 4) for r in result["rows"]] == [0.1, 0.01, 0.001]
    assert all(r["abs_diff_vs_linear_solve"] < 1e-10 for r in result["rows"])
    assert "| 0.1000 |" in e0.to_markdown(result)


def test_quick_runs_never_target_the_committed_results(tmp_path):
    assert e0.default_out(quick=True) == e0.RESULTS / "smoke"
    assert e0.default_out(quick=False) == e0.RESULTS
    e0.main(["--quick", "--replications", "2", "--out", str(tmp_path)])
    assert (tmp_path / "e0.json").exists() and (tmp_path / "e0.md").exists()
