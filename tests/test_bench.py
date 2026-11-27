from bench import latency


def test_smallest_size_runs_and_solvers_agree():
    row = latency.run_size(12, pi_max_states=1_000)
    assert row["states"] > 50
    assert row["pi_max_value_gap"] < 1e-6
    assert row["optimal_p_timeout"] < 1e-6
    result = {"machine": latency.machine(), "rows": [row]}
    table = latency.to_markdown(result)
    assert "12x12" in table and "skipped" not in table


def test_partial_runs_go_to_the_smoke_folder():
    full = list(latency.DEFAULT_SIZES)
    assert latency.default_out(full, latency.DEFAULT_PI_MAX_STATES) == latency.RESULTS
    assert latency.default_out([12, 34], latency.DEFAULT_PI_MAX_STATES) == latency.RESULTS / "smoke"
    assert latency.default_out(full, 1_000) == latency.RESULTS / "smoke"
