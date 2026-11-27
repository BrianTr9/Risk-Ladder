import itertools

import pytest
from experiments import e3


@pytest.fixture(scope="module")
def quick():
    return e3.run(quick=True)


def test_every_penalty_vertex_is_a_corner_of_the_constrained_frontier(quick):
    """The penalty planner reaches only corners; the constrained solver reaches the same corners
    and every point on the segments between them."""
    cmdp = {round(point["p_fail"], 9): point["cost"] for point in quick["frontier"]["cmdp"]}
    for vertex in quick["frontier"]["penalty"]:
        assert cmdp[round(vertex["p_fail"], 9)] == pytest.approx(vertex["cost"], abs=1e-6)


def test_the_frontier_is_convex_and_falls_as_risk_is_allowed(quick):
    points = sorted(quick["frontier"]["cmdp"], key=lambda p: p["p_fail"])
    costs = [p["cost"] for p in points]
    assert all(b <= a + 1e-9 for a, b in itertools.pairwise(costs))
    slopes = [
        (b["cost"] - a["cost"]) / (b["p_fail"] - a["p_fail"])
        for a, b in itertools.pairwise(points)
        if b["p_fail"] - a["p_fail"] > 1e-9
    ]
    assert all(b >= a - 1e-6 for a, b in itertools.pairwise(slopes))
    assert all(p["gap"] == pytest.approx(0.0, abs=1e-6) for p in points)


def test_the_constrained_solver_holds_the_bound_everywhere_and_the_penalty_does_not(quick):
    spread = quick["spread"]
    assert all(r["cmdp"]["p_fail"] <= spread["bound"] + 1e-9 for r in spread["rows"])
    assert any(r["penalty"]["p_fail"] > spread["bound"] + 1e-9 for r in spread["rows"])


def test_tail_risk_rises_as_the_bound_tightens(quick):
    rows = quick["cvar"]
    assert [r["p_fail"] for r in rows] == sorted((r["p_fail"] for r in rows), reverse=True)
    assert all(low <= r["cvar95_success"] <= high for r in rows for low, high in [r["cvar_ci"]])
    assert rows[-1]["cvar95_success"] > rows[0]["cvar95_success"]


def test_main_writes_the_tables_and_figures_where_asked(tmp_path):
    assert e3.default_out(True) == e3.RESULTS / "smoke" and e3.default_out(False) == e3.RESULTS
    e3.main(["--quick", "--out", str(tmp_path)])
    for name in ("e3.json", "e3.md", "figures/e3_frontier.png", "figures/e3_risk_spread.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_figure_styling_does_not_leak_into_other_plots(tmp_path):
    """Regression: the style used to change matplotlib's global settings, so a plot drawn later in
    the same test run came out different (and a test on its title failed depending on the order)."""
    import matplotlib.pyplot as plt

    before = dict(plt.rcParams)
    e3.main(["--quick", "--out", str(tmp_path)])
    assert dict(plt.rcParams) == before
    _, ax = plt.subplots()
    ax.set_title("later plot")
    assert ax.get_title() == "later plot"  # default (centred) placement, not the figure style's
    plt.close("all")
