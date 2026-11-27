import itertools

import pytest
from experiments import e2


@pytest.fixture(scope="module")
def quick():
    return e2.run(quick=True)


def cell(result, believed, true):
    return next(
        c for c in result["grid"]["cells"] if c["believed"] == believed and c["true"] == true
    )


def test_a_correct_belief_costs_nothing_and_breaks_nothing(quick):
    for slip in e2.SLIPS:
        c = cell(quick, slip, slip)
        assert c["mean_violation"] == pytest.approx(0.0, abs=1e-9)
        assert c["mean_extra_cost"] == pytest.approx(0.0, abs=1e-9)


def test_only_an_optimistic_belief_breaks_the_bound(quick):
    """Measured on these maps (not a general theorem): a planner that believes the slip is lower
    than it is breaks the bound, one that believes it is higher never does but pays for it."""
    for believed, true in itertools.product(e2.SLIPS, repeat=2):
        c = cell(quick, believed, true)
        if believed > true:
            assert c["mean_violation"] == pytest.approx(0.0, abs=1e-9)
            assert c["mean_extra_cost"] >= -1e-9
        if believed < true:
            assert c["mean_extra_cost"] is not None  # the cost can be negative: risk is bought


def test_the_violation_grows_with_the_size_of_the_mistake(quick):
    for true in e2.SLIPS:
        column = [cell(quick, b, true)["mean_violation"] for b in e2.SLIPS if b <= true]
        assert column == sorted(column, reverse=True)  # lowest belief breaks the bound the most


def test_a_larger_margin_is_safer_and_costs_more_when_the_belief_was_right(quick):
    rows = quick["margin"]["rows"]
    for factor in map(str, quick["margin"]["underestimates"]):
        violations = [r["violation_by_underestimate"][factor] for r in rows]
        assert violations == sorted(violations, reverse=True)
    extra = [r["right_belief_mean_extra_cost"] for r in rows]
    assert extra[0] == pytest.approx(0.0, abs=1e-9) and extra == sorted(extra)
    assert rows[0]["violation_by_underestimate"]["3.0"] > 0  # an unprotected planner is hurt
    assert rows[-1]["violation_by_underestimate"]["3.0"] == pytest.approx(0.0, abs=1e-9)


def test_main_writes_the_outputs_where_asked(tmp_path):
    assert e2.default_out(True) == e2.RESULTS / "smoke" and e2.default_out(False) == e2.RESULTS
    e2.main(["--quick", "--out", str(tmp_path)])
    for name in ("e2.json", "e2.md", "figures/e2_misspecification.png"):
        assert (tmp_path / name).stat().st_size > 0
