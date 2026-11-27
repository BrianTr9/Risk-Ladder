from pathlib import Path

import numpy as np
import pytest

from riskladder import service
from riskladder.env import build_model, format_map, generate, load_map
from riskladder.oracle import evaluate_mixture
from riskladder.service import ServiceError

MAP_FILE = Path(__file__).parents[1] / "experiments" / "maps" / "ridge" / "ridge-6x30-s3000.map"
TEXT = MAP_FILE.read_text()
GOLDEN_COST = 29.587684696664425  # constrained optimum on this map for a 1% bound


def test_constrained_plan_meets_the_bound_exactly_and_is_certified_optimal():
    result = service.plan(TEXT, delta=0.01)
    assert result.solver == "cmdp" and result.feasible
    assert result.outcome.p_fail == pytest.approx(0.01, abs=1e-9)
    assert result.outcome.cost == pytest.approx(GOLDEN_COST, abs=1e-6)
    assert result.optimality_gap == pytest.approx(0.0, abs=1e-6)
    assert sum(c.weight for c in result.components) == pytest.approx(1.0)
    assert result.states == len(load_map(MAP_FILE).layout.passable_cells)


def test_a_loose_bound_equals_the_environment_optimum_and_a_penalty_lowers_risk():
    environment = service.plan(TEXT)
    assert environment.solver == "environment" and environment.multiplier == 0.0
    assert service.plan(TEXT, delta=1.0).outcome == environment.outcome
    penalised = service.plan(TEXT, extra_penalty=500.0)
    assert penalised.solver == "penalty"
    assert penalised.outcome.p_fail < environment.outcome.p_fail
    assert penalised.outcome.cost >= environment.outcome.cost


def test_returned_policy_grids_reproduce_the_outcome_when_evaluated():
    """Round trip: what /plan returns, /evaluate scores to the same exact numbers."""
    plan = service.plan(TEXT, delta=0.003)
    outcome, seconds = service.evaluate(TEXT, [(c.weight, c.actions) for c in plan.components])
    assert outcome.cost == pytest.approx(plan.outcome.cost, abs=1e-9)
    assert outcome.p_fail == pytest.approx(plan.outcome.p_fail, abs=1e-12)
    assert seconds >= 0


def test_policy_grid_marks_only_decision_cells():
    plan = service.plan(TEXT)
    grid = np.array(plan.components[0].actions)
    spec = load_map(MAP_FILE)
    decisions = np.zeros(grid.shape, dtype=bool)
    for r, c in spec.layout.passable_cells:
        decisions[r, c] = True
    assert (grid[~decisions] == service.NO_ACTION).all()
    assert ((grid[decisions] >= 0) & (grid[decisions] < 8)).all()


def test_slip_override_changes_the_outcome():
    base = service.plan(TEXT)
    noisier = service.plan(TEXT, p_slip=0.03)
    assert noisier.outcome.cost > base.outcome.cost
    explicit = format_map(load_map(MAP_FILE).with_noise(type(load_map(MAP_FILE).noise)(0.03, 2.0)))
    assert service.plan(explicit).outcome.cost == pytest.approx(noisier.outcome.cost)


def test_an_unreachable_bound_is_reported_not_faked():
    trap = "{}\n---\n#XXX#\n#S.G#\n#XXX#"
    result = service.plan(trap, delta=0.0)
    assert not result.feasible and result.optimality_gap is None


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"text": "nonsense"}, 422),
        ({"text": TEXT, "delta": 0.1, "extra_penalty": 5.0}, 422),
        ({"text": TEXT, "delta": 1.5}, 422),
        ({"text": TEXT, "p_slip": 2.0}, 422),
        ({"text": "x" * (service.MAX_MAP_CHARS + 1)}, 413),
        ({"text": 123}, 413),
    ],
)
def test_bad_plan_requests_are_service_errors_with_a_status(kwargs, status):
    with pytest.raises(ServiceError) as caught:
        service.plan(**kwargs)
    assert caught.value.status == status and caught.value.message


def test_oversized_maps_are_rejected_by_state_count(monkeypatch):
    monkeypatch.setattr(service, "MAX_STATES", 10)
    with pytest.raises(ServiceError) as caught:
        service.plan(TEXT)
    assert caught.value.status == 413 and "decision cells" in caught.value.message
    big = format_map(generate("field", 12, seed=0))
    monkeypatch.setattr(service, "MAX_STATES", 5)
    with pytest.raises(ServiceError):
        service.evaluate(big, [(1.0, [[0] * 12] * 12)])


def valid_grid():
    return service.plan(TEXT).components[0].actions


@pytest.mark.parametrize(
    "mutate",
    [
        lambda g: g[:-1],  # one row short
        lambda g: [row[:-1] for row in g],  # one column short
        lambda g: [[8 if v >= 0 else v for v in row] for row in g],  # action out of range
        lambda g: [[-1 for _ in row] for row in g],  # decision cells left empty
        lambda g: [[True if v >= 0 else v for v in row] for row in g],  # booleans are not actions
        lambda g: [[0.5 if v >= 0 else v for v in row] for row in g],  # not integers
    ],
    ids=["rows", "columns", "range", "empty", "bool", "float"],
)
def test_bad_policy_grids_are_rejected(mutate):
    with pytest.raises(ServiceError) as caught:
        service.evaluate(TEXT, [(1.0, mutate(valid_grid()))])
    assert caught.value.status == 422


def test_bad_mixtures_are_rejected():
    grid = valid_grid()
    for components in ([], [(0.5, grid)], [(0.7, grid), (0.7, grid)], [(-0.5, grid), (1.5, grid)]):
        with pytest.raises(ServiceError) as caught:
            service.evaluate(TEXT, components)
        assert caught.value.status == 422


def test_service_agrees_with_calling_the_library_directly():
    mdp = build_model(load_map(MAP_FILE))
    from riskladder.agents import solve_cmdp

    direct = evaluate_mixture(mdp, solve_cmdp(mdp, 0.005).mixture)
    assert service.plan(TEXT, delta=0.005).outcome.cost == pytest.approx(direct.cost, abs=1e-9)
