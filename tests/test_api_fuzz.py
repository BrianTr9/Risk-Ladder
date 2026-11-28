"""Random request bodies: the API answers with a clean status and never with a server error."""

from pathlib import Path

import pytest
from app.main import app
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

TINY = "#######\n#S..XG#\n#.~...#\n#######"
RIDGE = (
    Path(__file__).parents[1] / "experiments" / "maps" / "ridge" / "ridge-6x30-s3000.map"
).read_text()
HEADER = RIDGE.split("---")[0]

client = TestClient(app)

maps = st.one_of(
    st.just(TINY),
    st.just(HEADER + "---\n" + TINY),
    st.just(RIDGE[: len(RIDGE) // 2]),  # truncated
    st.text(max_size=300),
    st.sampled_from(["", "{}", "---", "#", "S", "{}\n---\nS", RIDGE.replace("G", "S", 1)]),
)
numbers = st.one_of(
    st.none(), st.floats(allow_nan=False, allow_infinity=False, min_value=-5, max_value=5),
    st.integers(-5, 5), st.just("0.1"), st.just(True),
)  # fmt: skip
grids = st.lists(st.lists(st.integers(-2, 9) | st.none(), max_size=8), max_size=6)


def acceptable(response) -> bool:
    return response.status_code in (200, 413, 422)


@settings(max_examples=60, deadline=None)
@given(map_text=maps, delta=numbers, penalty=numbers, p_slip=numbers)
def test_plan_never_fails_with_a_server_error(map_text, delta, penalty, p_slip):
    response = client.post(
        "/plan",
        json={"map": map_text, "delta": delta, "extra_penalty": penalty, "p_slip": p_slip},
    )
    assert acceptable(response), (response.status_code, response.text[:200])
    if response.status_code == 200:
        outcome = response.json()["outcome"]
        assert -1e-9 <= outcome["p_fail"] <= 1 + 1e-9 and outcome["cost"] >= 0


@settings(max_examples=60, deadline=None)
@given(
    map_text=maps,
    weight=st.floats(allow_nan=False, allow_infinity=False, min_value=-1, max_value=2),
    grid=grids,
    p_slip=numbers,
)
def test_evaluate_never_fails_with_a_server_error(map_text, weight, grid, p_slip):
    body = {"map": map_text, "components": [{"weight": weight, "actions": grid}], "p_slip": p_slip}
    response = client.post("/evaluate", json=body)
    assert acceptable(response), (response.status_code, response.text[:200])


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        "text",
        7,
        {"map": 5},
        {"map": TINY, "components": "x"},
        {"map": TINY, "delta": [1]},
    ],
)
@pytest.mark.parametrize("path", ["/plan", "/evaluate"])
def test_malformed_json_bodies_are_rejected_cleanly(path, body):
    assert client.post(path, json=body).status_code == 422
