from pathlib import Path

import pytest
from app.main import app, commit_sha
from fastapi.testclient import TestClient

from riskladder import __version__, service

TEXT = (
    Path(__file__).parents[1] / "experiments" / "maps" / "ridge" / "ridge-6x30-s3000.map"
).read_text()


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health_reports_version_commit_and_limits(client, monkeypatch):
    monkeypatch.setenv("COMMIT_SHA", "abc1234")
    body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "version": __version__,
        "commit": "abc1234",
        "max_states": service.MAX_STATES,
        "max_map_chars": service.MAX_MAP_CHARS,
    }


def test_commit_comes_from_the_variable_then_the_file_then_unknown(monkeypatch, tmp_path):
    file = tmp_path / "COMMIT_SHA"
    monkeypatch.delenv("COMMIT_SHA", raising=False)
    assert commit_sha(file) == "unknown"
    file.write_text("deadbeef\n")
    assert commit_sha(file) == "deadbeef"
    monkeypatch.setenv("COMMIT_SHA", "fromenv")
    assert commit_sha(file) == "fromenv"


def test_plan_returns_the_constrained_solution(client):
    response = client.post("/plan", json={"map": TEXT, "delta": 0.01})
    assert response.status_code == 200
    body = response.json()
    assert body["solver"] == "cmdp" and body["feasible"]
    assert body["outcome"]["p_fail"] == pytest.approx(0.01, abs=1e-9)
    assert body["optimality_gap"] == pytest.approx(0.0, abs=1e-6)
    assert body["action_names"][:2] == ["move_N", "move_E"]
    assert len(body["action_names"]) == 8 and body["seconds"] > 0


def test_evaluate_scores_the_policy_that_plan_returned(client):
    plan = client.post("/plan", json={"map": TEXT, "delta": 0.003}).json()
    components = [{"weight": c["weight"], "actions": c["actions"]} for c in plan["components"]]
    scored = client.post("/evaluate", json={"map": TEXT, "components": components})
    assert scored.status_code == 200
    assert scored.json()["outcome"]["cost"] == pytest.approx(plan["outcome"]["cost"], abs=1e-9)


@pytest.mark.parametrize(
    ("path", "payload", "status"),
    [
        ("/plan", {"map": "nonsense"}, 422),
        ("/plan", {"map": TEXT, "delta": 2}, 422),  # rejected by the schema
        ("/plan", {"map": TEXT, "delta": -0.1}, 422),
        ("/plan", {"map": TEXT, "delta": 0.1, "extra_penalty": 3}, 422),  # rejected by the service
        ("/plan", {}, 422),
        ("/plan", {"map": 5}, 422),
        ("/plan", {"map": "x" * (service.MAX_MAP_CHARS + 1)}, 413),
        ("/evaluate", {"map": TEXT, "components": []}, 422),
        ("/evaluate", {"map": TEXT, "components": [{"weight": 1, "actions": [[0]]}]}, 422),
        ("/evaluate", {"map": TEXT, "components": [{"weight": 3, "actions": [[0]]}]}, 422),
    ],
)
def test_bad_requests_get_client_errors_never_a_server_error(client, path, payload, status):
    response = client.post(path, json=payload)
    assert response.status_code == status
    assert "detail" in response.json()


def test_unknown_routes_and_methods(client):
    assert client.get("/nowhere").status_code == 404
    assert client.get("/plan").status_code == 405


def test_the_api_description_documents_the_endpoints(client):
    schema = client.get("/openapi.json").json()
    assert {"/health", "/plan", "/evaluate"} <= set(schema["paths"])
