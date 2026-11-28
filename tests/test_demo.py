"""The demo's computations (no web framework) and, when Gradio is installed, its mounting."""

import pytest
from app import demo
from app.main import create_app
from fastapi.testclient import TestClient

from riskladder import service

RIDGE = next(iter(demo.EXAMPLES))
TEXT = demo.EXAMPLES[RIDGE].read_text()


def test_every_example_map_exists_and_parses_with_its_own_slip():
    for name in demo.EXAMPLES:
        text, slip, same = demo.example_view(name)
        assert 0.0 <= slip <= 1.0 and slip == same and text.strip()


@pytest.mark.parametrize("template", sorted(demo.TEMPLATES))
def test_a_generated_map_is_a_valid_map_and_the_same_seed_gives_the_same_map(template):
    first, slip, _ = demo.generated_view(template, 3)
    assert service.load_model(first)[1].n_states > 0 and 0.0 <= slip <= 1.0
    assert demo.generated_view(template, 3)[0] == first
    assert demo.generated_view(template, 4)[0] != first


def test_the_plan_view_reports_the_three_solvers_and_the_bound_holds():
    fig, text = demo.plan_view(TEXT, 0.01, 0.01, 50.0)
    assert len(fig.axes) >= 2
    rows = [line for line in text.splitlines() if line.startswith("| ") and "%" in line]
    assert len(rows) == 3
    constrained = float(rows[2].split("|")[2].strip().rstrip("%"))
    assert constrained <= 1.0 + 1e-6  # the constrained plan keeps the bound it was given
    assert "Optimality gap" in text


def test_the_plan_view_agrees_with_the_service_it_wraps():
    _, text = demo.plan_view(TEXT, 0.01, 0.01, 50.0)
    direct = service.plan(TEXT, delta=0.01, p_slip=0.01).outcome
    assert f"{100 * direct.p_fail:.2f}%" in text and f"{direct.cost:.2f}" in text


def test_a_wrong_slip_belief_breaks_the_bound_and_a_right_one_does_not():
    _, wrong = demo.misspecification_view(TEXT, 0.02, 0.005, 0.01)
    assert "**broken**" in wrong
    _, right = demo.misspecification_view(TEXT, 0.02, 0.02, 0.01)
    assert "**holds**" in right


def test_the_results_page_lists_the_headline_figures_without_image_markup():
    text, figures = demo.results_page()
    assert "![" not in text and "Headline figures" in text
    assert len(figures) == len(demo.HEADLINES)


def test_a_bad_request_becomes_a_sentence_not_a_stack_trace():
    with pytest.raises(service.ServiceError) as caught:
        demo.plan_view("not a map", 0.1, 0.01, 10.0)
    assert demo.explain(caught.value) == caught.value.message and caught.value.status == 422
    assert demo.explain(ValueError("boom")).startswith("Unexpected error")


def test_the_api_works_with_or_without_the_demo_mounted():
    client = TestClient(create_app())
    assert client.get("/health").status_code == 200
    root = client.get("/", follow_redirects=False)
    assert root.status_code in (302, 307) and root.headers["location"] in ("/demo", "/docs")


def test_the_demo_page_is_served_when_gradio_is_installed():
    pytest.importorskip("gradio")
    client = TestClient(create_app())
    page = client.get("/demo/")
    assert page.status_code == 200 and "gradio" in page.text.lower()
    assert client.post("/plan", json={"map": TEXT, "delta": 0.05}).status_code == 200
