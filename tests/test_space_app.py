"""The Gradio Space entry: it builds, answers `/plan` exactly, and its pins agree with the lock."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
from scripts import deploy_space, smoke_api, smoke_space

ROOT = Path(__file__).parents[1]
TEXT = smoke_api.MAP_FILE.read_text()


@pytest.fixture(scope="module")
def entry():
    spec = importlib.util.spec_from_file_location("space_app", ROOT / "space" / "space_app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["space_app"] = module
    spec.loader.exec_module(module)
    return module


def test_the_entry_builds_the_demo_and_the_plan_endpoint(entry):
    import gradio as gr

    assert isinstance(entry.demo, gr.Blocks)
    names = {fn.api_name for fn in entry.demo.fns.values()}
    assert "plan" in names


def test_the_plan_endpoint_returns_the_exact_constrained_optimum(entry):
    answer = entry.plan_api(TEXT, smoke_api.GOLDEN_DELTA)
    assert answer["outcome"]["p_fail"] == pytest.approx(smoke_api.GOLDEN_DELTA, abs=1e-9)
    assert answer["outcome"]["cost"] == pytest.approx(smoke_api.GOLDEN_COST, abs=1e-6)
    assert answer["optimality_gap"] == pytest.approx(0.0, abs=1e-6) and answer["feasible"]
    assert sum(c["weight"] for c in answer["components"]) == pytest.approx(1.0)
    assert len(answer["action_names"]) == 8


def test_a_slip_override_changes_the_plan_and_a_negative_one_keeps_the_maps_own(entry):
    own = entry.plan_api(TEXT, 0.01, -1.0)["outcome"]["cost"]
    assert entry.plan_api(TEXT, 0.01)["outcome"]["cost"] == own
    assert entry.plan_api(TEXT, 0.01, 0.03)["outcome"]["cost"] != own


def test_a_bad_map_is_a_service_error_not_a_crash(entry):
    from riskladder.service import ServiceError

    with pytest.raises(ServiceError):
        entry.plan_api("not a map", 0.01)


def test_the_gpu_placeholder_exists_for_zerogpu_and_is_harmless_elsewhere(entry):
    assert callable(entry.gpu_placeholder) and entry.gpu_placeholder() is None


def test_requirements_readme_and_lock_pin_the_same_gradio():
    pins = dict(
        line.split("==")
        for line in (ROOT / "space" / "requirements.txt").read_text().split()
        if "==" in line
    )
    readme = (ROOT / "space" / "README.md").read_text()
    sdk = re.search(r"^sdk_version: (\S+)", readme, re.M).group(1)
    lock = re.search(r'name = "gradio"\nversion = "([^"]+)"', (ROOT / "uv.lock").read_text())
    assert pins["gradio"] == sdk == lock.group(1)
    assert re.search(r"^sdk: gradio$", readme, re.M)
    assert re.search(r"^app_file: space_app\.py$", readme, re.M)


def test_the_space_imports_nothing_it_does_not_install():
    """Every third-party module the demo and the entry import must be in the Space requirements."""
    needed = {"numpy", "scipy", "matplotlib", "gradio", "spaces"}
    declared = {
        re.split("[=<>]", line)[0]
        for line in (ROOT / "space" / "requirements.txt").read_text().split()
    }
    assert needed <= declared
    for path in (ROOT / "app" / "ui.py", ROOT / "app" / "demo.py", ROOT / "space" / "space_app.py"):
        for module in re.findall(r"^(?:from|import) (\w+)", path.read_text(), re.M):
            assert module not in {"fastapi", "uvicorn", "pydantic", "httpx"}, (path.name, module)


@pytest.mark.slow
def test_the_space_smoke_test_passes_against_the_demo_served_locally(entry):
    _, url, _ = entry.demo.launch(prevent_thread_lock=True, server_port=7899, quiet=True)
    try:
        assert smoke_space.run(url.rstrip("/"), calls=2, budget=60.0) == 0
    finally:
        entry.demo.close()


def test_a_staged_space_runs_from_its_own_files(tmp_path):
    """The Space has no `pyproject`, no tests and no editable install: run the entry from the staged
    folder and check that the planner it imports is the staged copy."""
    deploy_space.stage(tmp_path, "abc")
    code = (
        "import runpy; "
        "ns = runpy.run_path('space_app.py', run_name='not_main'); "
        "import riskladder; "
        "text = open('experiments/maps/ridge/ridge-6x30-s3000.map').read(); "
        "print(riskladder.__file__); print(ns['plan_api'](text, 0.01)['outcome']['p_fail'])"
    )
    run = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=300
    )
    assert run.returncode == 0, run.stderr[-800:]
    *_, imported, p_fail = run.stdout.strip().splitlines()
    assert Path(imported).resolve().is_relative_to(tmp_path.resolve())  # the staged copy
    assert float(p_fail) == pytest.approx(0.01, abs=1e-9)
