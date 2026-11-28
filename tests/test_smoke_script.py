"""The smoke script and the deploy staging, tested against a real HTTP server."""

import socket
import threading
import time

import pytest
import uvicorn
from app.main import app
from scripts import deploy_space, smoke_api

pytestmark = pytest.mark.slow


def test_percentile_uses_nearest_rank():
    # a mean of 825 ms looks fine while one request in ten took 1.2 s or more and one took 5 s
    times = [180, 190, 210, 220, 240, 260, 300, 450, 1200, 5000]
    assert sum(times) / len(times) == 825
    assert smoke_api.percentile(times, 50) == 240  # rank ceil(0.5 * 10) = 5
    assert smoke_api.percentile(times, 95) == 5000  # rank ceil(0.95 * 10) = 10
    assert smoke_api.percentile([7], 99) == 7


@pytest.fixture(scope="module")
def server():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    for _ in range(100):
        if instance.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    instance.should_exit = True
    thread.join(timeout=5)


def test_smoke_script_passes_against_a_live_server(server, capsys):
    assert smoke_api.main(["--url", server, "--calls", "5"]) == 0
    assert "PASS" in capsys.readouterr().out


def test_smoke_script_fails_when_the_golden_value_is_wrong(server, monkeypatch, capsys):
    monkeypatch.setattr(smoke_api, "GOLDEN_COST", 1.0)
    assert smoke_api.main(["--url", server, "--calls", "2"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_smoke_script_fails_on_a_wrong_commit_and_a_tight_budget(server, capsys):
    assert smoke_api.main(["--url", server, "--calls", "2", "--commit", "0000000"]) == 1
    assert smoke_api.main(["--url", server, "--calls", "2", "--budget", "0.000001"]) == 1


def test_smoke_script_reports_an_unreachable_server_as_an_error():
    with pytest.raises(OSError):
        smoke_api.main(["--url", "http://127.0.0.1:9", "--calls", "1"])


def test_stale_template_files_are_never_part_of_the_upload(tmp_path):
    names = deploy_space.stage(tmp_path, "abc")
    assert not set(deploy_space.STALE) & set(names)  # they are deleted from the Space, not sent


def test_deploy_stages_only_what_the_space_needs(tmp_path):
    names = deploy_space.stage(tmp_path, "abc1234def")
    assert {"space_app.py", "requirements.txt", "README.md", "LICENSE", "COMMIT_SHA"} <= set(names)
    assert "app/ui.py" in names and "src/riskladder/service.py" in names
    assert "results/SUMMARY.md" in names and any(n.startswith("experiments/maps/") for n in names)
    assert not any(n.startswith(("tests", "docs", "tools", "bench")) for n in names)
    code = [
        n for n in names if n.startswith("experiments") and not n.startswith("experiments/maps")
    ]
    assert not code
    assert not any(n.startswith(("results/raw", "results/smoke")) for n in names)
    assert "Dockerfile" not in names  # a Gradio Space is built from its requirements
    assert not any("__pycache__" in n for n in names)
    assert (tmp_path / "COMMIT_SHA").read_text().strip() == "abc1234def"
    assert (tmp_path / "README.md").read_text().startswith("---\ntitle:")  # the Space's own README


@pytest.mark.parametrize(
    ("space", "url"),
    [
        ("BrianTr9/risk-ladder", "https://briantr9-risk-ladder.hf.space"),
        ("Owner/My_Space.v2", "https://owner-my-space-v2.hf.space"),
    ],
)
def test_space_url(space, url):
    assert deploy_space.space_url(space) == url


def test_smoke_script_can_skip_the_demo_page_and_fails_when_it_is_missing(server, capsys):
    assert smoke_api.main(["--url", server, "--calls", "2", "--no-demo"]) == 0
    capsys.readouterr()
    assert smoke_api.run(server + "/nowhere", None, 1, 2.0) == 1  # nothing answers there
