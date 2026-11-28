"""Deploy the API to a Hugging Face Space, wait until it serves this commit, then smoke test it.

    HF_TOKEN=... HF_SPACE=owner/name COMMIT=<sha> python scripts/deploy_space.py

Needs `huggingface_hub` (CI installs it with `uv run --with huggingface_hub`). Only the files the
image needs are uploaded: the Space does not receive tests, experiment code or notes.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parents[1]
FILES = ["Dockerfile", "pyproject.toml", "uv.lock", "LICENSE"]
FOLDERS = ["src", "app", "experiments/maps", "results"]  # the demo shows these maps and results
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "raw", "smoke")
TIMEOUT_SECONDS = 20 * 60


def stage(destination: Path, commit: str) -> list[str]:
    """Copy exactly what the Space needs into `destination`; return the relative file names."""
    for name in FILES:
        shutil.copy(ROOT / name, destination / name)
    for name in FOLDERS:
        shutil.copytree(ROOT / name, destination / name, ignore=IGNORE)
    shutil.copy(ROOT / "space" / "README.md", destination / "README.md")
    (destination / "COMMIT_SHA").write_text(commit + "\n")
    return sorted(str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file())


def space_url(space: str) -> str:
    """The public URL of a Space: `owner/name` becomes `owner-name.hf.space` in lower case."""
    return "https://" + re.sub(r"[._]", "-", space.replace("/", "-")).lower() + ".hf.space"


def served_commit(url: str) -> str | None:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=10) as response:
            return json.loads(response.read()).get("commit")
    except Exception:  # the Space is still building or waking up
        return None


def main() -> int:
    from huggingface_hub import HfApi

    token, space, commit = os.environ["HF_TOKEN"], os.environ["HF_SPACE"], os.environ["COMMIT"]
    api = HfApi(token=token)
    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp), commit)
        print(f"uploading {len(files)} files to {space}")
        api.upload_folder(
            folder_path=tmp, repo_id=space, repo_type="space", commit_message=f"Deploy {commit[:7]}"
        )

    url = space_url(space)
    deadline = time.time() + TIMEOUT_SECONDS
    while time.time() < deadline:
        stage_name = api.get_space_runtime(space).stage
        print(f"space stage: {stage_name}")
        if stage_name.endswith("_ERROR"):
            print("the Space failed to build or start")
            return 1
        if stage_name == "RUNNING" and (served_commit(url) or "").startswith(commit[:7]):
            break
        time.sleep(15)
    else:
        print("timed out waiting for the Space to serve this commit")
        return 1

    sys.path.insert(0, str(ROOT / "scripts"))
    import smoke_api

    return smoke_api.run(url, commit, calls=20, budget=3.0)


if __name__ == "__main__":
    sys.exit(main())
