"""Deploy the demo to a Hugging Face Space (Gradio SDK), wait until it runs, then smoke test it.

    HF_TOKEN=... HF_SPACE=owner/name COMMIT=<sha> python scripts/deploy_space.py

Needs `huggingface_hub` and `gradio_client` (CI installs them with `uv run --with`). A free account
cannot create a Docker or a CPU Space, so the Space is a Gradio one on ZeroGPU hardware, created by
hand once. Only the files it needs are uploaded: no tests, experiment code or notes.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
# source in the repository -> name in the Space
FILES = {
    "space/space_app.py": "space_app.py",
    "space/requirements.txt": "requirements.txt",
    "space/README.md": "README.md",
    "LICENSE": "LICENSE",
}
FOLDERS = ["src", "app", "experiments/maps", "results"]  # the demo shows these maps and results
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "raw", "smoke")
TIMEOUT_SECONDS = 20 * 60


def stage(destination: Path, commit: str) -> list[str]:
    """Copy exactly what the Space needs into `destination`; return the relative file names."""
    for source, name in FILES.items():
        shutil.copy(ROOT / source, destination / name)
    for name in FOLDERS:
        shutil.copytree(ROOT / name, destination / name, ignore=IGNORE)
    (destination / "COMMIT_SHA").write_text(commit + "\n")
    return sorted(str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file())


def space_url(space: str) -> str:
    """The public URL of a Space: `owner/name` becomes `owner-name.hf.space` in lower case."""
    return "https://" + re.sub(r"[._]", "-", space.replace("/", "-")).lower() + ".hf.space"


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
    head = api.repo_info(space, repo_type="space").sha  # the Space's own commit, new or unchanged

    deadline = time.time() + TIMEOUT_SECONDS
    while time.time() < deadline:
        runtime = api.get_space_runtime(space)
        running = (runtime.raw or {}).get("sha")
        print(f"space stage: {runtime.stage}, running {str(running)[:7]}, want {head[:7]}")
        if runtime.stage.endswith("_ERROR"):
            print("the Space failed to build or start")
            return 1
        if runtime.stage == "RUNNING" and running == head:  # RUNNING alone may be the old build
            break
        time.sleep(15)
    else:
        print("timed out waiting for the Space to run this commit")
        return 1

    sys.path.insert(0, str(ROOT))
    from scripts import smoke_space

    return smoke_space.run(space_url(space), calls=3, budget=90.0)


if __name__ == "__main__":
    sys.exit(main())
