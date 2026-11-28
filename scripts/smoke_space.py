"""Smoke test a running Gradio Space (or a local copy of it) through its `/plan` endpoint.

    python scripts/smoke_space.py --url https://owner-name.hf.space

Needs `gradio_client`. It sends the golden request of `smoke_api.py` and checks that the answer is
the exact constrained optimum, then reports how long a call takes. Exit status 1 means a check
failed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from gradio_client import Client

sys.path.insert(0, str(Path(__file__).parents[1]))  # so that `python scripts/smoke_space.py` works
from scripts import smoke_api


def run(url: str, calls: int = 3, budget: float = 60.0) -> int:
    failures: list[str] = []
    client = Client(url, verbose=False)
    text = smoke_api.MAP_FILE.read_text()
    times = []
    answer: dict = {}
    for _ in range(calls):
        started = time.perf_counter()
        answer = client.predict(text, smoke_api.GOLDEN_DELTA, -1.0, api_name="/plan")
        times.append(time.perf_counter() - started)
    outcome = answer.get("outcome", {})
    smoke_api.check(
        abs(outcome.get("p_fail", 1) - smoke_api.GOLDEN_DELTA) < 1e-9,
        "failure probability equals the bound",
        failures,
    )
    smoke_api.check(
        abs(outcome.get("cost", 0) - smoke_api.GOLDEN_COST) < 1e-6,
        "expected cost matches the golden value",
        failures,
    )
    smoke_api.check(abs(answer.get("optimality_gap", 1)) < 1e-6, "certified optimal", failures)
    slowest = max(times)
    print(f"{calls} calls, slowest {slowest:.1f} s (a cold Space adds a wake-up)")
    smoke_api.check(slowest <= budget, f"within the {budget:g} s budget", failures)
    print("PASS" if not failures else f"{len(failures)} check(s) failed")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--budget", type=float, default=60.0, help="seconds for the slowest call")
    args = parser.parse_args(argv)
    return run(args.url, args.calls, args.budget)


if __name__ == "__main__":
    sys.exit(main())
