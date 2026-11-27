"""Smoke test a running RiskLadder API: health, a golden plan, a round trip and latency.

    python scripts/smoke_api.py --url http://localhost:7860 [--commit SHA] [--budget 2.0]

Standard library only, so it runs in CI, against a local container and against the deployed Space
alike. Latency is reported as percentiles (p50, p95) because a mean hides a slow tail; the check is
on p95. Exit status 1 means a check failed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parents[1]
MAP_FILE = ROOT / "experiments" / "maps" / "ridge" / "ridge-6x30-s3000.map"
# Constrained optimum for a 1% failure bound on that map, computed by the oracle.
GOLDEN_DELTA, GOLDEN_COST = 0.01, 29.587684696664425


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: the value at rank ceil(p / 100 * n) of the sorted sample."""
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p / 100 * len(ordered))) - 1]


def call(url: str, payload: dict | None = None, timeout: float = 30.0) -> tuple[int, dict, float]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read()
    return status, json.loads(body), time.perf_counter() - started


def check(condition: bool, message: str, failures: list[str]) -> None:
    print(("ok    " if condition else "FAIL  ") + message)
    if not condition:
        failures.append(message)


def run(url: str, commit: str | None, calls: int, budget: float) -> int:
    url = url.rstrip("/")
    failures: list[str] = []
    text = MAP_FILE.read_text()

    status, health, _ = call(f"{url}/health")
    check(status == 200 and health.get("status") == "ok", "GET /health is ok", failures)
    if commit:
        check(
            health.get("commit", "").startswith(commit),
            f"deployed commit is {commit[:7]}",
            failures,
        )

    status, plan, _ = call(f"{url}/plan", {"map": text, "delta": GOLDEN_DELTA})
    outcome = plan.get("outcome", {})
    check(status == 200, "POST /plan answers", failures)
    check(
        abs(outcome.get("p_fail", 1) - GOLDEN_DELTA) < 1e-9,
        "failure probability equals the bound",
        failures,
    )
    check(
        abs(outcome.get("cost", 0) - GOLDEN_COST) < 1e-6,
        "expected cost matches the golden value",
        failures,
    )
    check(abs(plan.get("optimality_gap", 1)) < 1e-6, "solution is certified optimal", failures)

    components = [
        {"weight": c["weight"], "actions": c["actions"]} for c in plan.get("components", [])
    ]
    status, scored, _ = call(f"{url}/evaluate", {"map": text, "components": components})
    same = abs(scored.get("outcome", {}).get("cost", 0) - GOLDEN_COST) < 1e-6
    check(status == 200 and same, "POST /evaluate reproduces the plan's outcome", failures)

    status, _, _ = call(f"{url}/plan", {"map": "nonsense"})
    check(status == 422, "a malformed map is a client error (422)", failures)

    times = [call(f"{url}/plan", {"map": text, "delta": GOLDEN_DELTA})[2] for _ in range(calls)]
    p50, p95 = percentile(times, 50), percentile(times, 95)
    print(f"latency over {calls} calls: p50 {1000 * p50:.0f} ms, p95 {1000 * p95:.0f} ms")
    check(p95 <= budget, f"p95 latency within the {budget:g} s budget", failures)

    print("PASS" if not failures else f"{len(failures)} check(s) failed")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:7860")
    parser.add_argument("--commit", help="expect /health to report this commit (prefix match)")
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--budget", type=float, default=2.0, help="p95 latency budget in seconds")
    args = parser.parse_args(argv)
    return run(args.url, args.commit, args.calls, args.budget)


if __name__ == "__main__":
    sys.exit(main())
