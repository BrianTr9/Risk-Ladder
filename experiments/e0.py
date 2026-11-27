"""E0: exact evaluation versus Monte Carlo (evidence for the oracle claim).

    uv run python -m experiments.e0 [--replications 300] [--quick] [--out DIR]

Fixed map and fixed policy (the shortest route, which runs beside the hazards). The slip probability
is tuned so that the policy's true failure probability is 10%, 1% and 0.1%. For each level we report
how many Monte Carlo rollouts are needed to estimate it within +-10% (relative) with 95% confidence,
how long that takes, and how long the exact evaluation takes.

Writes e0.json and e0.md to results/ (a `--quick` run writes to results/smoke/ so that it can never
overwrite the committed results; use `--out` to choose another folder).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

from riskladder.env import FAIL, Noise, build_model, load_map, steps_to_goal
from riskladder.env.dynamics import DELTA, DIRECTIONS
from riskladder.env.layout import GOAL, PASSABLE
from riskladder.oracle import evaluate, rollout, solve_policy

ROOT = Path(__file__).parents[1]
MAP = ROOT / "experiments" / "maps" / "dilemma" / "dilemma-30x30-s0.map"
RESULTS = ROOT / "results"
Z95 = 1.96
REL_ERROR = 0.10
TARGETS = (0.10, 0.01, 0.001)


def shortest_route_policy(spec) -> np.ndarray:
    """Per state, the 1-cell move that most reduces the hazard-free step distance to the goal."""
    layout = spec.layout
    dist = steps_to_goal(layout)
    mdp = build_model(spec)
    policy = np.zeros(mdp.n_states, dtype=int)
    for s, (r, c) in enumerate(mdp.cells):
        best = []
        for d in DIRECTIONS:  # the first four actions are the 1-cell moves, in this order
            dr, dc = DELTA[d]
            nr, nc = r + dr, c + dc
            inside = 0 <= nr < layout.rows and 0 <= nc < layout.cols
            best.append(
                dist[nr, nc] if inside and layout.grid[nr][nc] in PASSABLE | {GOAL} else np.inf
            )
        policy[s] = int(np.argmin(best))
    return policy


def failure_probability(spec, policy, p_slip: float) -> float:
    mdp = build_model(spec.with_noise(Noise(p_slip=p_slip)))
    return evaluate(mdp, policy, start=mdp.start).at(mdp.start).p_fail


def tune_slip(spec, policy, target: float) -> float:
    """Bisection on p_slip (failure probability is increasing in it)."""
    low, high = 1e-7, 0.5
    for _ in range(60):
        mid = math.sqrt(low * high)
        if failure_probability(spec, policy, mid) < target:
            low = mid
        else:
            high = mid
    return math.sqrt(low * high)


def required_rollouts(p: float) -> int:
    """n so that the standard error is REL_ERROR * p / Z95 (normal approximation)."""
    return math.ceil(Z95**2 * (1.0 - p) / (p * REL_ERROR**2))


def run(replications: int = 300, quick: bool = False) -> dict:
    spec = load_map(MAP)
    policy = shortest_route_policy(spec)
    rows = []
    for target in TARGETS:
        p_slip = tune_slip(spec, policy, target)
        noisy = spec.with_noise(Noise(p_slip=p_slip))

        t0 = time.perf_counter()
        mdp = build_model(noisy)
        build_ms = 1000 * (time.perf_counter() - t0)
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            exact = evaluate(mdp, policy, start=mdp.start).at(mdp.start)
            times.append(time.perf_counter() - t0)
        oracle_ms = 1000 * statistics.median(times)
        unbounded = solve_policy(mdp, policy)[2][mdp.start]  # independent linear-solve cross-check

        n = required_rollouts(exact.p_fail)
        n_run = min(n, 20_000) if quick else n
        rng = np.random.default_rng(0)
        estimates, seconds = [], []
        for _ in range(replications):
            t0 = time.perf_counter()
            estimates.append(rollout(mdp, policy, n_run, rng=rng).rate(FAIL))
            seconds.append(time.perf_counter() - t0)
        within = np.mean([abs(e - exact.p_fail) <= REL_ERROR * exact.p_fail for e in estimates])
        rows.append(
            {
                "p_fail_exact": exact.p_fail,
                "p_slip": p_slip,
                "rollouts_needed": n,
                "rollouts_run": n_run,
                "mc_seconds": statistics.median(seconds),
                "coverage_within_10pct": float(within),
                "oracle_ms": oracle_ms,
                "model_build_ms": build_ms,
                "speedup": statistics.median(seconds) / (oracle_ms / 1000),
                "abs_diff_vs_linear_solve": abs(exact.p_fail - unbounded),
            }
        )
    return {
        "map": MAP.name,
        "policy": "shortest route (1-cell moves toward the goal)",
        "replications": replications,
        "rows": rows,
    }


def to_markdown(result: dict) -> str:
    head = (
        f"E0: exact evaluation vs Monte Carlo on `{result['map']}` ({result['policy']}).\n\n"
        "Rollouts needed for a +-10% relative error at 95% confidence "
        f"(coverage is measured over {result['replications']} repeated runs of that size).\n\n"
        "| P(fail) exact | MC rollouts | MC time | Coverage | Oracle time | Speed-up |\n"
        "|---|---|---|---|---|---|\n"
    )
    lines = [
        f"| {r['p_fail_exact']:.4f} | {r['rollouts_needed']:,} | {r['mc_seconds']:.2f} s | "
        f"{100 * r['coverage_within_10pct']:.0f}% | {r['oracle_ms']:.1f} ms | "
        f"{r['speedup']:,.0f}x |"
        for r in result["rows"]
    ]
    return head + "\n".join(lines) + "\n"


def default_out(quick: bool) -> Path:
    return RESULTS / "smoke" if quick else RESULTS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replications", type=int, default=300)
    parser.add_argument("--quick", action="store_true", help="cap rollouts at 20k (smoke test)")
    parser.add_argument(
        "--out", type=Path, help="output folder (default: results/, or results/smoke/)"
    )
    args = parser.parse_args(argv)
    result = run(args.replications, args.quick)
    out = args.out or default_out(args.quick)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e0.json").write_text(json.dumps(result, indent=2) + "\n")
    table = to_markdown(result)
    (out / "e0.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
