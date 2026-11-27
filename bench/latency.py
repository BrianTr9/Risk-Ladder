"""Latency and memory of the oracle as the number of states grows.

    uv run python -m bench.latency [--sizes 12 34 102 320 1002] [--pi-max-states 150000] [--out DIR]

Each size runs in its own subprocess so that peak memory is measured per size. The map is open
terrain with sparse hazards, rocks and rough ground (`field` template), solved at the default noise.
Writes bench.json, bench.md and figures/bench_latency.png to results/. Any run that is not the full
default one writes to results/smoke/ instead, so it cannot overwrite the committed results.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results"
# Grid sides giving about 1e2, 1e3, 1e4, 1e5 and 1e6 decision states.
DEFAULT_SIZES = (12, 34, 102, 320, 1002)
DEFAULT_PI_MAX_STATES = 150_000
TERRAIN = {"hazard_density": 0.02, "rock_density": 0.02, "rough_density": 0.05}


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1e3  # bytes on macOS, KiB on Linux


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def run_size(size: int, pi_max_states: int) -> dict:
    from riskladder.env import build_model, generate
    from riskladder.oracle import evaluate, policy_iteration, value_iteration

    spec = generate("field", size, seed=0, **TERRAIN)
    mdp, build_s = timed(lambda: build_model(spec))
    row: dict = {"grid": size, "states": mdp.n_states, "build_s": build_s}

    vi, row["vi_s"] = timed(lambda: value_iteration(mdp))
    row["vi_iterations"] = vi.iterations
    row["vi_ms_per_iteration"] = 1000 * row["vi_s"] / vi.iterations

    ev, row["evaluate_from_start_s"] = timed(lambda: evaluate(mdp, vi.policy, start=mdp.start))
    row["evaluate_steps"] = ev.steps
    row["optimal_p_timeout"] = ev.at(mdp.start).p_timeout
    if mdp.n_states <= 150_000:
        _, row["evaluate_all_states_s"] = timed(lambda: evaluate(mdp, vi.policy))

    if mdp.n_states <= pi_max_states:
        pi, row["pi_s"] = timed(lambda: policy_iteration(mdp))
        row["pi_iterations"] = pi.iterations
        row["pi_max_value_gap"] = float(np.abs(pi.V[pi.live] - vi.V[vi.live]).max())
    row["peak_rss_mb"] = peak_rss_mb()
    return row


def machine() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def to_markdown(result: dict) -> str:
    lines = [
        "| Grid | States | Build | VI (iterations) | PI (iterations) | Evaluate | Peak memory |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        pi = f"{r['pi_s']:.2f} s ({r['pi_iterations']})" if "pi_s" in r else "skipped"
        lines.append(
            f"| {r['grid']}x{r['grid']} | {r['states']:,} | {r['build_s']:.2f} s | "
            f"{r['vi_s']:.2f} s ({r['vi_iterations']}) | {pi} | "
            f"{1000 * r['evaluate_from_start_s']:.0f} ms | {r['peak_rss_mb']:,.0f} MB |"
        )
    m = result["machine"]
    note = (
        f"\nMeasured on {m['platform']} ({m['machine']}), Python {m['python']}, "
        f"NumPy {m['numpy']}, SciPy {m['scipy']}.\n"
    )
    return "\n".join(lines) + "\n" + note


def plot(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["rows"]
    n = [r["states"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    series = [
        ("model build", "build_s"),
        ("value iteration", "vi_s"),
        ("policy iteration (direct solve)", "pi_s"),
        ("evaluate one policy", "evaluate_from_start_s"),
    ]
    for label, key in series:
        pts = [(r["states"], r[key]) for r in rows if key in r]
        ax.plot(*zip(*pts, strict=True), marker="o", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("decision states")
    ax.set_ylabel("seconds")
    ax.set_title("Oracle latency by map size")
    ax.set_xticks(n)
    ax.set_xticklabels([f"{v:,}" for v in n], rotation=30, ha="right")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def default_out(sizes: list[int], pi_max_states: int) -> Path:
    full = tuple(sizes) == DEFAULT_SIZES and pi_max_states == DEFAULT_PI_MAX_STATES
    return RESULTS if full else RESULTS / "smoke"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--pi-max-states", type=int, default=DEFAULT_PI_MAX_STATES)
    parser.add_argument("--child", type=int, help="internal: run one size and print JSON")
    parser.add_argument(
        "--out", type=Path, help="output folder (default: results/, or results/smoke/)"
    )
    args = parser.parse_args(argv)

    if args.child is not None:
        print(json.dumps(run_size(args.child, args.pi_max_states)))
        return

    rows = []
    for size in args.sizes:
        out = subprocess.run(
            [
                sys.executable,
                "-m",
                "bench.latency",
                "--child",
                str(size),
                "--pi-max-states",
                str(args.pi_max_states),
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        )
        rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
        print(f"done {size}x{size}: {rows[-1]['states']:,} states", flush=True)
    result = {"machine": machine(), "pi_max_states": args.pi_max_states, "rows": rows}
    out = args.out or default_out(args.sizes, args.pi_max_states)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps(result, indent=2) + "\n")
    table = to_markdown(result)
    (out / "bench.md").write_text(table)
    plot(result, out / "figures" / "bench_latency.png")
    print(table)


if __name__ == "__main__":
    main()
