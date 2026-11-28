"""One entry point for every experiment.

    uv run python -m experiments.run list
    uv run python -m experiments.run all            # every experiment, then SUMMARY.md (long)
    uv run python -m experiments.run e1 e3 summary
    uv run python -m experiments.run all --smoke    # tiny versions into results/smoke/ (minutes)

Settings live next to the code they belong to (module constants, and the two tuned-settings files
`e1_config.json`, `e1b_config.json` and `e4_config.json`, rewritten only by `--tune`). Every
experiment writes `<name>.json`, `<name>.md` and its figures; `summary` then builds `SUMMARY.md`.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

from bench import latency

from experiments import e0, e1, e1b, e1c, e2, e3, e4, e5, summary

RESULTS = Path(__file__).parents[1] / "results"

# name -> (what it shows, main(), arguments for a smoke run)
EXPERIMENTS: dict[str, tuple[str, Callable, list[str]]] = {
    "bench": ("oracle latency by map size", latency.main, ["--sizes", "12", "34"]),
    "e0": ("exact evaluation versus Monte Carlo", e0.main, ["--quick"]),
    "e5": ("A* plan under slip (first rung)", e5.main, ["--quick"]),
    "e3": ("penalty versus constraint, tail risk", e3.main, ["--quick"]),
    "e2": ("wrong slip estimate and safety margin", e2.main, ["--quick"]),
    "e4": ("learners and generalisation to new maps", e4.main, ["--quick"]),
    "e1": ("the cost of learning and prior knowledge", e1.main, ["--quick"]),
    "e1b": ("the shield on SARSA, two map families", e1b.main, ["--quick"]),
    "e1c": ("which potential shaping should follow", e1c.main, ["--quick"]),
}


def run(names: list[str], smoke: bool, out: Path | None) -> Path:
    """Run the named experiments (and the summary) and return the folder written to."""
    target = out or (RESULTS / "smoke" if smoke else RESULTS)
    for name in names:
        if name == "summary":
            continue
        _, main, quick = EXPERIMENTS[name]
        started = time.perf_counter()
        main([*(quick if smoke else []), "--out", str(target)])
        print(f"[{name}] {time.perf_counter() - started:.0f} s", flush=True)
    if "summary" in names:
        summary.main(["--results", str(target)])
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="experiment names, 'all', 'summary' or 'list'")
    parser.add_argument("--smoke", action="store_true", help="tiny versions (results/smoke/)")
    parser.add_argument("--out", type=Path, help="output folder")
    args = parser.parse_args(argv)
    if not args.names or args.names == ["list"]:
        for name, (what, _, _) in EXPERIMENTS.items():
            print(f"{name:6s} {what}")
        print("summary  results/SUMMARY.md from the result files")
        return
    names = [*EXPERIMENTS, "summary"] if args.names == ["all"] else args.names
    unknown = [n for n in names if n not in EXPERIMENTS and n != "summary"]
    if unknown:
        parser.error(f"unknown experiment(s) {unknown}; choose from {[*EXPERIMENTS, 'summary']}")
    run(names, args.smoke, args.out)


if __name__ == "__main__":
    main()
