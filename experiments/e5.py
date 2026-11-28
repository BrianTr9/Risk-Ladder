"""E5: the first rung. A* plans for a rover that never slips; what does that cost when it does?

    uv run python -m experiments.e5 [--quick] [--out DIR]

A* finds the cheapest route on the noise-free map, which is also the optimal policy of the
noise-free problem (checked here against value iteration). Played closed-loop, replanning from
wherever the rover ends up, that policy is scored exactly under the true slip probability and set
against the optimum for the same slip. On the `dilemma` maps the shortest route runs beside a
hazard strip, so ignoring the noise is cheap until the rover slips.

Writes e5.json, e5.md and figures/e5_astar.png (to results/smoke/ with --quick).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import plotting as style
from riskladder.agents import astar
from riskladder.env import Noise, build_model, load_map
from riskladder.oracle import Oracle, value_iteration

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results"
MAPS = sorted((ROOT / "experiments" / "maps" / "dilemma").glob("*.map"))
SLIPS = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4)
DASH_MULT = 2.0  # the default: a dash slips twice as often as a move


def noise_free_policy(spec) -> tuple[np.ndarray, float]:
    """The policy A* implies from every cell, and its cost from the start (equal to A*'s)."""
    model = build_model(spec.with_noise(Noise(p_slip=0.0, dash_slip_mult=0.0)))
    solution = value_iteration(model)
    cost = float(solution.V[model.start])
    plan = astar(spec)
    if not np.isclose(plan.cost, cost):  # the two views of the noise-free problem must agree
        raise AssertionError(f"A* cost {plan.cost} differs from value iteration {cost}")
    return solution.policy, cost


def run(quick: bool = False) -> dict:
    maps = MAPS[:3] if quick else MAPS
    slips = SLIPS[::2] if quick else SLIPS
    rows = {p: [] for p in slips}
    for path in maps:
        spec = load_map(path)
        policy, _ = noise_free_policy(spec)
        for p_slip in slips:
            oracle = Oracle(build_model(spec.with_noise(Noise(p_slip, DASH_MULT))))
            mine, best = oracle.evaluate(policy), oracle.optimal()
            rows[p_slip].append(
                {
                    "map": path.stem,
                    "p_fail": mine.p_fail,
                    "optimal_p_fail": best.p_fail,
                    "regret": (mine.cost - best.cost) / best.cost,
                    "p_timeout": mine.p_timeout,
                }
            )
    curve = []
    for p_slip, items in rows.items():
        fail = np.array([r["p_fail"] for r in items])
        regret = np.array([r["regret"] for r in items])
        curve.append(
            {
                "p_slip": p_slip,
                "p_fail": {
                    "mean": float(fail.mean()),
                    "min": float(fail.min()),
                    "max": float(fail.max()),
                },
                "regret": {
                    "median": float(np.median(regret)),
                    "min": float(regret.min()),
                    "max": float(regret.max()),
                },
                "optimal_p_fail": float(np.mean([r["optimal_p_fail"] for r in items])),
            }
        )
    return {"maps": len(maps), "dash_slip_mult": DASH_MULT, "curve": curve, "rows": rows}


def plot(result: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    with style.context():
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
        x = np.array([c["p_slip"] for c in result["curve"]])
        fail = result["curve"]
        mean = 100 * np.array([c["p_fail"]["mean"] for c in fail])
        lo = 100 * np.array([c["p_fail"]["min"] for c in fail])
        hi = 100 * np.array([c["p_fail"]["max"] for c in fail])
        axes[0].fill_between(x, lo, hi, color=style.BLUE, alpha=0.18, linewidth=0)
        axes[0].plot(x, mean, color=style.BLUE, linewidth=2, label="A* plan, replanned each step")
        axes[0].plot(
            x,
            100 * np.array([c["optimal_p_fail"] for c in fail]),
            color=style.ORANGE,
            linewidth=2,
            label="optimal policy for the true slip",
        )
        axes[0].set_ylabel("Probability of losing the rover (%)")
        axes[0].set_title("Failure probability of the noise-free plan")
        axes[0].legend(loc="center right")
        regret = result["curve"]
        axes[1].fill_between(
            x,
            100 * np.array([c["regret"]["min"] for c in regret]),
            100 * np.array([c["regret"]["max"] for c in regret]),
            color=style.BLUE,
            alpha=0.18,
            linewidth=0,
        )
        axes[1].plot(
            x,
            100 * np.array([c["regret"]["median"] for c in regret]),
            color=style.BLUE,
            linewidth=2,
        )
        axes[1].set_ylabel("Expected cost above the optimum (%)")
        axes[1].set_title("What the failures cost")
        for ax in axes:
            ax.set_xlabel("True slip probability of a move (a dash slips twice as often)")
            ax.set_xlim(0, x.max())
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)


def to_markdown(result: dict) -> str:
    lines = [
        "E5: the noise-free A* plan, scored under the true slip probability.\n",
        f"{result['maps']} dilemma maps; exact evaluation. Band and range are over maps; "
        "regret is the expected cost above the optimum for the same slip.\n",
        "| Slip | A* P(fail), mean (range) | Optimum P(fail) |"
        " Cost above the optimum, median (range) |",
        "|---|---|---|---|",
    ]
    for c in result["curve"]:
        f, r = c["p_fail"], c["regret"]
        lines.append(
            f"| {c['p_slip']:g} | {100 * f['mean']:.1f}% ({100 * f['min']:.1f} to "
            f"{100 * f['max']:.1f}) | {100 * c['optimal_p_fail']:.1f}% | "
            f"{100 * r['median']:.0f}% ({100 * r['min']:.0f} to {100 * r['max']:.0f}) |"
        )
    return "\n".join(lines) + "\n"


def default_out(quick: bool) -> Path:
    return RESULTS / "smoke" if quick else RESULTS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="three maps, fewer slips (smoke)")
    parser.add_argument("--out", type=Path, help="output folder (default: results/ or smoke/)")
    args = parser.parse_args(argv)
    result = run(args.quick)
    out = args.out or default_out(args.quick)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e5.json").write_text(json.dumps(result, indent=2) + "\n")
    table = to_markdown(result)
    (out / "e5.md").write_text(table)
    plot(result, out / "figures" / "e5_astar.png")
    print(table)


if __name__ == "__main__":
    main()
