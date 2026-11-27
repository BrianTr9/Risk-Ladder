"""E2: what a wrong model costs.

    uv run python -m experiments.e2 [--quick] [--out DIR]

The planner solves the constrained problem (bound 1%) for the slip probability it *believes*; the
oracle then scores that plan under the slip probability that is *true*. Averaged over the ridge
maps this gives two grids: how far the plan breaks the bound, and how much more or less it costs
than the best plan for the true model. A second table tests a cheap remedy, planning with a safety
margin on the slip probability, and shows its price when the belief was right.

Writes e2.json, e2.md and figures/e2_misspecification.png (to results/smoke/ with --quick).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import plotting as style
from riskladder.agents import solve_cmdp
from riskladder.env import Noise, build_model, load_map
from riskladder.oracle import evaluate_mixture

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results"
RIDGES = sorted((ROOT / "experiments" / "maps" / "ridge").glob("*.map"))
BOUND = 0.01
SLIPS = (0.0025, 0.005, 0.01, 0.02, 0.04)
MARGINS = (1.0, 2.0, 4.0)  # plan with margin * believed slip
UNDERESTIMATES = (1.5, 2.0, 3.0)  # the planner believes the true slip divided by this factor


class Planner:
    """Constrained solutions cached per (map, slip), so each is solved once."""

    def __init__(self, maps: list[Path]) -> None:
        self.maps = maps
        self._models: dict[tuple[int, float], object] = {}
        self._plans: dict[tuple[int, float], object] = {}

    def model(self, index: int, slip: float):
        key = (index, slip)
        if key not in self._models:
            spec = load_map(self.maps[index]).with_noise(Noise(p_slip=slip))
            self._models[key] = build_model(spec)
        return self._models[key]

    def plan(self, index: int, slip: float):
        key = (index, slip)
        if key not in self._plans:
            self._plans[key] = solve_cmdp(self.model(index, slip), BOUND)
        return self._plans[key]

    def score(self, index: int, planned_slip: float, true_slip: float) -> dict:
        """Plan for `planned_slip`, run under `true_slip`, compare with planning for the truth."""
        plan = self.plan(index, planned_slip)
        outcome = evaluate_mixture(self.model(index, true_slip), plan.mixture)
        best = self.plan(index, true_slip).result
        return {
            "p_fail": outcome.p_fail,
            "violation": max(0.0, outcome.p_fail - BOUND),
            "extra_cost": outcome.cost - best.cost,
        }


def grid_data(planner: Planner) -> dict:
    n = len(planner.maps)
    cells = []
    for believed in SLIPS:
        for true in SLIPS:
            scores = [planner.score(i, believed, true) for i in range(n)]
            cells.append(
                {
                    "believed": believed,
                    "true": true,
                    "mean_violation": float(np.mean([s["violation"] for s in scores])),
                    "share_violating": float(np.mean([s["violation"] > 1e-9 for s in scores])),
                    "mean_extra_cost": float(np.mean([s["extra_cost"] for s in scores])),
                }
            )
    return {"slips": list(SLIPS), "cells": cells}


def margin_data(planner: Planner) -> dict:
    """Planning with `margin * believed slip` when the belief is too low by a given factor, and
    its price when the belief was right. Planning slips off the grid are solved on demand."""
    n = len(planner.maps)
    rows = []
    for margin in MARGINS:
        row = {"margin": margin, "violation_by_underestimate": {}, "share_by_underestimate": {}}
        for factor in UNDERESTIMATES:
            scores = [
                planner.score(i, margin * true / factor, true)
                for true in SLIPS[1:]
                for i in range(n)
            ]
            row["violation_by_underestimate"][str(factor)] = float(
                np.mean([s["violation"] for s in scores])
            )
            row["share_by_underestimate"][str(factor)] = float(
                np.mean([s["violation"] > 1e-9 for s in scores])
            )
        right = [planner.score(i, margin * true, true) for true in SLIPS for i in range(n)]
        row["right_belief_mean_extra_cost"] = float(np.mean([s["extra_cost"] for s in right]))
        rows.append(row)
    return {"underestimates": list(UNDERESTIMATES), "rows": rows}


def plot_grids(data: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    slips = data["slips"]
    n = len(slips)
    violation = np.zeros((n, n))
    extra = np.zeros((n, n))
    for cell in data["cells"]:
        i, j = slips.index(cell["believed"]), slips.index(cell["true"])
        violation[i, j] = 100 * cell["mean_violation"]
        extra[i, j] = cell["mean_extra_cost"]

    sequential = LinearSegmentedColormap.from_list("sequential", style.SEQUENTIAL[:6])
    diverging = LinearSegmentedColormap.from_list("diverging", ["#2a78d6", "#f0efec", "#e34948"])
    limit = max(abs(extra.min()), abs(extra.max()), 1e-9)

    with style.context():
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
        panels = (
            (axes[0], violation, sequential, 0.0, max(violation.max(), 1e-9),
             "Mean excess risk over the 1% bound (percentage points)", "{:.2f}"),
            (axes[1], extra, diverging, -limit, limit,
             "Cost against the best plan for the true model", "{:+.2f}"),
        )  # fmt: skip
        for ax, grid, cmap, low, high, title, fmt in panels:
            ax.imshow(grid, cmap=cmap, vmin=low, vmax=high, origin="lower", aspect="equal")
            ax.set_title(title, fontsize=9.5)
            ax.set_xticks(range(n), [f"{s:g}" for s in slips])
            ax.set_yticks(range(n), [f"{s:g}" for s in slips])
            ax.set_xlabel("True slip probability")
            ax.set_ylabel("Slip probability the planner believed")
            ax.grid(False)
            for i in range(n):
                for j in range(n):
                    shade = (grid[i, j] - low) / (high - low)
                    dark = shade > 0.62 if cmap is sequential else abs(grid[i, j]) / limit > 0.6
                    ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center", fontsize=8,
                            color="#ffffff" if dark else style.INK)  # fmt: skip
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)


def to_markdown(result: dict) -> str:
    factors = result["margin"]["underestimates"]
    lines = [
        "E2: planning with the wrong slip probability (bound 1%, "
        f"{result['maps']} ridge maps, exact scoring under the true model).\n",
        "Planning with a margin: the believed slip is multiplied by the margin before planning. "
        "Cells show the mean excess risk over the 1% bound, in percentage points, when the belief "
        "is the true slip divided by the factor, and the extra cost when the belief is right.\n",
        "| Margin | "
        + " | ".join(f"Belief low by {f:g}x" for f in factors)
        + " | Belief right: extra cost |",
        "|---|" + "---|" * (len(factors) + 1),
    ]
    for r in result["margin"]["rows"]:
        cells = " | ".join(f"{100 * r['violation_by_underestimate'][str(f)]:.2f}" for f in factors)
        lines.append(f"| {r['margin']:g}x | {cells} | {r['right_belief_mean_extra_cost']:+.3f} |")
    return "\n".join(lines) + "\n"


def run(quick: bool = False) -> dict:
    maps = RIDGES[:3] if quick else RIDGES
    planner = Planner(maps)
    return {
        "maps": len(maps),
        "bound": BOUND,
        "grid": grid_data(planner),
        "margin": margin_data(planner),
    }


def default_out(quick: bool) -> Path:
    return RESULTS / "smoke" if quick else RESULTS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="three maps only (smoke test)")
    parser.add_argument("--out", type=Path, help="output folder (default: results/, or smoke/)")
    args = parser.parse_args(argv)
    result = run(args.quick)
    out = args.out or default_out(args.quick)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e2.json").write_text(json.dumps(result, indent=2) + "\n")
    table = to_markdown(result)
    (out / "e2.md").write_text(table)
    plot_grids(result["grid"], out / "figures" / "e2_misspecification.png")
    print(table)


if __name__ == "__main__":
    main()
