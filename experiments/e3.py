"""E3: a failure penalty is not a risk constraint.

    uv run python -m experiments.e3 [--quick] [--out DIR]

Three results on the ridge maps, all computed exactly by the oracle:

1. Frontier. The constrained solver traces the whole cost/risk curve; the penalty planner can only
   reach the corners of it, one policy per price.
2. Risk spread. Pick the penalty that meets a 1% bound on one map, then reuse that same penalty on
   other maps and at other slip levels. The failure probability is far from 1% in both directions;
   the constrained solver meets the bound on every map at the lowest cost.
3. Tail risk. Expected cost and the expected cost of the worst 5% of successful missions
   (CVaR 0.95, a Monte Carlo estimate with a bootstrap interval) along the frontier.

Writes e3.json, e3.md and figures/e3_*.png (to results/smoke/ with --quick).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import plotting as style
from riskladder.agents import penalty_planner, solve_cmdp
from riskladder.env import Noise, build_model, load_map
from riskladder.oracle import PolicyResult, cvar, evaluate_mixture, rollout

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results"
RIDGES = sorted((ROOT / "experiments" / "maps" / "ridge").glob("*.map"))
BOUND = 0.01  # the risk bound used for the spread experiment
SLIP_LEVELS = (0.005, 0.01, 0.02)
CVAR_BOUNDS = (None, 0.03, 0.01, 0.003, 0.0)


def penalty_vertices(mdp, n: int = 200) -> list[dict]:
    """Distinct (P(fail), cost) pairs reached by the penalty planner over a fine price grid."""
    seen: dict[tuple[float, float], float] = {}
    for extra in [0.0, *np.geomspace(1.0, 3000.0, n)]:
        r = evaluate_mixture(mdp, penalty_planner(mdp, extra).policy)
        seen.setdefault((round(r.p_fail, 9), round(r.cost, 6)), float(extra))
    return [
        {"p_fail": f, "cost": c, "first_price": extra}
        for (f, c), extra in sorted(seen.items(), reverse=True)
    ]


def frontier_data(mdp, points: int) -> dict:
    """The exact frontier is piecewise linear with a corner at every penalty vertex, so the bounds
    include each vertex's failure probability; the other bounds only add points on the segments."""
    unconstrained = evaluate_mixture(mdp, penalty_planner(mdp, 0.0).policy)
    vertices = penalty_vertices(mdp)
    deltas = {0.0, *(v["p_fail"] for v in vertices)}
    deltas |= set(np.geomspace(1e-4, unconstrained.p_fail, points).tolist())
    curve = []
    for delta in sorted(deltas):
        s = solve_cmdp(mdp, float(min(delta, 1.0)))
        curve.append({"delta": float(delta), "p_fail": s.result.p_fail, "cost": s.result.cost,
                      "gap": s.gap, "parts": len(s.mixture)})  # fmt: skip
    return {
        "unconstrained": {"p_fail": unconstrained.p_fail, "cost": unconstrained.cost},
        "cmdp": curve,
        "penalty": vertices,
    }


def spread_data(maps: list[Path]) -> dict:
    """The penalty tuned on the first map at the middle slip level, reused everywhere."""
    tuning = build_model(load_map(maps[0]).with_noise(Noise(p_slip=0.01)))
    price = solve_cmdp(tuning, BOUND).multiplier
    rows = []
    for p_slip in SLIP_LEVELS:
        for index, path in enumerate(maps):
            mdp = build_model(load_map(path).with_noise(Noise(p_slip=p_slip)))
            penalised = evaluate_mixture(mdp, penalty_planner(mdp, price).policy)
            constrained = solve_cmdp(mdp, BOUND)
            rows.append(
                {
                    "p_slip": p_slip,
                    "map": index,
                    "penalty": {"p_fail": penalised.p_fail, "cost": penalised.cost},
                    "cmdp": {"p_fail": constrained.result.p_fail, "cost": constrained.result.cost},
                }
            )
    summary = {}
    for p_slip in SLIP_LEVELS:
        mine = [r for r in rows if r["p_slip"] == p_slip]
        violating = [r for r in mine if r["penalty"]["p_fail"] > BOUND + 1e-9]
        safe = [r for r in mine if r["penalty"]["p_fail"] <= BOUND + 1e-9]
        summary[str(p_slip)] = {
            "maps": len(mine),
            "penalty_violates_bound": len(violating),
            "penalty_max_p_fail": max(r["penalty"]["p_fail"] for r in mine),
            "cmdp_max_p_fail": max(r["cmdp"]["p_fail"] for r in mine),
            "cmdp_violates_bound": sum(r["cmdp"]["p_fail"] > BOUND + 1e-9 for r in mine),
            "mean_excess_cost_when_penalty_is_safe": (
                float(np.mean([r["penalty"]["cost"] - r["cmdp"]["cost"] for r in safe]))
                if safe
                else None
            ),
        }
    return {"bound": BOUND, "price": price, "rows": rows, "summary": summary}


def cvar_table(mdp, samples: int) -> list[dict]:
    rows = []
    for bound in CVAR_BOUNDS:
        if bound is None:
            mixture = ((1.0, penalty_planner(mdp, 0.0).policy),)
        else:
            mixture = solve_cmdp(mdp, bound).mixture
        exact: PolicyResult = evaluate_mixture(mdp, mixture)
        sims = rollout(mdp, mixture, samples, rng=7)
        tail = cvar(sims.success_costs, 0.95, rng=7)
        rows.append(
            {
                "bound": bound,
                "p_fail": exact.p_fail,
                "expected_cost": exact.cost,
                "mean_success_cost": float(sims.success_costs.mean()),
                "cvar95_success": tail.value,
                "cvar_ci": [tail.ci_low, tail.ci_high],
            }
        )
    return rows


def _dots(ax, x, y, colour, label):
    """Markers of at least 8 px with a 2 px ring in the surface colour so overlaps stay readable."""
    ax.plot(
        x, y, linestyle="none", marker="o", markersize=8, color=colour,
        markeredgecolor=style.SURFACE, markeredgewidth=2, label=label,
    )  # fmt: skip


def plot_frontier(data: dict, name: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    style.apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    curve = data["cmdp"]
    ax.plot(
        [100 * p["p_fail"] for p in curve], [p["cost"] for p in curve],
        color=style.BLUE, linewidth=2, label="Constrained solver (mixes two policies)",
    )  # fmt: skip
    _dots(
        ax, [100 * v["p_fail"] for v in data["penalty"]], [v["cost"] for v in data["penalty"]],
        style.ORANGE, "Penalty planner (one policy per price)",
    )  # fmt: skip
    top = data["unconstrained"]
    ax.annotate(
        "failure priced as in\nthe environment",
        xy=(100 * top["p_fail"], top["cost"]), xytext=(-10, 16), textcoords="offset points",
        ha="right", fontsize=8.5, color=style.INK_2,
        arrowprops={"arrowstyle": "-", "color": style.MUTED, "lw": 0.8},
    )  # fmt: skip
    ax.set_xlabel("Probability of losing the rover (%)")
    ax.set_ylabel("Expected cost, including failures")
    ax.set_title(f"Cost against risk on {name}")
    ax.set_xlim(left=-0.1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_spread(data: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    style.apply_style()
    fig, axes = plt.subplots(1, len(SLIP_LEVELS), figsize=(9.6, 4.0), sharey=True)
    bound = 100 * data["bound"]
    for ax, p_slip in zip(axes, SLIP_LEVELS, strict=True):
        rows = [r for r in data["rows"] if r["p_slip"] == p_slip]
        x = np.array([r["map"] for r in rows], dtype=float)
        ax.axhline(bound, color=style.MUTED, linewidth=1, linestyle=(0, (4, 3)))
        # the two series sit side by side so that equal values do not hide each other
        _dots(ax, x - 0.14, [100 * r["penalty"]["p_fail"] for r in rows], style.ORANGE,
              "Same penalty on every map")  # fmt: skip
        _dots(ax, x + 0.14, [100 * r["cmdp"]["p_fail"] for r in rows], style.BLUE,
              "Constrained solver, bound 1%")  # fmt: skip
        ax.set_title(f"Slip probability {p_slip}")
        ax.set_xlabel("Map")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(i)) for i in x], fontsize=7.5)
        ax.text(x[-1] + 0.4, bound + 0.1, "1% bound", ha="right", va="bottom",
                fontsize=8, color=style.INK_2)  # fmt: skip
    axes[0].set_ylabel("Probability of losing the rover (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def to_markdown(result: dict) -> str:
    lines = ["E3: a failure penalty is not a risk constraint.\n"]
    lines += [
        f"Penalty price tuned on map 0 at slip 0.01 to meet a {100 * BOUND:.0f}% bound "
        f"(extra price {result['spread']['price']:.1f} on top of the environment's failure cost), "
        "then reused unchanged.\n",
        "| Slip | Maps | Penalty breaks the bound | Worst penalty P(fail) |"
        " Constrained worst P(fail) | Extra cost of the penalty where it is safe |",
        "|---|---|---|---|---|---|",
    ]
    for slip, s in result["spread"]["summary"].items():
        excess = s["mean_excess_cost_when_penalty_is_safe"]
        lines.append(
            f"| {slip} | {s['maps']} | {s['penalty_violates_bound']} of {s['maps']} | "
            f"{100 * s['penalty_max_p_fail']:.2f}% | {100 * s['cmdp_max_p_fail']:.2f}% | "
            f"{'n/a' if excess is None else f'{excess:+.2f}'} |"
        )
    lines += [
        "\nExpected cost against tail cost along the frontier (map 0, slip 0.01). "
        "CVaR is the mean mission cost of the worst 5% of successful missions.\n",
        "| Bound | P(fail) | Expected cost | Mean mission cost | CVaR 0.95 (95% interval) |",
        "|---|---|---|---|---|",
    ]
    for r in result["cvar"]:
        bound = "none" if r["bound"] is None else f"{100 * r['bound']:g}%"
        low, high = r["cvar_ci"]
        lines.append(
            f"| {bound} | {100 * r['p_fail']:.2f}% | {r['expected_cost']:.2f} | "
            f"{r['mean_success_cost']:.2f} | {r['cvar95_success']:.2f} ({low:.2f} to {high:.2f}) |"
        )
    return "\n".join(lines) + "\n"


def run(quick: bool = False) -> dict:
    mdp = build_model(load_map(RIDGES[0]))
    return {
        "map": RIDGES[0].stem,
        "frontier": frontier_data(mdp, points=12 if quick else 60),
        "spread": spread_data(RIDGES[:4] if quick else RIDGES),
        "cvar": cvar_table(mdp, samples=20_000 if quick else 200_000),
    }


def default_out(quick: bool) -> Path:
    return RESULTS / "smoke" if quick else RESULTS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="fewer points and maps (smoke test)")
    parser.add_argument(
        "--out", type=Path, help="output folder (default: results/ or results/smoke/)"
    )
    args = parser.parse_args(argv)
    result = run(args.quick)
    out = args.out or default_out(args.quick)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e3.json").write_text(json.dumps(result, indent=2) + "\n")
    table = to_markdown(result)
    (out / "e3.md").write_text(table)
    plot_frontier(result["frontier"], result["map"], out / "figures" / "e3_frontier.png")
    plot_spread(result["spread"], out / "figures" / "e3_risk_spread.png")
    print(table)


if __name__ == "__main__":
    main()
