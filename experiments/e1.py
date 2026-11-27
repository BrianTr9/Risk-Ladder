"""E1: what prior knowledge buys back from the cost of learning (the headline experiment).

    uv run python -m experiments.e1 [--quick] [--tune] [--out DIR]

Eight ways to learn the same ten `dilemma` maps, every one with the same step budget and the same
seeds, scored exactly by the oracle:

    Q, SARSA                      model-free, nothing known
    Q + shaping                   a potential from the hazard-free distance to the goal
    Q + shield                    actions that could lose the rover in one step are blocked
    Q + shield + shaping          both
    MB-count                      estimates a model from counts, knows nothing of the map
    MB-struct                     knows the geometry, learns the slip probabilities and costs
    MB-struct + CMDP              the same, planning with a failure-probability bound

The cost of learning is the number of rovers lost while training (x axis); what the learner ends
up with is its exact regret against the optimum (y axis). Learner step sizes come from
`experiments/e4_config.json` (tuned on other maps); the shaping scale is tuned here on dilemma
maps that are not among the ten (`--tune`), and the shield uses its documented defaults.

Writes e1.json, e1.md and figures/e1_*.png (to results/smoke/ with --quick).
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments import e4
from experiments import plotting as style
from riskladder.env import generate, save_map
from riskladder.harness import RunSpec, summarize
from riskladder.harness.runner import mean_ci, write_jsonl

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results"
CONFIG = Path(__file__).parent / "e1_config.json"
MAPS = sorted((ROOT / "experiments" / "maps" / "dilemma").glob("*.map"))
TUNE_SEEDS = range(100, 105)  # dilemma maps that are not among the evaluation maps
SHIELD = {"p_max": 0.4, "theta": 0.05}
DELTA = 0.01
TAU = 0.05
SHAPING_GRID = (0.5, 1.0, 2.0)
BUDGET = {"full": 1_000_000, "quick": 40_000}
MAIN = ("Q", "Q + shield + shaping", "MB-struct", "MB-struct + CMDP")  # the learning-curve panel


def configs(scale: float) -> dict[str, tuple[str, dict]]:
    """name -> (agent, settings). The tabular step sizes are the tuned ones from E4."""
    tabular = e4.load_config()
    q, sarsa = tabular.get("q", {}), tabular.get("sarsa", {})
    return {
        "Q": ("q", dict(q)),
        "SARSA": ("sarsa", dict(sarsa)),
        "Q + shaping": ("q", {**q, "shaping": scale}),
        "Q + shield": ("q", {**q, "shield": SHIELD}),
        "Q + shield + shaping": ("q", {**q, "shield": SHIELD, "shaping": scale}),
        "MB-count": ("mb_count", {}),
        "MB-struct": ("mb_struct", {}),
        "MB-struct + CMDP": ("mb_struct", {"cmdp": True, "delta": DELTA}),
    }


def load_scale() -> float:
    return json.loads(CONFIG.read_text())["shaping"] if CONFIG.exists() else 1.0


def _pool_size() -> int:
    return max(1, min(10, (os.cpu_count() or 2) - 1))


def tune(budget: int, scratch: Path) -> dict:
    """Pick the shaping scale with the smallest mean regret over the run on the tuning maps."""
    paths = []
    for seed in list(TUNE_SEEDS)[:2] if budget < 100_000 else TUNE_SEEDS:
        spec = generate("dilemma", seed=seed, rows=30)
        path = scratch / f"{spec.name}.map"
        scratch.mkdir(parents=True, exist_ok=True)
        save_map(spec, path)
        paths.append(str(path))
    q = e4.load_config().get("q", {})
    specs = [
        RunSpec("q", p, 0, budget, {**q, "shaping": scale}, 10, TAU)
        for scale in SHAPING_GRID
        for p in paths
    ]
    with ProcessPoolExecutor(_pool_size()) as pool:
        records = list(pool.map(e4._run_one, specs, chunksize=1))
    scores = []
    for i in range(len(SHAPING_GRID)):
        part = records[i * len(paths) : (i + 1) * len(paths)]
        scores.append(
            float(
                np.mean(
                    [
                        np.mean([c["regret"] / r["optimal_cost"] for c in r["checkpoints"]])
                        for r in part
                    ]
                )
            )
        )
    return {
        "shaping": SHAPING_GRID[int(np.argmin(scores))],
        "tuned_on": f"dilemma seeds {TUNE_SEEDS.start}-{TUNE_SEEDS.stop - 1}",
        "scores": [
            {"shaping": g, "mean_regret": s} for g, s in zip(SHAPING_GRID, scores, strict=True)
        ],
    }


def compare(maps: list[Path], seeds: int, budget: int, scale: float, raw: Path | None) -> dict:
    named = configs(scale)
    jobs = [
        (label, RunSpec(agent, str(m), seed, budget, settings, 25, TAU))
        for label, (agent, settings) in named.items()
        for m in maps
        for seed in range(seeds)
    ]
    with ProcessPoolExecutor(_pool_size()) as pool:  # results come back in submission order
        records = list(pool.map(e4._run_one, [spec for _, spec in jobs], chunksize=1))
    labelled = [(label, record) for (label, _), record in zip(jobs, records, strict=True)]
    if raw is not None:
        write_jsonl(
            raw / "e1_runs.jsonl", [{"label": label, **record} for label, record in labelled]
        )
    out = {}
    for label in named:
        mine = [e4._relative(r) for name, r in labelled if name == label]
        summary = summarize(mine)
        last = [r["checkpoints"][-1] for r in mine]
        out[label] = {
            "summary": summary,
            "train_failures": _stat(np.array([c["train_failures"] for c in last], dtype=float)),
            "regret": _stat(np.array([c["regret"] for c in last])),
            "p_fail": _stat(np.array([c["p_fail"] for c in last])),
            "learned_fraction": summary["learned_fraction"],
            "learned_median": summary["learned_median"],
        }
    return out


def _stat(values: np.ndarray) -> dict:
    mean, half = mean_ci(values)
    return {"mean": float(mean), "ci95": float(half)}


def run_all(quick: bool = False, raw: Path | None = None) -> dict:
    budget = BUDGET["quick" if quick else "full"]
    maps = MAPS[:2] if quick else MAPS
    seeds = 2 if quick else 5
    scale = load_scale()
    return {
        "budget": budget, "seeds_per_map": seeds, "maps": len(maps), "shaping": scale,
        "shield": SHIELD, "delta": DELTA,
        "configs": compare(maps, seeds, budget, scale, raw),
    }  # fmt: skip


FAMILY = {  # model-free in blue, model-based in orange
    "Q": style.BLUE, "SARSA": style.BLUE, "Q + shaping": style.BLUE, "Q + shield": style.BLUE,
    "Q + shield + shaping": style.BLUE, "MB-count": style.ORANGE, "MB-struct": style.ORANGE,
    "MB-struct + CMDP": style.ORANGE,
}  # fmt: skip


def plot_pareto(result: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    with style.context():
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        for label, c in result["configs"].items():
            x, y = c["train_failures"], c["regret"]
            ax.errorbar(
                x["mean"], 100 * y["mean"], xerr=np.nan_to_num(x["ci95"]),
                yerr=100 * np.nan_to_num(y["ci95"]), fmt="o", markersize=8, color=FAMILY[label],
                markeredgecolor=style.SURFACE, markeredgewidth=2, elinewidth=1.2, capsize=0,
                alpha=0.95,
            )  # fmt: skip
            ax.annotate(label, (x["mean"], 100 * y["mean"]), xytext=(7, 6),
                        textcoords="offset points", fontsize=8.5, color=style.INK_2)  # fmt: skip
        ax.set_xscale("symlog", linthresh=10)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("Rovers lost while learning (mean per run, symmetric log scale)")
        ax.set_ylabel("Final regret, % of the optimal cost (symmetric log scale)")
        ax.set_title("What it costs to learn, and what you end up with")
        ax.text(0.99, 0.97, "lower left is better", transform=ax.transAxes, ha="right",
                va="top", fontsize=8.5, color=style.MUTED)  # fmt: skip
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)


def plot_curves(result: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    with style.context():
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        for label in MAIN:
            s = result["configs"][label]["summary"]
            x = np.array(s["steps"][1:]) / 1e3
            y = 100 * np.array(s["regret"]["mean"][1:])
            ax.plot(x, y, color=FAMILY[label], linewidth=2, linestyle="-" if "+" not in label
                    else "--", label=label)  # fmt: skip
        ax.axhline(100 * TAU, color=style.MUTED, linewidth=1, linestyle=(0, (1, 3)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Environment steps (thousands)")
        ax.set_ylabel("Regret, % of the optimal cost")
        ax.set_title("Regret against training steps")
        ax.legend(loc="center right")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)


def to_markdown(result: dict) -> str:
    lines = [
        "E1: the cost of learning and what prior knowledge buys back.\n",
        f"{result['maps']} dilemma maps x {result['seeds_per_map']} seeds, "
        f"{result['budget']:,} steps each; regret is exact and relative to the optimal cost. "
        "'Learned' means regret at most 5%. Intervals are 95% across runs.\n",
        "| Configuration | Rovers lost while learning | Final regret | Final P(fail) | Learned |"
        " Median steps to learn |",
        "|---|---|---|---|---|---|",
    ]
    for label, c in result["configs"].items():
        f, r, p = c["train_failures"], c["regret"], c["p_fail"]
        median = c["learned_median"]
        lines.append(
            f"| {label} | {f['mean']:,.0f} ± {np.nan_to_num(f['ci95']):,.0f} | "
            f"{100 * r['mean']:.1f}% ± {100 * np.nan_to_num(r['ci95']):.1f} | "
            f"{100 * p['mean']:.1f}% | {100 * c['learned_fraction']:.0f}% | "
            f"{'not reached' if median is None else f'{median:,.0f}'} |"
        )
    cfg = result["configs"]
    base = cfg["Q"]["train_failures"]["mean"]
    lines.append("")
    for label in ("Q + shield + shaping", "MB-struct", "MB-struct + CMDP"):
        got = cfg[label]["train_failures"]["mean"]
        lines.append(
            f"- {label}: {100 * (1 - got / base):.1f}% fewer rovers lost than Q "
            f"({got:,.0f} against {base:,.0f})."
            if base
            else f"- {label}: {got:,.0f} rovers lost."
        )
    return "\n".join(lines) + "\n"


def default_out(quick: bool) -> Path:
    return RESULTS / "smoke" if quick else RESULTS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="few maps, steps and seeds (smoke)")
    parser.add_argument("--tune", action="store_true", help="re-tune the shaping scale first")
    parser.add_argument("--out", type=Path, help="output folder (default: results/ or smoke/)")
    args = parser.parse_args(argv)
    out = args.out or default_out(args.quick)
    if args.tune:
        chosen = tune(BUDGET["quick" if args.quick else "full"], out / "raw" / "tuning_maps")
        target = out / "e1_config.json" if args.quick else CONFIG
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(chosen, indent=2) + "\n")
        print(f"tuned shaping scale {chosen['shaping']} -> {target}")
        if args.quick:  # a smoke run must not change what the real run uses
            return
    result = run_all(args.quick, out / "raw" if args.quick else RESULTS / "raw")
    out.mkdir(parents=True, exist_ok=True)
    (out / "e1.json").write_text(json.dumps(result, indent=2) + "\n")
    table = to_markdown(result)
    (out / "e1.md").write_text(table)
    plot_pareto(result, out / "figures" / "e1_pareto.png")
    plot_curves(result, out / "figures" / "e1_curves.png")
    print(table)


if __name__ == "__main__":
    main()
