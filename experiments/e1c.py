"""E1c: why shaping lost rovers in E1, and what fixes it.

    uv run python -m experiments.e1c [--quick] [--out DIR]

Potential-based shaping keeps the optimal policy whatever the potential (the oracle checks this), so
the potential cannot change what is best; it changes what a learner does while it explores. In E1
the potential was the distance along the shortest hazard-free route. On the `dilemma` maps that
route runs beside the hazard strip, so a learner pulled along it loses more rovers. The same
shaping with the distance along a route that keeps one cell clear of hazards does the opposite.
Scales are not tuned here: each is shown.

Writes e1c.json, e1c.md and figures/e1c_potential.png (to results/smoke/ with --quick).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import e1, e4
from experiments import plotting as style
from experiments.common import final_summary, relative, run_specs
from riskladder.harness import RunSpec
from riskladder.harness.runner import write_jsonl

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results"
TAU = 0.05
BUDGET = {"full": 1_000_000, "quick": 40_000}
SEEDS = {"full": 5, "quick": 2}
# label -> (shaping scale, route); None means no shaping
CONFIGS = {
    "no shaping": (None, "free"),
    "shortest route, scale 1": (1.0, "free"),
    "shortest route, scale 2": (2.0, "free"),
    "route clear of hazards, scale 1": (1.0, "safe"),
    "route clear of hazards, scale 2": (2.0, "safe"),
}


def run(quick: bool = False, raw: Path | None = None) -> dict:
    mode = "quick" if quick else "full"
    q = e4.load_config().get("q", {})
    maps = e1.MAPS[:2] if quick else e1.MAPS
    jobs = [
        (
            label,
            RunSpec(
                "q", str(m), seed, BUDGET[mode],
                {**q, **({} if scale is None else {"shaping": scale, "shaping_route": route})},
                25, TAU,
            ),
        )
        for label, (scale, route) in CONFIGS.items()
        for m in maps
        for seed in range(SEEDS[mode])
    ]  # fmt: skip
    records = run_specs([spec for _, spec in jobs])
    labelled = [(label, r) for (label, _), r in zip(jobs, records, strict=True)]
    if raw is not None:
        write_jsonl(raw / "e1c_runs.jsonl", [{"label": k, **r} for k, r in labelled])
    configs = {
        label: final_summary([relative(r, TAU) for name, r in labelled if name == label])
        for label in CONFIGS
    }
    early = {
        label: float(
            np.mean(
                [r["checkpoints"][2]["train_failures"] for name, r in labelled if name == label]
            )
        )
        for label in CONFIGS
    }  # failures within the first 8% of the budget (checkpoint 2 of 25)
    return {
        "budget": BUDGET[mode], "seeds_per_map": SEEDS[mode], "maps": len(maps),
        "configs": configs, "early_failures": early,
    }  # fmt: skip


def plot(result: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    with style.context():
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        labels = list(result["configs"])
        mean = np.array([result["configs"][k]["train_failures"]["mean"] for k in labels])
        half = np.nan_to_num([result["configs"][k]["train_failures"]["ci95"] for k in labels])
        colours = [style.BLUE if "clear" not in k else style.ORANGE for k in labels]
        y = np.arange(len(labels))[::-1]
        ax.barh(y, mean, xerr=half, color=colours, height=0.6, error_kw={"elinewidth": 1.2})
        ax.set_yticks(y, labels)
        ax.set_xlabel("Rovers lost while learning (mean per run, 95% interval)")
        ax.set_title("What the potential teaches while the learner explores")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)


def to_markdown(result: dict) -> str:
    lines = [
        "E1c: the route a shaping potential follows decides how many rovers the learner loses.\n",
        f"Q-learning on {result['maps']} dilemma maps x {result['seeds_per_map']} seeds, "
        f"{result['budget']:,} steps. Same step size as E1; the scales are not tuned.\n",
        "| Shaping | Rovers lost while learning | Lost in the first 8% |"
        " Final regret: median [quartiles] | Runs that loop |",
        "|---|---|---|---|---|",
    ]
    for label, c in result["configs"].items():
        f, r = c["train_failures"], c["regret"]
        lines.append(
            f"| {label} | {f['mean']:,.0f} ± {np.nan_to_num(f['ci95']):,.0f} | "
            f"{result['early_failures'][label]:,.0f} | {100 * r['median']:.1f}% "
            f"[{100 * r['q1']:.1f} to {100 * r['q3']:.1f}] | {c['looping_runs']} of {c['runs']} |"
        )
    return "\n".join(lines) + "\n"


def default_out(quick: bool) -> Path:
    return RESULTS / "smoke" if quick else RESULTS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="two maps, fewer steps (smoke)")
    parser.add_argument("--out", type=Path, help="output folder (default: results/ or smoke/)")
    args = parser.parse_args(argv)
    out = args.out or default_out(args.quick)
    out.mkdir(parents=True, exist_ok=True)
    result = run(args.quick, out / "raw")
    (out / "e1c.json").write_text(json.dumps(result, indent=2) + "\n")
    table = to_markdown(result)
    (out / "e1c.md").write_text(table)
    plot(result, out / "figures" / "e1c_potential.png")
    print(table)


if __name__ == "__main__":
    main()
