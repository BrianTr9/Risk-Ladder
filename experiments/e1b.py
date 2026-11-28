"""E1b: does the shield help once the learner is SARSA, and when does being careful cost something?

    uv run python -m experiments.e1b [--quick] [--tune] [--out DIR]

E1 compared the shield only with Q-learning, on maps where the best route already avoids the
hazard, with step sizes tuned on another map family. This follow-up closes those gaps:

1. SARSA with each prior (shaping, shield, both), next to Q and the model-based reference, with the
   step sizes and the shaping scale retuned on maps of the same family as the evaluation maps.
2. Two map families: `dilemma` (the best route is safe) and `ridge` (the best policy accepts a
   small risk, so a cautious learner pays for being cautious).
3. A sweep of the true slip probability on the dilemma maps, including values above the shield's
   assumed bound.

Writes e1b.json, e1b.md and figures/e1b_*.png (to results/smoke/ with --quick).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import e1
from experiments import plotting as style
from experiments.common import final_summary, relative, run_specs
from riskladder.env import Noise, generate, load_map, save_map
from riskladder.harness import RunSpec
from riskladder.harness.runner import write_jsonl

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results"
CONFIG = Path(__file__).parent / "e1b_config.json"
RIDGES = sorted((ROOT / "experiments" / "maps" / "ridge").glob("*.map"))
SLIPS = (0.05, 0.3)  # swept on the dilemma maps (0.15 is the main comparison)
ALPHAS = (0.1, 0.3, 0.5)
SHAPINGS = (0.25, 0.5, 1.0, 2.0)
TAU = 0.05
FAMILIES = {
    "dilemma": {
        "maps": e1.MAPS, "tune": ("dilemma", range(100, 105), {"rows": 30}),
        "budget": 1_000_000, "seeds": 5,
    },
    "ridge": {
        "maps": RIDGES, "tune": ("ridge", range(3100, 3104),
                                 {"rows": 6, "cols": 30, "noise": Noise(p_slip=0.01)}),
        "budget": 300_000, "seeds": 3,
    },
}  # fmt: skip
QUICK = {"budget": 30_000, "seeds": 2, "maps": 2}


def configs(tuned: dict) -> dict[str, tuple[str, dict]]:
    """name -> (agent, settings) for one family; `tuned` holds its step sizes and shaping scale."""
    q, sarsa, scale = tuned["q"], tuned["sarsa"], tuned["shaping"]
    return {
        "Q": ("q", dict(q)),
        "Q + shield": ("q", {**q, "shield": e1.SHIELD}),
        "SARSA": ("sarsa", dict(sarsa)),
        "SARSA + shaping": ("sarsa", {**sarsa, "shaping": scale}),
        "SARSA + shield": ("sarsa", {**sarsa, "shield": e1.SHIELD}),
        "SARSA + shield + shaping": ("sarsa", {**sarsa, "shield": e1.SHIELD, "shaping": scale}),
        "MB-struct + CMDP": ("mb_struct", {"cmdp": True, "delta": e1.DELTA}),
    }


def _run_all(jobs: list[tuple[str, RunSpec]]) -> list[tuple[str, dict]]:
    records = run_specs([spec for _, spec in jobs])
    return [(label, r) for (label, _), r in zip(jobs, records, strict=True)]


def _mean_regret(records: list[dict]) -> float:
    return float(
        np.mean([np.mean([c["regret"] / r["optimal_cost"] for c in r["checkpoints"]])
                 for r in records])
    )  # fmt: skip


def tune(family: str, budget: int, scratch: Path) -> dict:
    """Step sizes for Q and SARSA, then the shaping scale for SARSA, on maps of the same family
    that are not evaluation maps (smallest mean regret over the run)."""
    template, seeds, kwargs = FAMILIES[family]["tune"]
    scratch.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed in list(seeds)[:2] if budget < 100_000 else seeds:
        spec = generate(template, seed=seed, **kwargs)
        path = scratch / f"{family}-{spec.name}.map"
        save_map(spec, path)
        paths.append(str(path))

    def score(agent: str, settings: dict) -> float:
        jobs = [("t", RunSpec(agent, p, 0, budget, settings, 10, TAU)) for p in paths]
        return _mean_regret([r for _, r in _run_all(jobs)])

    chosen: dict = {"tuned_on": f"{family} seeds {seeds.start}-{seeds.stop - 1}", "scores": {}}
    for agent in ("q", "sarsa"):
        results = [score(agent, {"alpha": a}) for a in ALPHAS]
        chosen[agent] = {"alpha": ALPHAS[int(np.argmin(results))]}
        chosen["scores"][agent] = dict(zip(map(str, ALPHAS), results, strict=True))
    results = [score("sarsa", {**chosen["sarsa"], "shaping": k}) for k in SHAPINGS]
    chosen["shaping"] = SHAPINGS[int(np.argmin(results))]
    chosen["scores"]["shaping"] = dict(zip(map(str, SHAPINGS), results, strict=True))
    return chosen


def load_tuned() -> dict:
    if CONFIG.exists():
        return json.loads(CONFIG.read_text())
    default = {"q": {}, "sarsa": {}, "shaping": 1.0}
    return {family: default for family in FAMILIES}


def summarise(labelled: list[tuple[str, dict]], names) -> dict:
    return {
        label: final_summary([relative(r, TAU) for name, r in labelled if name == label])
        for label in names
    }


def family_runs(family: str, tuned: dict, quick: bool, raw: Path | None) -> dict:
    spec = FAMILIES[family]
    maps = spec["maps"][: QUICK["maps"] if quick else None]
    budget = QUICK["budget"] if quick else spec["budget"]
    seeds = QUICK["seeds"] if quick else spec["seeds"]
    named = configs(tuned)
    jobs = [
        (label, RunSpec(agent, str(m), seed, budget, settings, 25, TAU))
        for label, (agent, settings) in named.items()
        for m in maps
        for seed in range(seeds)
    ]
    labelled = _run_all(jobs)
    if raw is not None:
        write_jsonl(raw / f"e1b_{family}.jsonl", [{"label": k, **r} for k, r in labelled])
    return {
        "budget": budget,
        "seeds": seeds,
        "maps": len(maps),
        "configs": summarise(labelled, named),
    }


def slip_sweep(tuned: dict, quick: bool, scratch: Path) -> dict:
    """The dilemma maps under other true slip probabilities (the shield still assumes 0.4)."""
    maps = e1.MAPS[: 2 if quick else 5]
    budget = QUICK["budget"] if quick else FAMILIES["dilemma"]["budget"]
    seeds = QUICK["seeds"] if quick else 3
    names = ("Q", "SARSA", "SARSA + shield", "MB-struct + CMDP")
    named = {k: v for k, v in configs(tuned).items() if k in names}
    scratch.mkdir(parents=True, exist_ok=True)
    out = {}
    for p_slip in SLIPS:
        paths = []
        for m in maps:
            spec = load_map(m).with_noise(Noise(p_slip=p_slip))
            path = scratch / f"{p_slip}-{m.stem}.map"
            save_map(spec, path)
            paths.append(path)
        jobs = [
            (label, RunSpec(agent, str(path), seed, budget, settings, 25, TAU))
            for label, (agent, settings) in named.items()
            for path in paths
            for seed in range(seeds)
        ]
        out[str(p_slip)] = summarise(_run_all(jobs), named)
    return {"budget": budget, "seeds": seeds, "maps": len(maps), "slips": out}


def run_all(quick: bool = False, raw: Path | None = None) -> dict:
    tuned = load_tuned()
    scratch = (raw or RESULTS / "raw") / "e1b_maps"
    return {
        "tuned": tuned,
        "families": {f: family_runs(f, tuned[f], quick, raw) for f in FAMILIES},
        "slip_sweep": slip_sweep(tuned["dilemma"], quick, scratch),
    }


COLOURS = {
    "Q": style.BLUE, "Q + shield": style.BLUE, "SARSA": style.BLUE, "SARSA + shaping": style.BLUE,
    "SARSA + shield": style.BLUE, "SARSA + shield + shaping": style.BLUE,
    "MB-struct + CMDP": style.ORANGE,
}  # fmt: skip


def plot_ablation(result: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    with style.context():
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
        letters = "ABCDEFG"  # points are lettered; the key is under the figure
        for ax, family in zip(axes, FAMILIES, strict=True):
            for i, (label, c) in enumerate(result["families"][family]["configs"].items()):
                x, y = c["train_failures"], c["regret"]
                spread = 100 * np.array([[y["median"] - y["q1"]], [y["q3"] - y["median"]]])
                ax.errorbar(
                    x["mean"], 100 * y["median"], xerr=np.nan_to_num(x["ci95"]),
                    yerr=spread, fmt="o", markersize=8,
                    color=COLOURS[label], markeredgecolor=style.SURFACE, markeredgewidth=2,
                    elinewidth=1.2, capsize=0,
                )  # fmt: skip
                ax.annotate(letters[i], (x["mean"], 100 * y["median"]), xytext=(8, 6),
                            textcoords="offset points", fontsize=9, color=style.INK)  # fmt: skip
            ax.set_xscale("symlog", linthresh=10)
            ax.set_yscale("symlog", linthresh=1)
            ax.set_title(f"{family} maps")
            ax.set_xlabel("Rovers lost while learning (mean per run)")
        axes[0].set_ylabel("Final regret, median and quartiles (% of the optimal cost)")
        names = list(result["families"]["dilemma"]["configs"])
        key = "    ".join(f"{letters[i]} = {name}" for i, name in enumerate(names))
        fig.text(0.5, 0.01, key, ha="center", fontsize=8.5, color=style.INK_2)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)


def _row(label: str, c: dict) -> str:
    f, r, p = c["train_failures"], c["regret"], c["p_fail"]
    return (
        f"| {label} | {f['mean']:,.0f} ± {np.nan_to_num(f['ci95']):,.0f} | "
        f"{100 * r['median']:.1f}% [{100 * r['q1']:.1f} to {100 * r['q3']:.1f}] "
        f"(mean {100 * r['mean']:.0f}%) | {c['looping_runs']} of {c['runs']} | "
        f"{100 * p['mean']:.1f}% (optimum {100 * c['optimal_p_fail']:.1f}%) | "
        f"{100 * c['learned_fraction']:.0f}% |"
    )


HEADER = [
    "| Configuration | Rovers lost while learning | Final regret: median [quartiles] (mean) |"
    " Runs that loop | Final P(fail) | Learned |",
    "|---|---|---|---|---|---|",
]


def to_markdown(result: dict) -> str:
    lines = [
        "E1b: does the shield help SARSA, and what does caution cost?\n",
        "Final regret is a fraction of the optimal cost and is heavy-tailed (a policy that loops "
        "pays until the horizon), so the median and quartiles come first. A run 'loops' when its "
        "final policy times out with probability above 0.5%.\n",
    ]
    for family, data in result["families"].items():
        t = result["tuned"][family]
        lines += [
            f"{family} maps: {data['maps']} maps x {data['seeds']} seeds, "
            f"{data['budget']:,} steps; step sizes Q {t['q'].get('alpha', 'default')}, "
            f"SARSA {t['sarsa'].get('alpha', 'default')}, shaping scale {t['shaping']} "
            f"(tuned on {t.get('tuned_on', 'nothing: defaults')}).\n",
            *HEADER,
            *(_row(k, c) for k, c in data["configs"].items()),
            "",
        ]
    sweep = result["slip_sweep"]
    lines.append(
        f"Slip sweep on {sweep['maps']} dilemma maps x {sweep['seeds']} seeds, "
        f"{sweep['budget']:,} steps (the shield still assumes at most 0.4 for any action):\n"
    )
    for p_slip, configs_ in sweep["slips"].items():
        lines += [f"True slip {p_slip}:\n", *HEADER, *(_row(k, c) for k, c in configs_.items()), ""]
    return "\n".join(lines)


def default_out(quick: bool) -> Path:
    return RESULTS / "smoke" if quick else RESULTS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="few maps, steps and seeds (smoke)")
    parser.add_argument("--tune", action="store_true", help="retune step sizes and shaping scale")
    parser.add_argument("--out", type=Path, help="output folder (default: results/ or smoke/)")
    args = parser.parse_args(argv)
    out = args.out or default_out(args.quick)
    if args.tune:
        chosen = {
            f: tune(
                f, QUICK["budget"] if args.quick else FAMILIES[f]["budget"], out / "raw" / "tune"
            )
            for f in FAMILIES
        }
        target = out / "e1b_config.json" if args.quick else CONFIG
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(chosen, indent=2) + "\n")
        print(f"tuned settings -> {target}")
        if args.quick:  # a smoke run must not change what the real run uses
            return
    result = run_all(args.quick, out / "raw" if args.quick else RESULTS / "raw")
    out.mkdir(parents=True, exist_ok=True)
    (out / "e1b.json").write_text(json.dumps(result, indent=2) + "\n")
    table = to_markdown(result)
    (out / "e1b.md").write_text(table)
    plot_ablation(result, out / "figures" / "e1b_ablation.png")
    print(table)


if __name__ == "__main__":
    main()
