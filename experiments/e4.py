"""E4: learning from samples, and what prior knowledge buys on maps never seen.

    uv run python -m experiments.e4 [--quick] [--tune] [--out DIR]

Tabular Q-learning, SARSA and a linear Q-learner on map-only features learn from sampled steps
alone. Every checkpoint of every run is scored *exactly* by the oracle (regret against the
optimum over the same horizon), so the curves have no evaluation noise.

1. Curves. Each agent learns each of the 20 held-out `field/test` maps from scratch.
2. Generalisation. The linear learner trains across the 20 `field/train` maps (one map per
   episode) and is then scored on the held-out maps with no further training. The tabular
   learner has to pay for samples on each map; the table says how many it needs to match.

Learner settings come from `experiments/e4_config.json`, which `--tune` rewrites from the
training maps only; the held-out maps are never used for tuning.

Writes e4.json, e4.md and figures/e4_*.png (to results/smoke/ with --quick); the raw per-run
records go to results/raw/ (not committed).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments import plotting as style
from experiments.common import LOOP, final_summary, pool_size, relative, run_one, stat
from riskladder.agents.linear import LinearQ
from riskladder.env import RoverEnv, load_map
from riskladder.harness import RunSpec, Scorer, summarize, train
from riskladder.harness.runner import mean_ci, write_jsonl

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results"
CONFIG = Path(__file__).parent / "e4_config.json"
TRAIN_MAPS = sorted((ROOT / "experiments" / "maps" / "field" / "train").glob("*.map"))
TEST_MAPS = sorted((ROOT / "experiments" / "maps" / "field" / "test").glob("*.map"))
AGENTS = ("q", "sarsa", "linear")
LABELS = {"q": "Q-learning", "sarsa": "SARSA", "linear": "Linear Q (one map)"}
TAU = 0.05

# Settings tried by --tune (step size, and how fast it decays with the number of visits / steps).
GRID = {
    "q": [{"alpha": a, "alpha_decay": d} for a in (0.1, 0.3, 0.5) for d in (0.0, 0.05)],
    "sarsa": [{"alpha": a, "alpha_decay": d} for a in (0.1, 0.3, 0.5) for d in (0.0, 0.05)],
    "linear": [{"alpha": a, "alpha_decay": d} for a in (0.02, 0.05, 0.1, 0.3) for d in (0.0, 1e-5)],
}
SCALE = {  # (steps per tabular run, steps training the linear learner across maps)
    "full": (3_000_000, 1_000_000),
    "quick": (60_000, 40_000),
}


def load_config() -> dict:
    return json.loads(CONFIG.read_text()) if CONFIG.exists() else {a: {} for a in AGENTS}


def _train_linear_across_maps(job: tuple[int, dict, int, list[str], list[str]]) -> dict:
    """Train a linear learner over several maps and score it exactly on others."""
    seed, config, budget, train_paths, score_paths = job
    env_seed, agent_seed = np.random.SeedSequence(seed).spawn(2)
    seeds = env_seed.spawn(len(train_paths))
    envs = [
        RoverEnv(load_map(p), seed=s).sample_interface()
        for p, s in zip(train_paths, seeds, strict=True)
    ]
    scorers = [Scorer(RoverEnv(load_map(p)), TAU) for p in score_paths]
    checkpoints = train(
        LinearQ(**config), envs, budget, np.random.default_rng(agent_seed), scorers, 25
    )
    return {
        "seed": seed, "config": config, "budget": budget,
        "optimal_cost": [s.optimal.cost for s in scorers],
        "checkpoints": [
            {
                "steps": cp.steps, "train_failures": cp.train_failures,
                "regret": [
                    sc.regret / s.optimal.cost for sc, s in zip(cp.scores, scorers, strict=True)
                ],
                "p_fail": [sc.p_fail for sc in cp.scores],
                "p_timeout": [sc.p_timeout for sc in cp.scores],
            }
            for cp in checkpoints
        ],
    }  # fmt: skip


def tune(quick: bool = False) -> dict:
    """Pick, per agent, the setting with the smallest mean regret over the run on training maps."""
    tabular_budget, linear_budget = SCALE["quick" if quick else "full"]
    tune_maps = [str(p) for p in TRAIN_MAPS[: 2 if quick else 5]]
    chosen: dict = {"tuned_on": "field/train", "scores": {}}
    with ProcessPoolExecutor(pool_size()) as pool:
        for agent in ("q", "sarsa"):
            grid = GRID[agent][:2] if quick else GRID[agent]
            specs = [
                RunSpec(agent, m, 0, tabular_budget, config, 10, TAU)
                for config in grid
                for m in tune_maps
            ]
            records = list(pool.map(run_one, specs, chunksize=1))
            scores = []
            for i in range(len(grid)):
                part = records[i * len(tune_maps) : (i + 1) * len(tune_maps)]
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
            chosen[agent] = grid[int(np.argmin(scores))]
            chosen["scores"][agent] = [
                {"config": g, "mean_regret": s} for g, s in zip(grid, scores, strict=True)
            ]
        grid = GRID["linear"][:2] if quick else GRID["linear"]
        paths = [str(p) for p in TRAIN_MAPS[: 4 if quick else None]]
        jobs = [(0, config, linear_budget, paths, paths) for config in grid]
        records = list(pool.map(_train_linear_across_maps, jobs, chunksize=1))
        scores = [float(np.mean([np.mean(c["regret"]) for c in r["checkpoints"]])) for r in records]
        chosen["linear"] = grid[int(np.argmin(scores))]
        chosen["scores"]["linear"] = [
            {"config": g, "mean_regret": s} for g, s in zip(grid, scores, strict=True)
        ]
    return chosen


def curves(config: dict, maps: list[Path], seeds: int, budget: int, raw: Path | None) -> dict:
    specs = [
        RunSpec(agent, str(m), seed, budget, config.get(agent, {}), 25, TAU)
        for agent in AGENTS
        for m in maps
        for seed in range(seeds)
    ]
    with ProcessPoolExecutor(pool_size()) as pool:
        records = list(pool.map(run_one, specs, chunksize=1))
    if raw is not None:
        write_jsonl(raw, records)
    out = {}
    for agent in AGENTS:
        mine = [relative(r, TAU) for r in records if r["agent"] == agent]
        summary = summarize(mine)
        summary["final"] = {k: v for k, v in final_summary(mine).items() if k != "curves"}
        summary["per_map_final_regret"] = {
            m.stem: float(
                np.mean([r["checkpoints"][-1]["regret"] for r in mine if r["map"] == m.stem])
            )
            for m in maps
        }
        summary["per_map_regret"] = {
            m.stem: np.mean(
                [[c["regret"] for c in r["checkpoints"]] for r in mine if r["map"] == m.stem],
                axis=0,
            ).tolist()
            for m in maps
        }
        out[agent] = summary
    return out


def generalisation(
    config: dict, train_maps: list[Path], test_maps: list[Path], seeds: int, budget: int,
    raw: Path | None,
) -> dict:  # fmt: skip
    jobs = [
        (seed, config.get("linear", {}), budget, [str(p) for p in train_maps],
         [str(p) for p in test_maps])
        for seed in range(seeds)
    ]  # fmt: skip
    with ProcessPoolExecutor(pool_size()) as pool:
        records = list(pool.map(_train_linear_across_maps, jobs, chunksize=1))
    if raw is not None:
        write_jsonl(raw, records)
    steps = [c["steps"] for c in records[0]["checkpoints"]]
    regret = np.array([[c["regret"] for c in r["checkpoints"]] for r in records])  # seed,ckpt,map
    p_fail = np.array([[c["p_fail"] for c in r["checkpoints"]] for r in records])
    timeout = np.array([[c["p_timeout"] for c in r["checkpoints"]] for r in records])
    mean_regret, half = mean_ci(regret.mean(axis=2))
    return {
        "seeds": seeds, "train_steps": budget, "steps": steps, "maps": [m.stem for m in test_maps],
        "regret_mean": mean_regret.tolist(), "regret_ci95": half.tolist(),
        "final_regret_per_map": regret[:, -1, :].mean(axis=0).tolist(),
        "final_p_fail_per_map": p_fail[:, -1, :].mean(axis=0).tolist(),
        "final_p_timeout_per_map": timeout[:, -1, :].mean(axis=0).tolist(),
        "final": {
            "regret": stat(regret[:, -1, :].ravel()),
            "looping_runs": int((timeout[:, -1, :] > LOOP).sum()),
            "runs": int(regret[:, -1, :].size),
        },
    }  # fmt: skip


def steps_to_match(per_map_curve: dict, steps: list[int], target: list[float], maps: list[str]):
    """For each map, the first checkpoint where the tabular curve is at or below the zero-shot
    linear regret on that map; None where it never gets there within the budget."""
    out = []
    for m, goal in zip(maps, target, strict=True):
        curve = per_map_curve[m]
        out.append(next((s for s, r in zip(steps, curve, strict=True) if r <= goal), None))
    return out


def match_table(curve_summary: dict, general: dict) -> dict:
    steps = curve_summary["q"]["steps"]
    maps = general["maps"]
    target = general["final_regret_per_map"]
    matched = steps_to_match(curve_summary["q"]["per_map_regret"], steps, target, maps)
    done = [m for m in matched if m is not None]
    return {
        "per_map": dict(zip(maps, matched, strict=True)),
        "matched_maps": len(done),
        "maps": len(maps),
        "median_steps": float(np.median(done)) if len(done) * 2 >= len(maps) else None,
        "budget": steps[-1],
    }


def plot_curves(summary: dict, general: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    def band(ax, agent, colour, label):
        s = summary[agent]
        x = np.array(s["steps"]) / 1e6
        y = np.array(s["regret"]["mean"])
        ci = np.nan_to_num(np.array(s["regret"]["ci95"]))
        ax.plot(x, y, color=colour, linewidth=2, label=label)
        ax.fill_between(x, np.maximum(y - ci, 1e-4), y + ci, color=colour, alpha=0.18, linewidth=0)

    with style.context():
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
        band(axes[0], "q", style.BLUE, LABELS["q"])
        band(axes[0], "sarsa", style.ORANGE, LABELS["sarsa"])
        axes[0].set_title("Two tabular learners")
        band(axes[1], "q", style.BLUE, LABELS["q"])
        band(axes[1], "linear", style.ORANGE, LABELS["linear"])
        zero = np.mean(general["final_regret_per_map"])
        axes[1].axhline(zero, color=style.INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
        axes[1].text(
            0.02, zero * 1.12, "linear Q trained on other maps, no samples here",
            transform=axes[1].get_yaxis_transform(), fontsize=8.5, color=style.INK_2,
        )  # fmt: skip
        axes[1].set_title("Tabular against linear features")
        for ax in axes:
            ax.axhline(TAU, color=style.MUTED, linewidth=1, linestyle=(0, (1, 3)))
            ax.set_yscale("log")
            ax.set_xlabel("Environment steps on the map (millions)")
            ax.legend(loc="upper right")
        axes[0].set_ylabel("Regret, fraction of the optimal cost")
        axes[0].text(0.02, TAU * 1.15, "5%: learned", transform=axes[0].get_yaxis_transform(),
                     fontsize=8, color=style.INK_2)  # fmt: skip
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)


def plot_generalisation(general: dict, match: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    with style.context():
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
        x = np.array(general["steps"]) / 1e6
        y = np.array(general["regret_mean"])
        ci = np.nan_to_num(np.array(general["regret_ci95"]))
        axes[0].plot(x, y, color=style.BLUE, linewidth=2)
        axes[0].fill_between(x, y - ci, y + ci, color=style.BLUE, alpha=0.18, linewidth=0)
        axes[0].set_xlabel("Training steps, spread over the 20 training maps (millions)")
        axes[0].set_ylabel("Regret on the 20 held-out maps (fraction of optimum)")
        axes[0].set_title("Held-out regret as training proceeds")
        axes[0].set_yscale("log")
        steps = [s for s in match["per_map"].values()]
        budget = match["budget"]
        shown = np.array([budget if s is None else s for s in steps], dtype=float) / 1e6
        reached = np.array([s is not None for s in steps])
        order = np.argsort(shown)
        axes[1].plot(
            np.arange(len(order))[reached[order]], shown[order][reached[order]], linestyle="none",
            marker="o", markersize=8, color=style.BLUE, markeredgecolor=style.SURFACE,
            markeredgewidth=2,
        )  # fmt: skip
        if (~reached).any():
            axes[1].plot(
                np.arange(len(order))[~reached[order]], shown[order][~reached[order]],
                linestyle="none", marker="^", markersize=8, color=style.ORANGE,
                markeredgecolor=style.SURFACE, markeredgewidth=2,
                label="not matched within the budget",
            )  # fmt: skip
            axes[1].legend(loc="lower right")
        axes[1].set_xlabel("Held-out maps, sorted")
        axes[1].set_ylabel("Q-learning steps needed to match (millions)")
        axes[1].set_title("Samples a tabular learner needs per map")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)


def to_markdown(result: dict) -> str:
    s, g, m = result["curves"], result["generalisation"], result["match"]
    lines = [
        "E4: learning from samples; regret is exact (oracle) and relative to the optimal cost.\n",
        f"Each agent learns each of {len(g['maps'])} held-out maps from scratch "
        f"({result['seeds_per_map']} seeds per map, {s['q']['steps'][-1]:,} steps). Regret is "
        "heavy-tailed (a policy that loops pays until the horizon), so the median and quartiles "
        "are shown next to the mean. A run 'loops' when its final policy times out with "
        "probability above 0.5%. 'Learned' means regret at most 5% of the optimum.\n",
        "| Agent | Regret at the end: median [quartiles] (mean) | Runs that loop | P(fail) |"
        " Learned | Median steps to learn | Failures while training |",
        "|---|---|---|---|---|---|---|",
    ]
    for agent in AGENTS:
        a = s[agent]
        f = a["final"]
        r = f["regret"]
        median = a["learned_median"]
        lines.append(
            f"| {LABELS[agent]} | {100 * r['median']:.1f}% "
            f"[{100 * r['q1']:.1f} to {100 * r['q3']:.1f}] "
            f"(mean {100 * r['mean']:.0f}%) | {f['looping_runs']} of {f['runs']} | "
            f"{100 * f['p_fail']['mean']:.1f}% | {100 * a['learned_fraction']:.0f}% | "
            f"{'not reached' if median is None else f'{median:,.0f}'} | "
            f"{f['train_failures']['mean']:,.0f} |"
        )
    r = g["final"]["regret"]
    median_text = (
        "not reached on at least half of the maps"
        if m["median_steps"] is None
        else f"{m['median_steps']:,.0f}"
    )
    lines += [
        f"\nGeneralisation. A linear learner trained for {g['train_steps']:,} steps across the 20 "
        f"training maps ({g['seeds']} seeds), then scored with no further samples on the held-out "
        f"maps: regret {100 * r['median']:.1f}% [{100 * r['q1']:.1f} to {100 * r['q3']:.1f}] "
        f"(mean {100 * r['mean']:.0f}%), {g['final']['looping_runs']} of {g['final']['runs']} "
        f"policies loop, P(fail) {100 * float(np.mean(g['final_p_fail_per_map'])):.1f}%.\n",
        f"Q-learning matches that per-map regret within {m['budget']:,} steps on "
        f"{m['matched_maps']} of {m['maps']} maps; median steps to match: "
        f"{median_text}.",
    ]
    return "\n".join(lines) + "\n"


def run_all(quick: bool = False, raw: Path | None = None) -> dict:
    tabular_budget, linear_budget = SCALE["quick" if quick else "full"]
    config = load_config()
    test_maps = TEST_MAPS[:3] if quick else TEST_MAPS
    train_maps = TRAIN_MAPS[:4] if quick else TRAIN_MAPS
    seeds = 2 if quick else 3
    summary = curves(config, test_maps, seeds, tabular_budget, raw and raw / "e4_curves.jsonl")
    general = generalisation(
        config, train_maps, test_maps, 2 if quick else 5, linear_budget,
        raw and raw / "e4_linear_across_maps.jsonl",
    )  # fmt: skip
    return {
        "config": config, "seeds_per_map": seeds, "curves": summary, "generalisation": general,
        "match": match_table(summary, general),
    }  # fmt: skip


def default_out(quick: bool) -> Path:
    return RESULTS / "smoke" if quick else RESULTS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="few maps, steps and seeds (smoke)")
    parser.add_argument("--tune", action="store_true", help="re-tune on the training maps first")
    parser.add_argument("--out", type=Path, help="output folder (default: results/ or smoke/)")
    args = parser.parse_args(argv)
    out = args.out or default_out(args.quick)
    if args.tune:
        chosen = tune(args.quick)
        target = out / "e4_config.json" if args.quick else CONFIG
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(chosen, indent=2) + "\n")
        print(f"tuned settings -> {target}")
        if args.quick:  # a smoke run must not change what the real run uses
            return
    result = run_all(args.quick, RESULTS / "raw" if not args.quick else out / "raw")
    out.mkdir(parents=True, exist_ok=True)
    (out / "e4.json").write_text(json.dumps(result, indent=2) + "\n")
    table = to_markdown(result)
    (out / "e4.md").write_text(table)
    plot_curves(result["curves"], result["generalisation"], out / "figures" / "e4_curves.png")
    plot_generalisation(
        result["generalisation"], result["match"], out / "figures" / "e4_generalisation.png"
    )
    print(table)


if __name__ == "__main__":
    main()
