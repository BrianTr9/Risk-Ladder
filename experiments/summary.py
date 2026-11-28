"""Write `SUMMARY.md` from the result files, so no number in it is typed by hand.

    uv run python -m experiments.summary [--results DIR]

Each section says which question a figure answers, gives the few numbers that carry the answer, and
links the figure. Four figures are headlines (one per claim or sub-claim); the rest are supporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).parents[1] / "results"


def _load(results: Path, name: str) -> dict | None:
    path = results / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def _e5(d: dict) -> list[str]:
    at = {c["p_slip"]: c for c in d["curve"]}
    near = min(at, key=lambda p: abs(p - 0.15))  # the default slip, or the closest swept value
    c = at[near]
    return [
        f"At slip {near:g} the noise-free A* plan loses the rover with probability "
        f"{_pct(c['p_fail']['mean'])} (maps range {_pct(c['p_fail']['min'])} to "
        f"{_pct(c['p_fail']['max'])}) and costs a median {_pct(c['regret']['median'], 0)} more "
        f"than the optimum, which loses it with probability {_pct(c['optimal_p_fail'])}."
    ]


def _e3(d: dict) -> list[str]:
    lines = []
    for slip, s in d["spread"]["summary"].items():
        lines.append(
            f"At slip {slip}, one fixed penalty breaks the {_pct(d['spread']['bound'], 0)} bound "
            f"on {s['penalty_violates_bound']} of {s['maps']} maps "
            f"(worst {_pct(s['penalty_max_p_fail'], 2)}); the constrained solver breaks it on "
            f"{s['cmdp_violates_bound']} (worst {_pct(s['cmdp_max_p_fail'], 2)})."
        )
    return lines


def _e2(d: dict) -> list[str]:
    worst = max(d["grid"]["cells"], key=lambda c: c["mean_violation"])
    two = next(r for r in d["margin"]["rows"] if r["margin"] == 2.0)
    return [
        f"A planner that believes the slip is {worst['true'] / worst['believed']:.0f}x lower "
        f"than it is exceeds the {_pct(d['bound'], 0)} bound by "
        f"{100 * worst['mean_violation']:.1f} percentage points on average in the worst cell "
        "of the grid.",
        f"Planning with a 2x margin removes the excess for a 2x underestimate and costs "
        f"{two['right_belief_mean_extra_cost']:.2f} extra when the belief was right.",
    ]


def _e1(d: dict) -> list[str]:
    c = d["configs"]
    lost = {k: v["train_failures"]["mean"] for k, v in c.items()}
    shaping = "more" if lost["Q + shaping"] > lost["Q"] else "no more"
    return [
        f"Rovers lost while learning (mean per run): Q {lost['Q']:,.0f}, "
        f"SARSA {lost['SARSA']:,.0f}, Q + shield {lost['Q + shield']:,.0f}, "
        f"MB-count {lost['MB-count']:,.0f}, MB-struct {lost['MB-struct']:,.0f}, "
        f"MB-struct + CMDP {lost['MB-struct + CMDP']:,.0f}.",
        f"Q + shaping lost {lost['Q + shaping']:,.0f}, {shaping} than Q.",
        "Final median regret: "
        + ", ".join(f"{k} {_pct(v['regret']['median'])}" for k, v in c.items())
        + ".",
    ]


def _e1b(d: dict) -> list[str]:
    ridge = d["families"]["ridge"]["configs"]
    s, sh = ridge["SARSA"], ridge["SARSA + shield"]
    return [
        f"On ridge maps SARSA's final policy loops in {s['looping_runs']} of {s['runs']} runs "
        f"and loses {s['train_failures']['mean']:,.0f} rovers; with the shield, "
        f"{sh['looping_runs']} of {sh['runs']} loop and "
        f"{sh['train_failures']['mean']:,.0f} are lost."
    ]


def _e1c(d: dict) -> list[str]:
    lost = {k: v["train_failures"]["mean"] for k, v in d["configs"].items()}
    return [
        f"Rovers lost while learning: no shaping {lost['no shaping']:,.0f}, potential along the "
        f"shortest route {lost['shortest route, scale 2']:,.0f}, potential along a route clear of "
        f"hazards {lost['route clear of hazards, scale 2']:,.0f} (scale 2 each)."
    ]


def _e4(d: dict) -> list[str]:
    g, m = d["generalisation"], d["match"]
    r = g["final"]["regret"]
    steps = "not within the budget" if m["median_steps"] is None else f"{m['median_steps']:,.0f}"
    return [
        f"Trained on other maps and given no samples on the new one, the linear learner's median "
        f"regret is {_pct(r['median'])} (quartiles {_pct(r['q1'])} to {_pct(r['q3'])}).",
        f"Q-learning matches it per map within {m['budget']:,} steps on {m['matched_maps']} of "
        f"{m['maps']} maps; median steps to match: {steps}.",
    ]


def _e0(d: dict) -> list[str]:
    r = min(d["rows"], key=lambda row: row["p_fail_exact"])
    return [
        f"At P(fail) = {r['p_fail_exact']:.3g} the exact evaluation takes {r['oracle_ms']:.2f} ms; "
        f"a Monte Carlo estimate to +-10% needs {r['rollouts_needed']:,} rollouts, "
        f"{r['speedup']:,.0f}x slower, and its interval covers the truth "
        f"{_pct(r['coverage_within_10pct'], 0)} of the time."
    ]


def _bench(d: dict) -> list[str]:
    r = d["rows"][-1]
    return [
        f"The largest grid, {r['grid']}x{r['grid']} with {r['states']:,} states, is solved by "
        f"value iteration in {r['vi_s']:.1f} s."
    ]


# (key, title, question, figure or None, writer, headline?)
SECTIONS = [
    ("e3", "Penalty versus constraint (E3)",
     "Does one fixed failure penalty give the same risk everywhere?",
     "e3_risk_spread.png", _e3, True),
    ("e2", "A wrong slip estimate (E2)",
     "How much does a wrong model break the risk guarantee, and what does a margin buy?",
     "e2_misspecification.png", _e2, True),
    ("e1", "What prior knowledge buys back (E1)",
     "How many rovers does learning lose, and what does each kind of prior knowledge change?",
     "e1_pareto.png", _e1, True),
    ("e4", "Generalisation to new maps (E4)",
     "Do map-only features carry over to maps the learner never saw?",
     "e4_generalisation.png", _e4, True),
    ("e5", "The first rung: A* (E5)",
     "What does planning for a rover that never slips cost when it does?",
     "e5_astar.png", _e5, False),
    ("e0", "Exact versus Monte Carlo (E0)",
     "Is exact evaluation worth having?", None, _e0, False),
    ("bench", "Oracle latency", "How large a map can the oracle solve?",
     "bench_latency.png", _bench, False),
    ("e1c", "Which potential shaping follows (E1c)",
     "Why did shaping lose rovers, and does a different potential fix it?",
     "e1c_potential.png", _e1c, False),
    ("e1b", "Shield on SARSA (E1b)", "Does the shield help once the learner is SARSA?",
     "e1b_ablation.png", _e1b, False),
]  # fmt: skip
EXTRA_FIGURES = ["e3_frontier.png", "e1_curves.png", "e4_curves.png"]


def build(results: Path = RESULTS) -> str:
    lines = [
        "# Risk-Bounded Autonomy: results summary\n",
        "Generated by `uv run python -m experiments.summary`; every number comes from the "
        "`results/*.json` files written by the experiments.\n",
    ]
    for headline in (True, False):
        lines.append("## Headline figures\n" if headline else "## Supporting results\n")
        for key, title, question, figure, writer, is_headline in SECTIONS:
            if is_headline != headline:
                continue
            data = _load(results, key)
            lines += [f"### {title}\n", f"*{question}*\n"]
            if data is None:
                lines += [f"Not generated: `{key}.json` is missing.\n"]
                continue
            lines += [f"- {text}" for text in writer(data)] + [""]
            if figure and (results / "figures" / figure).exists():
                lines += [f"![{title}](figures/{figure})\n"]
    extras = [f for f in EXTRA_FIGURES if (results / "figures" / f).exists()]
    if extras:
        lines.append("Further figures: " + ", ".join(f"`figures/{f}`" for f in extras) + ".\n")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    text = build(args.results)
    (args.results / "SUMMARY.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
