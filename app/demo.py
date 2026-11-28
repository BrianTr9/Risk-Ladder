"""What the demo computes, with no web framework in sight (`app/ui.py` only lays it out).

Every answer comes from `riskladder.service`, the same code the API runs, so the demo cannot drift
from it. Figures are drawn with the shared renderer in `riskladder.viz`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # the server has no display
import numpy as np
from matplotlib.figure import Figure

from riskladder import service
from riskladder.env import Noise, format_map, generate, parse_map
from riskladder.oracle import evaluate
from riskladder.service import ServiceError
from riskladder.viz import plot_map

ROOT = Path(__file__).parents[1]
MAPS = ROOT / "experiments" / "maps"
RESULTS = Path(os.environ.get("RISKLADDER_RESULTS", ROOT / "results"))
EXAMPLES = {
    "ridge, strips beside the route (6x30)": MAPS / "ridge" / "ridge-6x30-s3000.map",
    "ridge, another seed": MAPS / "ridge" / "ridge-6x30-s3003.map",
    "dilemma, corridor or detour (30x30)": MAPS / "dilemma" / "dilemma-30x30-s0.map",
    "dilemma, another seed": MAPS / "dilemma" / "dilemma-30x30-s3.map",
}
TEMPLATES = {  # what the generator needs for each template, and the slip the experiments use
    "ridge": {"rows": 6, "cols": 30, "noise": Noise(p_slip=0.01)},
    "dilemma": {"rows": 30},
    "field": {"rows": 30},
}
HEADLINES = ("e3_risk_spread", "e2_misspecification", "e1_pareto", "e4_generalisation")


def _with_slip(text: str) -> tuple[str, float]:
    """A map and its own slip probability, to preset the sliders."""
    return text, parse_map(text).noise.p_slip


def example_view(name: str) -> tuple[str, float, float]:
    text, slip = _with_slip(EXAMPLES[name].read_text())
    return text, slip, slip


def generated_view(template: str, seed: int) -> tuple[str, float, float]:
    text, slip = _with_slip(format_map(generate(template, seed=int(seed), **TEMPLATES[template])))
    return text, slip, slip


def _grids(text: str, result: service.PlanResult, p_slip: float):
    """Per-cell failure probability of a plan (mixtures averaged by their weights) and the policy
    of its heaviest part, as grids for the renderer."""
    _, mdp = service.load_model(text, p_slip)
    risk = np.zeros(mdp.n_states)
    for part in result.components:
        policy = service.from_grid(mdp, part.actions)
        risk += part.weight * evaluate(mdp, policy, mdp.horizon).p_fail
    heavy = max(result.components, key=lambda c: c.weight)
    return mdp.to_grid(risk, fill=np.nan), np.array(heavy.actions)


def _draw(fig: Figure, ax, layout, risk, policy, title: str) -> None:
    plot_map(
        layout, values=risk, policy=policy, ax=ax, title=title, value_label="P(fail from cell)"
    )


def _row(label: str, r: service.PlanResult) -> str:
    o = r.outcome
    return (
        f"| {label} | {100 * o.p_fail:.2f}% | {o.cost:.2f} | {100 * o.p_timeout:.2f}% "
        f"| {1000 * r.seconds:.0f} ms |"
    )


def plan_view(text: str, p_slip: float, delta: float, penalty: float):
    """Three ways to plan for one map: the environment's own optimum, a failure penalty, and a
    bound on the failure probability. Returns a figure and a table."""
    environment = service.plan(text, p_slip=p_slip)
    priced = service.plan(text, extra_penalty=penalty, p_slip=p_slip)
    bounded = service.plan(text, delta=delta, p_slip=p_slip)
    layout = service.load_model(text, p_slip)[0].layout

    wide = layout.cols > 2 * layout.rows  # stack the two panels when the map is a long strip
    panels = (2, 1) if wide else (1, 2)
    fig = Figure(figsize=(0.3 * layout.cols + 2.5, 0.3 * layout.rows * panels[0] + 1.5))
    axes = fig.subplots(*panels)
    bound_label = f"Constrained, P(fail) <= {100 * delta:g}%"
    for ax, label, result in (
        (axes[0], "Environment optimum", environment),
        (axes[1], bound_label, bounded),
    ):
        risk, policy = _grids(text, result, p_slip)
        _draw(fig, ax, layout, risk, policy, label)
    fig.tight_layout()
    parts = len(bounded.components)
    note = "" if parts == 1 else f" The constrained plan mixes {parts} policies per episode."
    table = "\n".join(
        [
            "| Plan | P(fail) | Expected cost | P(timeout) | Solve time |",
            "|---|---|---|---|---|",
            _row("Environment optimum", environment),
            _row(f"Failure penalty +{penalty:g}", priced),
            _row(f"Constrained to {100 * delta:g}%", bounded),
        ]
    )
    gap = bounded.optimality_gap
    certificate = (
        "" if gap is None else f" Optimality gap of the constrained plan: {gap:.2g} (0 is exact)."
    )
    return fig, table + "\n\n" + note.strip() + certificate


def misspecification_view(text: str, p_true: float, p_believed: float, delta: float):
    """Plan under a believed slip probability, then score that plan under the true one."""
    informed = service.plan(text, delta=delta, p_slip=p_true)
    believer = service.plan(text, delta=delta, p_slip=p_believed)
    mixture = [(c.weight, c.actions) for c in believer.components]
    real, _ = service.evaluate(text, mixture, p_slip=p_true)

    fig = Figure(figsize=(5.6, 3.4))
    ax = fig.subplots()
    names = ["plan knows the\ntrue slip", "plan believes the\nwrong slip"]
    values = [100 * informed.outcome.p_fail, 100 * real.p_fail]
    ax.bar(names, values, color=["#2a78d6", "#eb6834"], width=0.5)
    ax.axhline(100 * delta, color="#52514e", linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(1.45, 100 * delta, " bound", va="bottom", ha="right", fontsize=9)
    ax.set_ylabel("True probability of losing the rover (%)")
    fig.tight_layout()

    excess = 100 * (real.p_fail - delta)
    verdict = (
        f"The bound is **broken**: the true failure probability is {100 * real.p_fail:.2f}%, "
        f"{excess:.2f} points above the {100 * delta:g}% the plan promised."
        if real.p_fail > delta + 1e-9
        else f"The bound **holds**: {100 * real.p_fail:.2f}% against {100 * delta:g}%."
    )
    table = "\n".join(
        [
            "| Plan made with | P(fail) it expects | True P(fail) | True expected cost |",
            "|---|---|---|---|",
            f"| the true slip ({p_true:g}) | {100 * informed.outcome.p_fail:.2f}% | "
            f"{100 * informed.outcome.p_fail:.2f}% | {informed.outcome.cost:.2f} |",
            f"| a believed slip ({p_believed:g}) | {100 * believer.outcome.p_fail:.2f}% | "
            f"{100 * real.p_fail:.2f}% | {real.cost:.2f} |",
        ]
    )
    return fig, verdict + "\n\n" + table


def results_page() -> tuple[str, list[tuple[str, str]]]:
    """The generated summary text (without its image lines) and the headline figures."""
    summary = RESULTS / "SUMMARY.md"
    text = summary.read_text() if summary.exists() else "No results are bundled with this build."
    text = re.sub(r"^!\[.*\]\(.*\)\n?", "", text, flags=re.MULTILINE)
    figures = [
        (str(RESULTS / "figures" / f"{name}.png"), name.split("_", 1)[0].upper())
        for name in HEADLINES
        if (RESULTS / "figures" / f"{name}.png").exists()
    ]
    return text, figures


def explain(error: Exception) -> str:
    """A user-facing sentence for a failed request."""
    return error.message if isinstance(error, ServiceError) else f"Unexpected error: {error}"
