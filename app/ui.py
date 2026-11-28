"""The Gradio demo: three tabs laid out over `app/demo.py`. No computation happens here."""

from __future__ import annotations

import gradio as gr

from app import demo

INTRO = """\
# RiskLadder

A rover must cross a hazard map without losing itself. The rover slips sideways with some
probability; entering a hazard ends the mission. Everything here is computed **exactly**, not
by simulation. Arrows show the policy, colour the probability of losing the rover from each cell.
"""


def _run(function, *args):
    try:
        return function(*args)
    except Exception as error:  # shown to the user instead of a stack trace
        raise gr.Error(demo.explain(error)) from None


def build() -> gr.Blocks:
    first = next(iter(demo.EXAMPLES))
    first_text, first_slip, _ = demo.example_view(first)
    with gr.Blocks(title="RiskLadder") as page:
        gr.Markdown(INTRO)
        text = gr.Textbox(value=first_text, lines=8, label="Map (editable, `.map` text format)")
        with gr.Row():
            example = gr.Dropdown(list(demo.EXAMPLES), value=first, label="Example map")
            template = gr.Dropdown(list(demo.TEMPLATES), value="ridge", label="Or generate")
            seed = gr.Number(value=1, precision=0, label="Seed")
            generate = gr.Button("Generate")

        with gr.Tab("Plan a mission"):
            with gr.Row():
                slip = gr.Slider(
                    0.0, 0.4, value=first_slip, step=0.005, label="True slip probability"
                )
                delta = gr.Slider(0.0, 0.2, value=0.01, step=0.005, label="Failure bound (delta)")
                penalty = gr.Slider(0.0, 500.0, value=50.0, step=5.0, label="Extra failure price")
            go = gr.Button("Plan", variant="primary")
            figure, table = gr.Plot(), gr.Markdown()
            go.click(
                lambda *a: _run(demo.plan_view, *a), [text, slip, delta, penalty], [figure, table]
            )

        with gr.Tab("A wrong estimate"):
            with gr.Row():
                true = gr.Slider(
                    0.0, 0.4, value=first_slip, step=0.005, label="True slip probability"
                )
                believed = gr.Slider(
                    0.0, 0.4, value=0.005, step=0.005, label="Slip the planner believes"
                )
                bound = gr.Slider(0.0, 0.2, value=0.01, step=0.005, label="Failure bound (delta)")
            go2 = gr.Button("Compare", variant="primary")
            figure2, verdict = gr.Plot(), gr.Markdown()
            go2.click(
                lambda *a: _run(demo.misspecification_view, *a),
                [text, true, believed, bound],
                [figure2, verdict],
            )

        with gr.Tab("Results"):
            summary, figures = demo.results_page()
            gr.Gallery(figures, label="Headline figures", columns=2, height="auto")
            gr.Markdown(summary)

        example.change(lambda name: _run(demo.example_view, name), example, [text, slip, true])
        generate.click(
            lambda *a: _run(demo.generated_view, *a), [template, seed], [text, slip, true]
        )
    return page
