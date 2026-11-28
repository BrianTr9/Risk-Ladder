"""Entry point of the Hugging Face Space (Gradio SDK, ZeroGPU hardware).

A free account cannot create a Docker or a CPU Space, so the public demo runs as a Gradio Space.
This file only wires `app/ui.py` (the demo) and `riskladder.service` (the planner) into the shape
the platform wants; the same code runs locally. The planner is CPU-only. ZeroGPU insists that one
function is registered for the GPU at startup, so a placeholder is registered and never called.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:  # must be imported before anything that touches CUDA; absent everywhere but on the Space
    import spaces
except ImportError:
    spaces = None

HERE = Path(__file__).parent
for folder in (HERE / "src", HERE):  # the Space does not `pip install` the package
    sys.path.insert(0, str(folder))

import gradio as gr  # noqa: E402
from app import ui  # noqa: E402

from riskladder import service  # noqa: E402


def gpu_placeholder() -> None:
    """Registered with ZeroGPU so the Space starts; the planner never needs a GPU."""


if spaces is not None:
    gpu_placeholder = spaces.GPU(duration=1)(gpu_placeholder)


def plan_api(map: str, delta: float, p_slip: float = -1.0) -> dict:
    """Cheapest policy whose probability of losing the rover is at most `delta`.

    `map` is the `.map` text format of docs/env_spec.md; a negative `p_slip` keeps the map's own
    slip probability. Returns the outcome, the policy (a mixture of at most two) and the solver
    certificate, as in the REST API's `POST /plan`.
    """
    result = service.plan(map, delta=delta, p_slip=None if p_slip < 0 else p_slip)
    return {
        "outcome": vars(result.outcome),
        "components": [{"weight": c.weight, "actions": c.actions} for c in result.components],
        "optimality_gap": result.optimality_gap,
        "feasible": result.feasible,
        "states": result.states,
        "seconds": result.seconds,
        "action_names": service.action_names(),
    }


demo = ui.build()
with demo:
    gr.api(plan_api, api_name="plan")


if __name__ == "__main__":
    demo.launch()
