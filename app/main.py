"""FastAPI app: `GET /health`, `POST /plan`, `POST /evaluate`.

All logic lives in `riskladder.service`; this module only describes the request and response
shapes, turns `ServiceError` into a response and reports the version and commit.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from riskladder import __version__, service
from riskladder.service import ServiceError

DESCRIPTION = """\
Plan a mission for a rover on a hazard map and measure its risk exactly.

* `POST /plan` returns the cheapest policy, optionally with the probability of losing the rover
  bounded by `delta` (a constrained MDP, solved exactly) or with a higher failure price.
* `POST /evaluate` scores a policy you bring.

Maps use the `.map` text format described in `docs/env_spec.md`. A policy is a grid of action
indices, `-1` where the rover never decides; `action_names` in the response names the indices.
"""


class PlanRequest(BaseModel):
    map: str = Field(description="the map in the `.map` text format (JSON header, `---`, grid)")
    delta: float | None = Field(
        None, ge=0, le=1, description="bound on P(fail); constrained solver"
    )
    extra_penalty: float | None = Field(
        None, ge=0, description="extra failure price; penalty planner"
    )
    p_slip: float | None = Field(
        None, ge=0, le=1, description="override the map's slip probability"
    )


class Outcome(BaseModel):
    cost: float
    p_success: float
    p_fail: float
    p_timeout: float


class ComponentOut(BaseModel):
    weight: float = Field(description="probability that this policy is the one used in an episode")
    actions: list[list[int]]


class PlanResponse(BaseModel):
    solver: str
    components: list[ComponentOut]
    outcome: Outcome
    multiplier: float | None
    optimality_gap: float | None = Field(description="0 means provably optimal (constrained only)")
    feasible: bool
    states: int
    seconds: float
    action_names: list[str]


class ComponentIn(BaseModel):
    weight: float = Field(ge=0, le=1)
    actions: list[list[int]]


class EvaluateRequest(BaseModel):
    map: str
    components: list[ComponentIn]
    p_slip: float | None = Field(None, ge=0, le=1)


class EvaluateResponse(BaseModel):
    outcome: Outcome
    seconds: float


class Health(BaseModel):
    status: str
    version: str
    commit: str
    max_states: int
    max_map_chars: int


COMMIT_FILE = Path(__file__).parents[1] / "COMMIT_SHA"


def commit_sha(file: Path = COMMIT_FILE) -> str:
    """The commit this build came from: the COMMIT_SHA variable, or a file written at build time."""
    if os.environ.get("COMMIT_SHA"):
        return os.environ["COMMIT_SHA"]
    return file.read_text().strip() if file.exists() else "unknown"


def create_app() -> FastAPI:
    app = FastAPI(title="RiskLadder", version=__version__, description=DESCRIPTION)

    @app.exception_handler(ServiceError)
    async def service_error(_: Request, error: ServiceError) -> JSONResponse:
        return JSONResponse(status_code=error.status, content={"detail": error.message})

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health(
            status="ok",
            version=__version__,
            commit=commit_sha(),
            max_states=service.MAX_STATES,
            max_map_chars=service.MAX_MAP_CHARS,
        )

    @app.post("/plan", response_model=PlanResponse)
    def plan(request: PlanRequest) -> PlanResponse:
        result = service.plan(request.map, request.delta, request.extra_penalty, request.p_slip)
        return PlanResponse(
            solver=result.solver,
            components=[
                ComponentOut(weight=c.weight, actions=c.actions) for c in result.components
            ],
            outcome=Outcome(**vars(result.outcome)),
            multiplier=result.multiplier,
            optimality_gap=result.optimality_gap,
            feasible=result.feasible,
            states=result.states,
            seconds=result.seconds,
            action_names=service.action_names(),
        )

    @app.post("/evaluate", response_model=EvaluateResponse)
    def evaluate(request: EvaluateRequest) -> EvaluateResponse:
        outcome, seconds = service.evaluate(
            request.map, [(c.weight, c.actions) for c in request.components], request.p_slip
        )
        return EvaluateResponse(outcome=Outcome(**vars(outcome)), seconds=seconds)

    _mount_demo(app)
    return app


def _mount_demo(app: FastAPI) -> None:
    """Serve the Gradio demo at /demo when it is installed; the API works without it."""
    try:
        import gradio

        from app.ui import build
    except ImportError:
        target = "/docs"
    else:
        gradio.mount_gradio_app(app, build(), path="/demo")
        target = "/demo"

    @app.get("/", include_in_schema=False)
    def home() -> RedirectResponse:
        return RedirectResponse(target)


app = create_app()
