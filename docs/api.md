# RiskLadder API

A thin FastAPI layer over `riskladder.service`. Interactive documentation is served at `/docs`.

## Endpoints

| Method and path | Purpose |
|---|---|
| `GET /health` | status, version, deployed commit and the request limits |
| `POST /plan` | plan for a map; optionally bound `P(fail)` by `delta` (constrained MDP) or price failure higher with `extra_penalty` |
| `POST /evaluate` | exact outcome of a policy (or a mixture of policies) that you supply |

A map is the `.map` text format of [env_spec.md](env_spec.md), sent as a string. A policy is a grid of action indices with `-1` where the rover never decides; `action_names` in the plan response names the eight indices. A mixture is a list of `{weight, actions}` components, one of which is chosen at the start of each episode.

```bash
curl -s localhost:7860/plan -H 'Content-Type: application/json' \
  -d "$(python3 -c "import json; print(json.dumps({'map': open('experiments/maps/ridge/ridge-6x30-s3000.map').read(), 'delta': 0.01}))")"
```

The response carries the policy, the exact `outcome` (`cost`, `p_success`, `p_fail`, `p_timeout`), the Lagrange `multiplier`, and `optimality_gap`: the distance to the dual bound, which is zero when the constrained solution is provably optimal. A bound that no policy can meet is reported with `feasible: false`, not approximated.

## Errors and limits

| Status | Meaning |
|---|---|
| 422 | malformed map, header or policy grid; `delta` outside [0, 1]; both `delta` and `extra_penalty`; mixture weights that do not sum to 1 |
| 413 | map text over 40,000 characters, or more than 2,500 decision cells |

Every failure of the input is a 4xx with a `detail` message; the tests check that no request reaches a 5xx.

## Latency

Planning without a binding risk bound is one value iteration (milliseconds). A binding bound is a search over the failure price, solving the problem roughly 60 to 100 times with a warm start. Measured on an Apple-silicon laptop:

| Case | Decision cells | Time |
|---|---|---|
| Environment optimum, any map up to the limit | up to 3,830 | under 75 ms |
| Binding bound, ridge 6 x 30 | 57 | about 30 ms |
| Binding bound, field 40 x 40 | 1,195 | about 0.9 s |
| Binding bound, field 60 x 60 (above the limit) | 2,785 | about 1.8 s |
| Binding bound, ridge map in the Docker container (20 calls) | 49 | p50 39 ms, p95 42 ms |

The limit of 2,500 cells keeps the worst case near 1.5 s. The smoke test (`scripts/smoke_api.py`) checks the **p95** of 20 calls against a budget of 2 s in CI and 3 s on the Space; it reports percentiles rather than a mean because a mean hides a slow tail. A free Space has a slower CPU than the measurements above, so expect a few times these figures there.

## The demo

When Gradio is installed (the `demo` extra, included in the Docker image) the same process serves an interactive demo at `/demo`, and `/` redirects to it; without Gradio `/` redirects to `/docs` and the API is unchanged. The demo is a thin layer over `riskladder.service`, the code the API runs (`app/demo.py` computes, `app/ui.py` lays out):

- **Plan a mission:** pick or generate a map, set the true slip probability, a failure bound and an extra failure price; see the environment optimum next to the constrained plan (arrows are the policy, colour the probability of losing the rover from each cell) and a table of the three solvers.
- **A wrong estimate:** plan with a believed slip, score the plan under the true one, and see whether the bound holds.
- **Results:** the headline figures and the generated summary from `results/`.

The image with the demo is about 1 GB uncompressed (Gradio and its dependencies add about 400 MB), so a cold start on a free Space is slower than for the API alone.

## Running it

```bash
uv sync --extra app --extra demo --extra exp
uv run uvicorn app.main:app --port 7860          # local; the demo is at /demo
docker build -t riskladder-api . && docker run -p 7860:7860 riskladder-api
uv run python scripts/smoke_api.py --url http://localhost:7860
```

## Deployment

CI builds the image, starts the container and smoke tests it on every push. The deploy job uploads to a Hugging Face Space, waits until `/health` reports the new commit, and smoke tests the live URL. It stays skipped until the Space is configured:

1. Create a Space on Hugging Face with the **Docker** SDK (blank template), for example `BrianTr9/risk-ladder`.
2. Create an access token with write access to it.
3. In the GitHub repository settings add the secret `HF_TOKEN` and the variable `HF_SPACE` (`owner/name`).

The free Space sleeps when idle, so the first request after a quiet period can take a minute.
