---
title: Risk-Bounded Autonomy
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: space_app.py
pinned: false
license: mit
short_description: Exact risk-bounded mission planning demo and API
---

# Risk-Bounded Autonomy

Plan a mission for a rover on a hazard map and measure its risk exactly. The demo plans with the
probability of losing the rover bounded by `delta` (a constrained MDP, solved exactly with an
optimality certificate), shows what a wrong slip estimate breaks, and browses the measured results.

The same planner is available as an API: `gradio_client.Client(<this space>).predict(map, delta,
api_name="/plan")`. The free Space sleeps when idle, so the first request after a quiet period can
take a minute while it wakes up. It runs on ZeroGPU hardware only because that is the free option;
nothing here uses a GPU.

Source and write-up: https://github.com/BrianTr9/Risk-Bounded-Autonomy
