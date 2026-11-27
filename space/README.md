---
title: Risk Ladder
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Exact risk-bounded mission planning API
---

# RiskLadder API

Plan a mission for a rover on a hazard map and measure its risk exactly. `POST /plan` returns the
cheapest policy with the probability of losing the rover bounded by `delta` (a constrained MDP,
solved exactly with an optimality certificate); `POST /evaluate` scores a policy you bring.

Open `/docs` for the interactive API documentation. The free Space sleeps when idle, so the first
request after a quiet period can take a minute while it wakes up.

Source and write-up: https://github.com/BrianTr9/Risk-Ladder
