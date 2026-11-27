# RiskLadder

**Exact benchmarking of safe planning and learning for autonomous agents under model uncertainty.**

> Status: under construction. The environment rules are fixed in [docs/env_spec.md](docs/env_spec.md); results and the live demo will appear here as each phase lands.

## The problem

An autonomous rover knows its map but not exactly how it moves: it may slip, and entering a hazard loses it. Should it plan with a model or learn from experience? How can it learn without being destroyed while learning? And if the model it trusts is wrong, how much does that cost?

RiskLadder answers these questions with **exact ground truth**. The environment is small enough to solve exactly, so every agent's true regret and true probability of failure are computed in closed form instead of estimated from noisy rollouts.

## The ladder

Each rung removes one piece of knowledge from the agent.

| Rung | What the agent knows | Agents |
|---|---|---|
| 1 | Deterministic dynamics, full model | A* |
| 2 | Stochastic dynamics, full model | Value / policy iteration, constrained MDP (Lagrangian) |
| 3 | Stochastic dynamics, no model | Q-learning, SARSA, linear Q-learning |
| 4 | Map known, probabilities unknown | Model-based RL, safety shield, reward shaping from the map |

## Planned claims

1. **Exact oracle:** expected cost and failure probability of any policy, exactly and faster than Monte Carlo at small risk levels.
2. **A penalty is not a constraint:** one fixed failure penalty gives very different risk across maps and noise levels; a constrained MDP holds `P(fail) ≤ δ` on every map at minimum cost. Model misspecification breaks that guarantee by a measurable amount.
3. **Prior knowledge buys back the cost of learning:** map-based shielding, shaping and structured models reduce failures during training at little cost in regret, and map features generalise to unseen maps.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## License

MIT
