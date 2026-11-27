"""Multi-seed runs scored by the oracle, written as JSON lines and summarised with 95% intervals."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

from riskladder.agents.linear import LinearQ
from riskladder.agents.tabular import SARSA, QLearning
from riskladder.env import RoverEnv, load_map
from riskladder.harness.scoring import Scorer
from riskladder.harness.train import Checkpoint, train

AGENTS = {"q": QLearning, "sarsa": SARSA, "linear": LinearQ}


@dataclass(frozen=True)
class RunSpec:
    """One training run: an agent on a map, one seed, one budget."""

    agent: str
    map_path: str
    seed: int
    budget: int
    config: dict = field(default_factory=dict)
    n_checkpoints: int = 25
    tau: float = 0.05


def checkpoint_row(cp: Checkpoint) -> dict:
    score = cp.scores[0]
    return {
        "steps": cp.steps, "episodes": cp.episodes, "train_failures": cp.train_failures,
        "train_successes": cp.train_successes, "train_cost": cp.train_cost,
        "regret": score.regret, "cost": score.cost, "p_success": score.p_success,
        "p_fail": score.p_fail, "p_timeout": score.p_timeout,
    }  # fmt: skip


def learned_at(rows: Sequence[dict], threshold: float) -> int | None:
    """Steps at the first checkpoint whose regret is within `threshold`; None if never."""
    for row in rows:
        if row["regret"] <= threshold:
            return int(row["steps"])
    return None


def run(spec: RunSpec, scorer: Scorer | None = None, env: RoverEnv | None = None) -> dict:
    """Train one agent and return its record. Pass `scorer`/`env` to reuse a map's exact model."""
    if spec.agent not in AGENTS:
        raise ValueError(f"unknown agent {spec.agent!r}; choose from {sorted(AGENTS)}")
    env_seed, agent_seed = np.random.SeedSequence(spec.seed).spawn(2)
    if env is None:
        env = RoverEnv(load_map(spec.map_path), seed=env_seed)
    learner = AGENTS[spec.agent](**spec.config)
    scorer = scorer or Scorer(env, spec.tau)
    checkpoints = train(
        learner, env.sample_interface(), spec.budget, np.random.default_rng(agent_seed),
        [scorer], spec.n_checkpoints,
    )  # fmt: skip
    rows = [checkpoint_row(cp) for cp in checkpoints]
    return {
        "agent": spec.agent, "config": learner.config, "map": Path(spec.map_path).stem,
        "seed": spec.seed, "budget": spec.budget, "tau": spec.tau,
        "optimal_cost": scorer.optimal.cost, "threshold": scorer.threshold,
        "learned_at": learned_at(rows, scorer.threshold), "checkpoints": rows,
    }  # fmt: skip


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for record in records:
            out.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean_ci(values: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Mean and half-width of the t interval over axis 0 (NaN width with a single sample)."""
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    mean = values.mean(axis=0)
    if n < 2:
        return mean, np.full_like(mean, np.nan)
    half = stats.t.ppf(0.5 + level / 2, n - 1) * values.std(axis=0, ddof=1) / np.sqrt(n)
    return mean, half


def summarize(records: Sequence[dict]) -> dict:
    """Curves across seeds (and maps) of one agent: mean and 95% half-width at each checkpoint.

    All records must share a budget and checkpoint grid. `learned_fraction` is the share of runs
    that reached the threshold; `learned_median` is the median steps among them, None unless at
    least half of the runs learned (otherwise the median would be censored by the budget).
    """
    if not records:
        raise ValueError("no records to summarise")
    grids = {tuple(row["steps"] for row in r["checkpoints"]) for r in records}
    if len(grids) != 1:
        raise ValueError("records have different checkpoint grids")
    steps = np.array(next(iter(grids)))
    out: dict = {"runs": len(records), "steps": steps.tolist()}
    for key in ("regret", "p_fail", "p_timeout", "train_failures", "train_cost"):
        data = np.array([[row[key] for row in r["checkpoints"]] for r in records], dtype=float)
        mean, half = mean_ci(data)
        out[key] = {"mean": mean.tolist(), "ci95": half.tolist()}
    learned = [r["learned_at"] for r in records if r["learned_at"] is not None]
    out["learned_fraction"] = len(learned) / len(records)
    out["learned_median"] = float(np.median(learned)) if len(learned) * 2 >= len(records) else None
    return out
