"""Helpers shared by the learning experiments (E1, E1b, E4): parallel runs and honest summaries.

Regret is a heavy-tailed quantity. A greedy policy that cycles never reaches the goal and pays its
step costs until the horizon, so one such run can outweigh nine good ones in a mean. Summaries
therefore report the median and quartiles next to the mean, and count the runs whose final policy
loops (a timeout probability above `LOOP`), instead of letting a few runs speak for all.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from riskladder.harness import RunSpec, run, summarize
from riskladder.harness.runner import mean_ci

LOOP = 0.005  # a final policy that times out with more than this probability counts as looping


def pool_size() -> int:
    return max(1, min(10, (os.cpu_count() or 2) - 1))


def run_one(spec: RunSpec) -> dict:
    return run(spec)


def run_specs(specs: Sequence[RunSpec]) -> list[dict]:
    """Run every spec in parallel; the records come back in the order of the specs."""
    with ProcessPoolExecutor(pool_size()) as pool:
        return list(pool.map(run_one, specs, chunksize=1))


def relative(record: dict, tau: float) -> dict:
    """The same record with regret as a fraction of the optimal cost (comparable across maps)."""
    scale = record["optimal_cost"]
    rows = [{**row, "regret": row["regret"] / scale} for row in record["checkpoints"]]
    return {**record, "checkpoints": rows, "threshold": tau}


def stat(values: np.ndarray) -> dict:
    """Mean with a 95% interval, and the median with its quartiles."""
    mean, half = mean_ci(values)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {
        "mean": float(mean), "ci95": float(half), "median": float(median),
        "q1": float(q1), "q3": float(q3),
    }  # fmt: skip


def final_summary(records: Sequence[dict]) -> dict:
    """What the runs of one configuration end with. `records` carry relative regret (`relative`)."""
    last = [r["checkpoints"][-1] for r in records]
    curves = summarize(records)
    return {
        "runs": len(records),
        "train_failures": stat(np.array([c["train_failures"] for c in last], dtype=float)),
        "regret": stat(np.array([c["regret"] for c in last])),
        "p_fail": stat(np.array([c["p_fail"] for c in last])),
        "looping_runs": int(sum(c["p_timeout"] > LOOP for c in last)),
        "optimal_p_fail": float(np.mean([r["optimal_p_fail"] for r in records])),
        "learned_fraction": curves["learned_fraction"],
        "learned_median": curves["learned_median"],
        "curves": curves,
    }
