"""Mutation check: do the tests notice deliberate bugs?

    uv run python tools/mutation_check.py [--jobs 2] [--only 3 7] [--dry-run]

A full run takes a long time (every mutant runs the suite), so it belongs at the end of a phase or
in the nightly job, not in the edit loop. `--dry-run` only checks that every mutation still applies.

Each mutant is one small, plausible bug applied to a copy of the repository (a changed operator, a
dropped cost, an off-by-one). The test suite then runs on the copy: a failing test KILLS the mutant,
a passing suite lets it SURVIVE, which means the tests would not have caught that bug.

The list is curated, not exhaustive: it targets the places where a fault would silently corrupt a
result (dynamics, costs, validation, solvers, evaluation, sampling). Some mutants are equivalent,
meaning no test can kill them because behaviour is unchanged; they are marked `equivalent=True` with
the reason. The run fails if a non-equivalent mutant survives or a mutation no longer applies.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parents[1]
SRC = "src/riskladder/"
QUIET = ("--no-header",)
TIMEOUT_SECONDS = 600  # the suite takes about a minute; ten means the mutant loops forever


@dataclass(frozen=True)
class Mutant:
    file: str
    old: str
    new: str
    why: str
    equivalent: bool = False


MUTANTS = [
    # --- dynamics and costs
    Mutant(
        "env/dynamics.py",
        "walk.collided * costs.collision",
        "walk.collided * 0.0",
        "collision cost dropped",
    ),
    Mutant(
        "env/dynamics.py",
        "rough += active & (tile == C_ROUGH)",
        "rough += tile == C_ROUGH",
        "rough cost charged although the rover had already stopped",
    ),
    Mutant(
        "env/dynamics.py",
        "(walk.status == FAIL) * spec.c_fail",
        "(walk.status == FAIL) * 0.0",
        "failure cost dropped",
    ),
    Mutant(
        "env/dynamics.py",
        "active &= ~(hazard | goal)",
        "active &= ~hazard",
        "a dash continues past the goal",
    ),
    Mutant(
        "env/dynamics.py",
        "status[goal] = SUCCESS",
        "status[goal] = RUNNING",
        "reaching the goal does not end the episode",
    ),
    Mutant(
        "env/dynamics.py",
        "return (act.direction, 1.0 - p), (cw, p / 2), (ccw, p / 2)",
        "return (act.direction, 1.0 - p), (cw, p / 2), (ccw, p / 3)",
        "slip probabilities do not add up to 1",
    ),
    Mutant(
        "env/dynamics.py",
        'PERPENDICULAR = {"N": ("E", "W"), "E": ("S", "N")',
        'PERPENDICULAR = {"N": ("E", "W"), "E": ("S", "S")',
        "east moves only ever slip south",
    ),
    Mutant(
        "env/dynamics.py",
        "return spec.costs.move if ACTIONS[action].cells == 1 else spec.costs.dash",
        "return spec.costs.move",
        "a dash costs the same as a move",
    ),
    # --- specification and validation
    Mutant(
        "env/spec.py",
        "self.fail_multiplier * max_reachable_distance(self.layout) * self.costs.move",
        "self.fail_multiplier * max_reachable_distance(self.layout)",
        "failure cost ignores the move cost",
    ),
    Mutant(
        "env/spec.py",
        "return 10 * (self.layout.rows + self.layout.cols)",
        "return 10 * self.layout.rows",
        "default horizon changed",
    ),
    Mutant(
        "env/spec.py",
        "and math.isfinite(value)",
        "and True",
        "NaN and infinity accepted as numbers",
    ),
    Mutant(
        "env/spec.py",
        "if not 1 <= self.horizon_override <= MAX_HORIZON:",
        "if self.horizon_override < 1:",
        "no upper limit on the horizon",
    ),
    Mutant(
        "env/spec.py", "if len(text) > MAX_MAP_CHARS:", "if False:", "no size limit on map text"
    ),
    Mutant(
        "env/layout.py",
        "raise MapError(\"no hazard-free route from 'S' to any 'G'\")",
        "return",
        "map validity check removed",
    ),
    Mutant(
        "env/distances.py",
        "return int(dist[reachable].max())",
        "return int(dist[np.isfinite(dist)].max())",
        "D_max also counts cells the rover cannot reach",
    ),
    Mutant(
        "env/generate.py",
        "safe >= MIN_DETOUR_RATIO * short",
        "safe >= 0.5 * short",
        "dilemma maps accepted without a clearly longer detour",
    ),
    # --- model and environment
    Mutant(
        "env/model.py",
        "nxt[status == FAIL] = NEXT_FAIL",
        "nxt[status == FAIL] = NEXT_SUCCESS",
        "failures recorded as successes in the model",
    ),
    Mutant(
        "env/model.py",
        "p_fail = (branch_prob[None] * (branch_next == NEXT_FAIL)).sum(axis=2)",
        "p_fail = (branch_prob[None] * (branch_next == NEXT_SUCCESS)).sum(axis=2)",
        "one-step failure probability counts successes",
    ),
    Mutant(
        "env/env.py",
        "truncated = not terminated and self._t >= self.horizon",
        "truncated = not terminated and self._t > self.horizon",
        "off by one at the horizon",
    ),
    Mutant(
        "env/env.py",
        'side="right"',
        'side="left"',
        "sampler uses the other side of equal cumulative values",
        True,
    ),
    # --- planners and the constrained solver
    Mutant(
        "agents/planners.py",
        "return min(spec.costs.move, spec.costs.dash / 2.0)",
        "return max(spec.costs.move, spec.costs.dash / 2.0)",
        "heuristic no longer a lower bound",
    ),
    Mutant(
        "agents/planners.py", "if cell == GOAL_NODE:", "if False:", "A* never ends at the goal node"
    ),
    Mutant(
        "agents/planners.py",
        "self.base.cost + self.extra * self.base.p_fail",
        "self.base.cost + self.extra",
        "penalty charged on every step, not only on failure",
    ),
    Mutant(
        "agents/cmdp.py",
        "weight_lo = min(1.0, max(0.0, (delta - hi_fail) / (lo_fail - hi_fail)))",
        "weight_lo = 0.5",
        "mixture weight does not hit the bound exactly",
    ),
    Mutant("agents/cmdp.py", "lam_hi *= 4.0", "lam_hi *= 1.0", "multiplier search never grows"),
    Mutant(
        "agents/cmdp.py",
        "hi_value - lam_hi * delta, True",
        "hi_value, True",
        "dual bound forgets the bound term",
    ),
    Mutant(
        "agents/cmdp.py",
        "def meets(fail: float) -> bool:\n        return fail <= delta + FEASIBILITY_SLACK",
        "def meets(fail: float) -> bool:\n        return fail <= delta + 0.01",
        "feasibility slack far too loose",
    ),
    # --- oracle
    Mutant(
        "oracle/evaluate.py", "if horizon < 1:", "if horizon < 0:", "horizon 0 accepted by evaluate"
    ),
    Mutant(
        "oracle/evaluate.py",
        "P = P[keep][:, keep]",
        "P = P[keep][:, keep] * 0.99",
        "restricted evaluation distorts transitions",
    ),
    Mutant(
        "oracle/evaluate.py",
        "(horizon - steps) * np.abs(term).max() <= tol",
        "np.abs(term).max() <= 1e-3",
        "early stop far too eager",
    ),
    Mutant(
        "oracle/solve.py",
        "better = q[policy, arange] - best > _TOL_REL * np.maximum(1.0, np.abs(best))",
        "better = q[policy, arange] - best > 0.5",
        "policy improvement ignores small gains",
    ),
    Mutant(
        "oracle/solve.py",
        "        new[~live] = 0.0\n",
        "        new[~live] = 5.0\n",
        "trapped states get a nonzero value during value iteration",
        True,
    ),
    Mutant(
        "oracle/rollout.py", "if horizon < 1:", "if horizon < 0:", "horizon 0 accepted by rollout"
    ),
    Mutant(
        "oracle/rollout.py",
        "status[active[nxt == NEXT_FAIL]] = FAIL",
        "status[active[nxt == NEXT_FAIL]] = SUCCESS",
        "rollouts label failures as successes",
    ),
    Mutant(
        "oracle/rollout.py",
        "cost[active] += mdp.branch_cost[s, a, branch]",
        "cost[active] += mdp.branch_cost[s, a, 0]",
        "rollouts always charge the first branch",
    ),
    Mutant(
        "oracle/rollout.py",
        "k = math.ceil((1.0 - alpha) * n - 1e-9)",
        "k = math.ceil((1.0 - alpha) * n)",
        "CVaR tail size hit by floating point rounding",
    ),
    Mutant(
        "oracle/types.py",
        "np.clip(1.0 - self.p_success - self.p_fail, 0.0, 1.0)",
        "(self.p_success + self.p_fail)",
        "timeout probability wrong",
    ),
    Mutant(
        "oracle/types.py",
        "np.any(weights < 0)",
        "np.any(weights < -1)",
        "negative mixture weights accepted",
    ),
    # --- learners, harness, priors (Phases 4 and 5)
    Mutant(
        "agents/tabular.py",
        "target = cost if terminated else cost + self._best(next_cell)",
        "target = cost + self._best(next_cell)",
        "a terminal step bootstraps from the next cell",
    ),
    Mutant(
        "agents/tabular.py",
        "self._next = None if truncated else following",
        "self._next = following",
        "SARSA keeps a pending action after a horizon cut",
    ),
    Mutant(
        "agents/tabular.py",
        "step = self.alpha / (1.0 + self.alpha_decay * self.visits[key])",
        "step = self.alpha",
        "the step size ignores the visit count",
    ),
    Mutant(
        "agents/tabular.py",
        "frac = min(1.0, self.t / self._ramp)",
        "frac = self.t / self._ramp",
        "exploration keeps falling below its floor",
    ),
    Mutant(
        "agents/linear.py",
        "target = cost + max(floor, float(q_next.min()))",
        "target = cost + float(q_next.min())",
        "the bootstrap is not clamped",
    ),
    Mutant(
        "agents/priors.py",
        "return values if allowed is None else np.where(allowed, values, np.inf)",
        "return values if allowed is None else np.where(allowed, values, -np.inf)",
        "blocked actions look best",
    ),
    Mutant(
        "harness/train.py",
        "failures += result.status == FAIL",
        "failures += result.status == SUCCESS",
        "successes are counted as failures",
    ),
    Mutant(
        "harness/train.py",
        "if steps % every == 0 or steps == budget_steps:",
        "if steps % every == 0:",
        "the last checkpoint is skipped",
    ),
    Mutant(
        "harness/scoring.py",
        "return self.tau * self.optimal.cost",
        "return self.tau",
        "the learned threshold ignores the optimal cost",
    ),
    Mutant(
        "safety/shield.py",
        "risk = (1.0 - p_max) * fail[..., 0] + 0.5 * p_max * fail[..., 1:].sum(axis=-1)",
        "risk = 1.0 * fail[..., 0] + 0.5 * p_max * fail[..., 1:].sum(axis=-1)",
        "intended hazard risk ignores slipping",
    ),
    Mutant(
        "safety/shield.py",
        "risk = (1.0 - p_max) * fail[..., 0] + 0.5 * p_max * fail[..., 1:].sum(axis=-1)",
        "risk = (1.0 - p_max) * fail[..., 0] + p_max * fail[..., 1:].sum(axis=-1)",
        "a slip side gets the whole slip probability",
    ),
    Mutant(
        "safety/shield.py",
        "ok = risk[r, c] <= bar + _EPS",
        "ok = risk[r, c] <= self.theta + _EPS",
        "cells that need a risky route lose it",
    ),
    Mutant(
        "safety/shield.py",
        "ok[a] = route[lr, lc] <= bar + _EPS",
        "pass",
        "landing in a dead end is allowed",
    ),
    Mutant(
        "safety/shaping.py",
        "after = 0.0 if terminated else float(potential[next_cell])",
        "after = float(potential[next_cell])",
        "the potential of an absorbing outcome is not zero",
    ),
    Mutant(
        "agents/model_based.py",
        "known = self.visits >= self.known_after",
        "known = self.visits > self.known_after",
        "an action counts as known one visit late",
    ),
    Mutant(
        "agents/model_based.py",
        "weight = match * np.array([1.0 - p, p / 2, p / 2])",
        "weight = match * np.array([1.0, p / 2, p / 2])",
        "the intended branch is always preferred",
    ),
    Mutant(
        "agents/model_based.py",
        "centre = p + z * z / (2 * trials)",
        "centre = p - z * z / (2 * trials)",
        "the confidence bound is not an upper bound",
    ),
    Mutant(
        "agents/model_based.py",
        "return max(self.mixture(layout, cells), key=lambda part: part[0])[1]",
        "return min(self.mixture(layout, cells), key=lambda part: part[0])[1]",
        "greedy reports the lightest part of the mixture",
    ),
    Mutant(
        "agents/features.py",
        "phi[..., 3] = b.status[intended] == FAIL",
        "phi[..., 3] = 0.0",
        "the hazard flag is always off",
    ),
]


def run_one(index: int, mutant: Mutant) -> tuple[int, str, str]:
    with tempfile.TemporaryDirectory(prefix="mutant-") as tmp:
        work = Path(tmp) / "repo"
        ignore = shutil.ignore_patterns(
            ".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis",
            "results", ".coverage", "dist", "build",
        )  # fmt: skip
        shutil.copytree(ROOT, work, ignore=ignore)
        target = work / SRC / mutant.file
        text = target.read_text()
        if text.count(mutant.old) != 1:
            return (
                index,
                "NOT-APPLIED",
                f"expected exactly one match, found {text.count(mutant.old)}",
            )
        target.write_text(text.replace(mutant.old, mutant.new))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(work / "src"), str(work)]), CI="1")
        try:
            run = subprocess.run(
                [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *QUIET],
                cwd=work, env=env, capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
            )  # fmt: skip
        except subprocess.TimeoutExpired:  # a mutant that loops forever is noticed by the clock
            return index, "KILLED", "timeout (the mutant never finishes)"
        if run.returncode == 0:
            return index, "SURVIVED", ""
        first = next((ln for ln in run.stdout.splitlines() if ln.startswith("FAILED")), "error")
        return index, "KILLED", first.removeprefix("FAILED ").split(" - ")[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--only", type=int, nargs="*", help="run only these mutant numbers")
    parser.add_argument("--dry-run", action="store_true", help="only check the mutations apply")
    args = parser.parse_args()
    chosen = [(i, m) for i, m in enumerate(MUTANTS) if not args.only or i in args.only]
    if args.dry_run:
        stale = [
            i for i, m in chosen
            if (ROOT / SRC / m.file).read_text().count(m.old) != 1
        ]  # fmt: skip
        print(f"{len(chosen) - len(stale)}/{len(chosen)} mutations apply; stale: {stale or 'none'}")
        return 1 if stale else 0
    with ThreadPoolExecutor(args.jobs) as pool:
        results = sorted(pool.map(lambda im: run_one(*im), chosen))

    killed = survived = equivalent = broken = 0
    print(f"{'#':>2}  {'result':<11} mutant")
    for index, status, detail in results:
        mutant = MUTANTS[index]
        note = " (equivalent)" if mutant.equivalent and status == "SURVIVED" else ""
        print(f"{index:>2}  {status:<11} {mutant.file}: {mutant.why}{note}")
        if status == "KILLED":
            killed += 1
        elif status == "SURVIVED" and mutant.equivalent:
            equivalent += 1
        elif status == "SURVIVED":
            survived += 1
            print(f"      -> no test noticed this bug: {mutant.old!r} -> {mutant.new!r}")
        else:
            broken += 1
            print(f"      -> {detail}")
    considered = len(results) - equivalent
    score = 100 * killed / considered if considered else 100.0
    print(
        f"\nkilled {killed}/{considered} non-equivalent mutants ({score:.0f}%), "
        f"{equivalent} equivalent survived, {survived} unexpected survivors, "
        f"{broken} not applied"
    )
    return 1 if survived or broken else 0


if __name__ == "__main__":
    sys.exit(main())
