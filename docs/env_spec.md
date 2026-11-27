# RiskLadder environment specification (v0.1)

This document fixes the rules of the environment before any code is written. Every module (dynamics, oracle, agents, API) must follow it, and any change to it is a versioned decision recorded in the changelog at the end.

## 1. Mission

A rover starts at `S` and must reach any goal `G` on a grid map. Entering a hazard `X` loses the rover and ends the mission. Movement is noisy: the rover may slip sideways. The objective is to finish the mission at the lowest expected cost, optionally subject to a bound on the probability of losing the rover.

## 2. Map

### 2.1 Tiles

| Symbol | Tile | Effect |
|---|---|---|
| `.` | ground | passable |
| `S` | start | ground; exactly one per map |
| `~` | rough | passable; entering it adds `rough_extra` cost |
| `#` | rock | impassable; moving into it is a collision |
| `X` | hazard | entering it ends the episode in `FAIL` and adds `c_fail` |
| `G` | goal | entering it ends the episode in `SUCCESS`; one or more per map |

Cells outside the grid behave like rock.

### 2.2 Validity

A map is valid only if all of the following hold:

- the grid is rectangular and uses only the symbols above;
- there is exactly one `S` and at least one `G`;
- some `G` is reachable from `S` through `.`, `~`, `S` cells only (4-connected), so a hazard-free route exists.

### 2.3 File format

A map file (`.map`) is a JSON header, a line containing only `---`, and the grid.

```
{
  "name": "dilemma-30-s7",
  "seed": 7,
  "noise": {"p_slip": 0.15, "dash_slip_mult": 2.0},
  "costs": {"move": 1.0, "dash": 1.5, "rough_extra": 1.0, "collision": 2.0},
  "fail_multiplier": 5.0
}
---
##########
#S..X...G#
#........#
##########
```

Optional header fields: `c_fail` (overrides `fail_multiplier`), `horizon` (overrides the default `H`). The same format is used by tests, experiments and the API.

## 3. Actions

Eight actions with a fixed index order:

| Index | Name | Cells | Base cost |
|---|---|---|---|
| 0–3 | `move_N`, `move_E`, `move_S`, `move_W` | 1 | `move` = 1.0 |
| 4–7 | `dash_N`, `dash_E`, `dash_S`, `dash_W` | 2 | `dash` = 1.5 |

A dash is cheaper per cell (0.75 vs 1.0) but slips more often, so the agent trades cost against risk even on a fixed route. Every action is available in every non-terminal state.

## 4. Dynamics

For an action with intended direction `d` and length `k`:

1. **Slip.** The effective direction is `d` with probability `1 − p`, and each of the two perpendicular directions with probability `p / 2`, where `p = p_slip` for a move and `p = dash_slip_mult · p_slip` for a dash. Parameters must satisfy `0 ≤ p ≤ 1` for both.
2. **Motion.** The rover advances up to `k` cells in the effective direction, one cell at a time:
   - next cell is rock or outside the grid: the rover stays in its current cell, `collision` cost is added once, motion stops;
   - next cell is `X`: motion stops, the episode ends in `FAIL`, `c_fail` is added;
   - next cell is `G`: motion stops, the episode ends in `SUCCESS`;
   - next cell is `~`: the rover enters it and `rough_extra` is added;
   - otherwise the rover enters the cell.
3. **Cost.** The base cost of the action is always charged, plus any collision, rough and failure costs incurred during motion.

The function `outcomes(s, a)` enumerates every branch of this process with its probability. Both the exact model and the sampler are built from it, so they cannot disagree.

## 5. Costs and the failure cost

All step costs are strictly positive (`move`, `dash` > 0, extras ≥ 0).

The failure cost is `c_fail = fail_multiplier · D_max`, where `D_max` is the largest deterministic, hazard-free step distance from any reachable cell to its nearest goal, multiplied by `move`. Default `fail_multiplier = 5.0`.

**Why it exists.** Entering a hazard ends the episode. Without a failure cost, any cost minimiser (including a constrained one) would prefer to end the episode early in a hazard rather than pay for the rest of the route. `c_fail` makes losing the rover always worse than finishing. This is checked by a test: with `p_slip = 0`, the optimal policy never enters a hazard.

The failure indicator `1[FAIL]` is reported separately from cost. It is never folded into cost except by the penalty baseline (section 7).

## 6. States, horizon and evaluation

### 6.1 States

The decision states are the passable cells (`.`, `S`, `~`) indexed in row-major order, plus two absorbing outcomes, `SUCCESS` and `FAIL`, that cost nothing further. Goal and hazard cells are not decision states: entering one ends the episode. The absorbing outcomes are implicit: the exact model has no row for them (the missing probability mass of each row is the absorption probability), and the sampler refuses to step after an episode has ended.

### 6.2 Planning objective

Planners solve a stochastic shortest path problem with **no discounting (γ = 1)**. Because step costs are positive, a policy that never terminates has infinite cost, so value iteration converges to the optimal proper policy. Cost and failure probability are measured on the same undiscounted scale, which keeps the Lagrangian of the constrained problem consistent.

- Value iteration stops when the maximum Bellman residual is below `1e-9`.
- Policy iteration starts from a proper policy and keeps every iterate proper.

### 6.2.1 Solver notes

- **Value iteration** is Jacobi (all states updated from the previous sweep) and starts at `V = 0`, so iterates increase monotonically towards `V*`. The Bellman residual is not an error bound for an undiscounted problem, so tests compare against policy iteration.
- **Policy iteration** evaluates each policy exactly with one sparse LU factorisation (SciPy `splu`, COLAMD ordering). It starts from a proper policy built from fewest-step layers (each state takes the action most likely to move one layer closer to termination), and an improvement only replaces the current action when it is better by more than `1e-10` (relative), so it cannot cycle.
- **Trapped cells.** A pocket of ground sealed off from every goal and hazard cannot terminate under any policy, so its value is infinite. It is detected once by a reverse graph search and excluded from both solvers. A live state can never move into a trapped one.
- **Scale.** Value iteration and policy evaluation run to about 10^6 states. Policy iteration with a direct solve is limited to about 10^5 states by memory (see `results/bench.md`).

### 6.3 Horizon

Every episode is truncated after `H` steps (`truncated = True`, no extra cost). Default `H = 10 · (rows + cols)`, for example 600 on a 30 × 30 map. `H` must be large enough that the optimal policy's timeout probability is below `1e-6` on every experiment map; a test enforces this.

### 6.4 Policy evaluation

The oracle evaluates any policy by an exact backward recursion over `H` steps on the policy's transition matrix. It returns, from every state:

- expected cost within the horizon,
- `P(success)`, `P(fail)`, `P(timeout)`, which sum to 1.

This never becomes singular, so it also works for poor learned policies that loop. **Regret** is `E_H[cost | π] − E_H[cost | π*]` at the start state.

The series is `X = sum_{k<H} P^k b` for the policy's transition matrix `P` and one-step vector `b = [cost, P(success), P(fail)]`. Each term is bounded by the previous one in max-norm, so the evaluation can stop early with a guaranteed absolute error below `tol` (default `1e-12`); a policy that never terminates keeps large terms and runs all `H` steps. When only the value at the start is needed, the series runs on the states reachable from the start, which is much faster on large maps.

A mixture policy (pick `π_i` with weight `w_i` once at the start of an episode) is evaluated as the weighted sum of the components' results, which is exact because every quantity above is linear in the mixture weights.

### 6.5 Tail risk

`CVaR_0.95` is computed on the mission cost of **successful** episodes only, by Monte Carlo rollouts with a bootstrap confidence interval. Failures are measured exactly by `P(fail)`. Mixing them into CVaR would make early failures, which have small cost, look like a safe tail.

## 7. Objectives

| Name | Objective | Used by |
|---|---|---|
| Environment cost | minimise `E[cost]` (includes `c_fail`) | all learners, unconstrained planner |
| Penalty baseline | minimise `E[cost] + λ_pen · P(fail)` | penalty planner |
| Constrained (CMDP) | minimise `E[cost]` subject to `P(fail) ≤ δ` | CMDP planner, constrained model-based learner |

## 8. Information available to each agent

| Interface | Map tiles | Noise parameters | Cost values | Exact model |
|---|---|---|---|---|
| Planner (`env.model()`) | yes | yes | yes | yes |
| Learner (`SampleEnv`) | yes | no | no (observed only through sampled costs) | no |
| Safety shield | yes | upper bound `p_max` only | no | no |

The learner interface (`SampleEnv`) exposes the map, the start, the goals, the action set, `H`, `reset()` and `step()`. It has no `model()` and no noise or cost attributes. Python cannot make the wrapped environment truly private, so this boundary is an interface contract: agent code uses only the public interface, and a test checks that the hidden attributes are absent. Features and reward shaping that need a distance use deterministic **step** distances on the map, which require no hidden values. Any potential function preserves the optimal policy under potential-based shaping, so this choice is safe.

Default shield assumption: `p_max = 0.4`, the upper end of the supported `p_slip` range.

## 9. Experiment defaults

| Parameter | Default | Sweep (where relevant) |
|---|---|---|
| Map size | 30 × 30 | 10 × 10 (tests), up to 1000 × 1000 (latency benchmark) |
| `p_slip` | 0.15 | 0.0, 0.05, 0.1, 0.2, 0.3 |
| `dash_slip_mult` | 2.0 | |
| Costs | move 1.0, dash 1.5, rough +1.0, collision +2.0 | |
| `fail_multiplier` | 5.0 | |
| `δ` | 0.01 | log grid 0.001 to 0.3 |
| Seeds per configuration | 10 | |

## Changelog

- v0.1: initial specification (Phase 0).
- v0.1.1: clarified that the learner boundary is an interface contract, not a hard guarantee (Phase 1).
- v0.1.2: documented solver details, trapped cells, early-stopped evaluation and scale limits (Phase 2).
- v0.1.3: absorbing outcomes described as implicit; dilemma generator rejects maps whose safe route is under 1.4x the corridor (verification pass).
