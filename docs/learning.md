# Learners and the training harness

How the learners (tabular Q-learning, SARSA, linear Q, two model-based learners) are trained and measured, and the prior knowledge that can be switched on. The
information boundary is in `env_spec.md` section 8: a learner receives `SampleEnv` and nothing else.

## The learner interface

A learner has five methods; the harness owns episodes, the step budget and all bookkeeping, so every
learner is run and measured identically.

| Method | Purpose |
|---|---|
| `begin(layout, budget_steps, rng)` | allocate tables, set the exploration schedule |
| `start_episode(layout)` | called before each episode (a multi-map learner switches map here) |
| `act(cell) -> action` | epsilon-greedy choice |
| `observe(cell, action, cost, next_cell, terminated, truncated)` | one sampled transition |
| `greedy(layout, cells) -> actions` | the greedy policy, used only for scoring |

`train(learner, envs, budget_steps, rng, scorers)` runs exactly `budget_steps` environment steps.
Episodes are cut at the horizon `H`; with several environments each episode picks one at random.
A checkpoint is taken before training and every `budget / 25` steps. It records the cumulative
training failures and cost, and scores the greedy policy **exactly** with the oracle
(`Scorer`: regret against the optimum over the same horizon, P(fail), P(timeout)). Scoring uses the
exact model, which only the harness holds; the learner never sees it.

A run is *learned* once its regret is at most `tau = 5%` of the optimal expected cost.
`summarize` gives the mean and a 95% t-interval at every checkpoint across seeds, the fraction of
runs that learned and the median steps to learn (withheld when fewer than half of the runs
learned, because the median would be cut off by the budget).

## Updates

Costs are positive and the learner estimates *cost-to-go*, so the greedy action is the arg-min.
A table of zeros is therefore an optimistic start: an untried action looks free.

- A failure ends the episode and its sampled cost already contains the failure cost, so the target
  of a terminal step is the sampled cost.
- A cut at the horizon is not a real end, so the target still bootstraps from the next cell.
- Q-learning bootstraps from the best next action, SARSA from the action it will really take.
- Exploration is epsilon-greedy with epsilon falling linearly over a fraction of the budget.
- The step size is `alpha / (1 + alpha_decay * visits)` for the tables.

## Linear Q on map-only features

`Q(s, a) = w . phi(s, a)` with ten features computed from the layout alone
(`agents/features.py`): the hazard-free distance to the goal after the intended move and after each
sideways slip, whether the move or a slip enters a hazard, how many hazard-adjacent and rough cells
remain on the shortest route from the landing cell, whether the action is a dash, and whether it
bumps into a rock or crosses rough ground. The geometry of the three branches of an action is used;
their probabilities and all costs are not, so the learner has to find out from samples how much each
feature matters. The update is semi-gradient Q-learning with the step normalised by the feature
norm, and the bootstrap value is clamped at zero (costs are never negative; without the clamp the
extrapolation drifts below zero and the weights diverge).

Because the weights belong to features rather than cells, one weight vector can be trained over many
maps and applied to a map it has never seen.

### Why these features

A first set that looked only at the landing cell (its distance to the goal and its closeness to a
hazard) could not represent the right values: even a least-squares fit to the *optimal* Q-values
produced a greedy policy that cycled between two cells on four of six maps. The hazard risk of a
route is paid at every step along it, not once, so the features now include what is left on the
shortest route from the landing cell (hazard-adjacent cells and rough tiles still to come). With
those the same fit never times out.

## Tuning

`uv run python -m experiments.e4 --tune` picks the step size and its decay for each learner from a
small grid, using only the 20 `field/train` maps (mean regret over the whole run). The choice is
saved in `experiments/e4_config.json` and used by every later run; the `field/test` maps are never
used for tuning.

## Prior knowledge as switches

Each of these is computed from the layout (and, for the shield, an assumed bound) alone, and can be
turned on independently, which is what the ablation in experiment E1 needs.

### Safety shield (`safety/shield.py`)

The shield sees the map and an assumed upper bound `p_max` on the slip probability of any single
action (default 0.4, the top of the supported range) and nothing else. For each cell and action it
computes the worst-case probability of entering a hazard in one step (the intended branch with
probability `1 - p_max`, each sideways slip with `p_max / 2`) and blocks the action when that exceeds
`theta` (default 0.05). The blocked set is applied while training, to exploration and to the greedy
choice alike, to the bootstrap target, and (optionally, `shield_at_run`) to the deployed policy.

A shield can paint the rover into a corner, so it is built to never do that. For every cell it also
finds the route to the goal whose riskiest step is smallest. A cell whose best route is riskier than
`theta` gets its own threshold raised to exactly that risk, so the route stays open; such cells are
reported (`ShieldTable.relaxed`). Entering such a cell is blocked, because the route from the cell
before it needs no risk; being in one is allowed to continue. Every cell keeps at least one action
and a route to the goal.

What it does *not* do: it bounds the risk of each step, not the probability of failing over an
episode. A rover that takes many steps just under the threshold can still fail often. That is the
job of the constrained solver, not of a shield.

### Reward shaping (`safety/shaping.py`)

A potential `Phi(s) = scale * (steps to the goal)`, zero in the absorbing outcomes. The steps are
counted along one of two routes: the shortest hazard-free route (`route="free"`) or the shortest
route that also keeps one cell clear of every hazard (`route="safe"`). A
transition costs `c + Phi(s') - Phi(s)`, so the shaped cost of an episode is the original minus
`Phi(s0)`, whatever the policy did (Ng, Harada and Russell, 1999). Shaped and original values differ
by the potential, so the best policy is unchanged; a test checks this against the oracle. The
oracle always scores on the original cost. Neither the scale nor the route affects correctness. The
scale decides how much of the cost-to-go the potential already explains, so it is tuned rather than
read from the hidden step cost. The route decides what the potential teaches while the learner
explores: the shortest route hugs the hazards on the dilemma maps and makes Q-learning lose more
rovers, while the route that keeps clear makes it lose fewer (experiment E1c). Linear Q clamps its
bootstrap at the lowest value the potential allows.

## Model-based learners (`agents/model_based.py`)

Both estimate a model from samples and plan on it with the exact solvers, replanning on a geometric
schedule (each plan after 1.1 times as many steps as the last) so planning cost stays small.

- **MB-count** keeps transition counts per (cell, action) and the mean sampled cost, and knows
  nothing of the map. Exploration is optimism and nothing else: an action tried fewer than three
  times is assumed to end the episode at no cost, so the plan is drawn to try it (R-max). It replans
  at most every 1,000 steps. The plan that is scored is not optimistic: an untried action ends the
  episode at ten times the dearest average cost seen, so the deployed plan avoids what it has not
  learned. Two earlier designs failed and are worth knowing about. Optimism with replans on a
  geometric schedule left most actions untried after 300,000 steps (the plan stood still while one
  action after another was tried). Epsilon-greedy under the pessimistic model worked on four of six
  dilemma maps and on the other two kept walking into a hazard for a million steps: pessimism
  makes every route not yet tried look worse than the known bad one, and random steps rarely find
  a long detour.
- **MB-struct** knows the geometry (where the three branches of each action lead) and learns only
  what is hidden: the slip probability of a move and of a dash, and five cost parameters (move,
  dash, rough, collision, failure) by ridge regression on the sampled costs. When the next cell fits
  more than one branch the observation is shared between them in proportion to the current slip
  estimate. Every sample informs every cell, so it needs no exploration (epsilon is 0). Its prior
  is one cost unit per step and nothing for the rest, so before the first failure it does not know
  that a hazard is costly; it learns this from the first rovers it loses.
- **MB-struct + CMDP** plans with the constrained solver under an upper confidence bound on the
  slip probabilities (Wilson bound, capped at the assumed `p_max`), so the plan bounds the failure
  probability by `delta` while the estimate is uncertain, and the bound tightens as data arrive.
  Because the constraint is on a probability, the plan is safe even before the failure cost is
  known. The deployed policy is a mixture of two policies chosen once per episode, which the
  harness scores exactly.

`MBCount` stores a dense `cells x actions x cells` table, so it is meant for maps of a few thousand
cells at most.
