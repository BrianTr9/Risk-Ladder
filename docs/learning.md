# Learners and the training harness

How the three learners (tabular Q-learning, SARSA, linear Q) are trained and measured. The
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
