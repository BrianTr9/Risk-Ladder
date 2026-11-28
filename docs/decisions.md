# Design decisions

The choices that shaped this project, each with the alternative I passed over, the reason, and the evidence or the cost. The last section lists the things I got wrong first.

## The problem and the oracle

1. A grid world, solved exactly. The alternative was a simulator with continuous state. A grid has a state space small enough to enumerate, so the expected cost and the failure probability of any policy come from a sparse linear solve instead of rollouts. Exact evaluation is 2,043 times faster than Monte Carlo at a failure probability of 0.001 (E0), and every comparison in the project has no evaluation noise. The cost is realism: nothing here involves perception or continuous control.

2. Failure has a price inside the cost, and is also tracked as a probability. The alternative was to let a hazard end the episode at the cost of the steps taken so far. Then every optimiser, including the constrained one, prefers to enter a hazard early because it is the cheapest way to end the episode. The price `C_fail` (5 times the longest hazard-free route times the move cost) removes that incentive, and the probability is then free to be constrained on its own.

3. No discounting, with a finite horizon for evaluation. Costs are positive, so a policy that never ends has infinite cost and value iteration from zero rises to the optimum. A discount would make the cost and the failure probability disagree about how much the future counts, which breaks the Lagrangian. Learned policies can loop, so evaluation runs for a fixed horizon and reports the probability of a timeout; a looping policy is charged for the steps it takes.

## Risk

4. The constraint is solved by Lagrangian bisection plus a mixture of two policies chosen once per episode. The alternative was a fixed failure penalty, which is what the penalty planner does. The failure probability of the Lagrangian-optimal policy jumps as the multiplier grows, and the constrained optimum sits at a jump, so only a mixture meets the bound exactly. Choosing the policy once per episode keeps the mixture's failure probability linear in the weights. One penalty tuned on a single map broke a 1% bound on 7 of 12 maps (E3). The solver comes with a dual bound, and its gap was about 1e-10.

5. A model the planner believes can be wrong, and the cost is measured. E2 plans with a wrong slip and scores the plan under the true one: a planner that believes the slip is 16 times too low exceeds the bound by 13.7 percentage points. A safety factor on the believed slip fixes an underestimate up to the factor and costs 0.75 when the belief was right.

## Learning

6. Learners are tabular or linear. The alternative was deep reinforcement learning. The question is what the lack of knowledge costs and what recovers it, which does not need function approximation of that size, and exact scoring needs a learner whose policy can be read out per cell.

7. The learner sees samples only. `SampleEnv` exposes the map, the start, the goals, the horizon, `reset` and `step`; the exact model lives in a separate interface that planners use. Python cannot make that boundary hard, so a test checks that the learners' code names no noise, cost or model symbol. The geometry of an action (where its three branches lead) is public, since it is the action set; the probabilities are not.

8. A model-based learner that knows nothing of the map, and one that knows its geometry. The second alone would look like cheating. The first plans on counted transitions with optimism about untried actions; the gap between them (270 against 17 rovers lost) is the value of the map.

9. The shield keeps a route open. A shield that blocks every risky action can leave a cell with no way out, so each cell's threshold is raised to the smallest risk of any route to the goal when that is above the base threshold, and those cells are reported. What it guarantees is the risk of each step under an assumed slip bound. It does not bound an episode.

10. Shaping uses a potential, and the route it follows matters. The potential preserves the optimal policy whatever it is. In E1 it followed the shortest hazard-free route, which hugs the hazards, and Q-learning lost more rovers. A potential following the route that keeps one cell clear lost 252 rovers against 1,252 without shaping (E1c).

## Reporting and engineering

11. Medians, not means. Regret is heavy-tailed because a looping policy pays until the horizon. Tables give the median with quartiles and count the looping runs. The mean of SARSA's regret on the ridge maps was 438%, the median 20%.

12. An independent reference implementation, property tests and a mutation check. The reference is written from the specification and compared with the vectorised dynamics; properties and parser fuzzing guard the inputs; 55 of 55 deliberate bugs are caught. The mutation run found two real gaps in the model-based tests, which I fixed.

13. One entry point for experiments, one place for settings. `experiments.run` runs everything, with a smoke mode of about a minute for CI. Settings sit next to the code they belong to and in three tuned-settings files that only `--tune` rewrites. I dropped YAML configuration because it added a layer without adding anything to reproduce.

14. The public Space is a Gradio Space on ZeroGPU hardware, not a Docker Space. A free account cannot create a Docker or CPU Space, so the demo and a `/plan` endpoint run as a Gradio app; the REST API runs locally and in the container that CI builds. The deploy job waits until the Space runs the new commit, not only until it reports running, because the running state can still be the old build.

## What I got wrong first

I expected shaping to help. It lost rovers, and I wrote it down as a negative result before finding that the potential was the cause (E1c).

I reported mean regret. The means were dominated by a few looping policies and told the wrong story about SARSA, the shield and the linear learner. The medians told a different one, and I reran three experiments.

I compared the shield only with Q-learning, on maps where the safe route is already the best. SARSA looked safe on its own. On the ridge maps, where the best policy accepts a risk, SARSA looped in a quarter of its runs and the shield removed every loop (E1b).

The first model-based learner without a map explored with epsilon-greedy under a pessimistic model and stayed on a bad route forever on 2 of 6 maps. The second used optimism with sparse replanning and left most actions untried. The version that works replans at least every 1,000 steps and keeps optimism for exploration only.

The first linear features described the landing cell only. A least-squares fit of such features to the optimal values still produced policies that cycled, because the risk of a route is paid at every step along it. Adding what is left on the shortest route from the landing cell fixed it.

I proposed a Docker Space for the demo. The account could not create one, and the Gradio route needed a placeholder function for ZeroGPU that the platform requires and the planner does not use.
