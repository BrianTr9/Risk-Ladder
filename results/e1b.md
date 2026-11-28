E1b: does the shield help SARSA, and what does caution cost?

Final regret is a fraction of the optimal cost and is heavy-tailed (a policy that loops pays until the horizon), so the median and quartiles come first. A run 'loops' when its final policy times out with probability above 0.5%.

dilemma maps: 10 maps x 5 seeds, 1,000,000 steps; step sizes Q 0.5, SARSA 0.3, shaping scale 2.0 (tuned on dilemma seeds 100-104).

| Configuration | Rovers lost while learning | Final regret: median [quartiles] (mean) | Runs that loop | Final P(fail) | Learned |
|---|---|---|---|---|---|
| Q | 1,252 ± 104 | 12.0% [8.0 to 24.0] (mean 56%) | 3 of 50 | 0.0% (optimum 0.0%) | 12% |
| Q + shield | 239 ± 17 | 12.0% [8.9 to 32.9] (mean 42%) | 1 of 50 | 0.0% (optimum 0.0%) | 14% |
| SARSA | 168 ± 28 | 6.7% [5.5 to 9.8] (mean 11%) | 0 of 50 | 0.0% (optimum 0.0%) | 72% |
| SARSA + shaping | 269 ± 33 | 11.5% [8.7 to 18.1] (mean 21%) | 0 of 50 | 0.0% (optimum 0.0%) | 8% |
| SARSA + shield | 68 ± 7 | 6.7% [5.2 to 8.9] (mean 27%) | 1 of 50 | 0.0% (optimum 0.0%) | 80% |
| SARSA + shield + shaping | 114 ± 11 | 9.1% [7.8 to 11.3] (mean 36%) | 2 of 50 | 0.0% (optimum 0.0%) | 10% |
| MB-struct + CMDP | 1 ± 0 | 0.0% [0.0 to 0.0] (mean 0%) | 0 of 50 | 0.0% (optimum 0.0%) | 100% |

ridge maps: 12 maps x 3 seeds, 300,000 steps; step sizes Q 0.5, SARSA 0.1, shaping scale 2.0 (tuned on ridge seeds 3100-3103).

| Configuration | Rovers lost while learning | Final regret: median [quartiles] (mean) | Runs that loop | Final P(fail) | Learned |
|---|---|---|---|---|---|
| Q | 7,245 ± 345 | 16.0% [13.3 to 22.4] (mean 29%) | 1 of 36 | 1.2% (optimum 3.0%) | 31% |
| Q + shield | 0 ± 0 | 14.2% [11.9 to 17.8] (mean 15%) | 0 of 36 | 0.0% (optimum 3.0%) | 28% |
| SARSA | 4,507 ± 459 | 20.2% [10.5 to 322.0] (mean 438%) | 9 of 36 | 2.1% (optimum 3.0%) | 28% |
| SARSA + shaping | 4,878 ± 515 | 18.1% [11.0 to 172.4] (mean 471%) | 9 of 36 | 2.1% (optimum 3.0%) | 47% |
| SARSA + shield | 0 ± 0 | 7.4% [5.3 to 10.3] (mean 8%) | 0 of 36 | 0.0% (optimum 3.0%) | 25% |
| SARSA + shield + shaping | 0 ± 0 | 7.1% [5.3 to 10.4] (mean 8%) | 0 of 36 | 0.0% (optimum 3.0%) | 22% |
| MB-struct + CMDP | 121 ± 6 | 2.6% [1.8 to 3.1] (mean 3%) | 0 of 36 | 1.0% (optimum 3.0%) | 92% |

Slip sweep on 5 dilemma maps x 3 seeds, 1,000,000 steps (the shield still assumes at most 0.4 for any action):

True slip 0.05:

| Configuration | Rovers lost while learning | Final regret: median [quartiles] (mean) | Runs that loop | Final P(fail) | Learned |
|---|---|---|---|---|---|
| Q | 2,628 ± 224 | 19.8% [16.4 to 27.7] (mean 24%) | 2 of 15 | 0.0% (optimum 0.0%) | 0% |
| SARSA | 113 ± 13 | 9.6% [8.1 to 11.0] (mean 10%) | 0 of 15 | 0.0% (optimum 0.0%) | 53% |
| SARSA + shield | 10 ± 2 | 8.2% [7.5 to 11.2] (mean 9%) | 0 of 15 | 0.0% (optimum 0.0%) | 40% |
| MB-struct + CMDP | 1 ± 0 | 0.0% [0.0 to 0.0] (mean 0%) | 0 of 15 | 0.0% (optimum 0.0%) | 100% |

True slip 0.3:

| Configuration | Rovers lost while learning | Final regret: median [quartiles] (mean) | Runs that loop | Final P(fail) | Learned |
|---|---|---|---|---|---|
| Q | 1,404 ± 245 | 40.9% [31.1 to 46.8] (mean 38%) | 0 of 15 | 0.0% (optimum 0.0%) | 0% |
| SARSA | 216 ± 30 | 16.7% [13.0 to 19.8] (mean 18%) | 0 of 15 | 0.0% (optimum 0.0%) | 20% |
| SARSA + shield | 154 ± 14 | 16.0% [11.5 to 23.3] (mean 19%) | 0 of 15 | 0.0% (optimum 0.0%) | 13% |
| MB-struct + CMDP | 1 ± 0 | 35.3% [35.2 to 37.2] (mean 37%) | 0 of 15 | 0.0% (optimum 0.0%) | 0% |
