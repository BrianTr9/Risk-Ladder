E1b: does the shield help SARSA, and what does caution cost?

dilemma maps: 10 maps x 5 seeds, 1,000,000 steps; step sizes Q 0.5, SARSA 0.3, shaping scale 2.0 (tuned on dilemma seeds 100-104).

| Configuration | Rovers lost while learning | Final regret | Final P(fail) | Learned |
|---|---|---|---|---|
| Q | 1,252 ± 104 | 56.1% ± 50.6 | 0.0% (optimum 0.0%) | 12% |
| Q + shield | 239 ± 17 | 41.8% ± 39.4 | 0.0% (optimum 0.0%) | 14% |
| SARSA | 168 ± 28 | 11.0% ± 3.3 | 0.0% (optimum 0.0%) | 72% |
| SARSA + shaping | 269 ± 33 | 21.4% ± 8.2 | 0.0% (optimum 0.0%) | 8% |
| SARSA + shield | 68 ± 7 | 27.1% ± 31.9 | 0.0% (optimum 0.0%) | 80% |
| SARSA + shield + shaping | 114 ± 11 | 35.5% ± 40.6 | 0.0% (optimum 0.0%) | 10% |
| MB-struct + CMDP | 1 ± 0 | 0.0% ± 0.1 | 0.0% (optimum 0.0%) | 100% |

ridge maps: 12 maps x 3 seeds, 300,000 steps; step sizes Q 0.5, SARSA 0.1, shaping scale 2.0 (tuned on ridge seeds 3100-3103).

| Configuration | Rovers lost while learning | Final regret | Final P(fail) | Learned |
|---|---|---|---|---|
| Q | 7,245 ± 345 | 29.1% ± 23.8 | 1.2% (optimum 3.0%) | 31% |
| Q + shield | 0 ± 0 | 15.0% ± 1.6 | 0.0% (optimum 3.0%) | 28% |
| SARSA | 4,507 ± 459 | 437.7% ± 287.4 | 2.1% (optimum 3.0%) | 28% |
| SARSA + shaping | 4,878 ± 515 | 470.6% ± 329.0 | 2.1% (optimum 3.0%) | 47% |
| SARSA + shield | 0 ± 0 | 8.3% ± 1.4 | 0.0% (optimum 3.0%) | 25% |
| SARSA + shield + shaping | 0 ± 0 | 8.2% ± 1.3 | 0.0% (optimum 3.0%) | 22% |
| MB-struct + CMDP | 121 ± 6 | 2.7% ± 0.5 | 1.0% (optimum 3.0%) | 92% |

Slip sweep on 5 dilemma maps x 3 seeds, 1,000,000 steps (the shield still assumes at most 0.4 for any action):

True slip 0.05:

| Configuration | Rovers lost while learning | Final regret | Final P(fail) | Learned |
|---|---|---|---|---|
| Q | 2,628 ± 224 | 24.1% ± 7.0 | 0.0% (optimum 0.0%) | 0% |
| SARSA | 113 ± 13 | 9.7% ± 1.3 | 0.0% (optimum 0.0%) | 53% |
| SARSA + shield | 10 ± 2 | 9.0% ± 1.1 | 0.0% (optimum 0.0%) | 40% |
| MB-struct + CMDP | 1 ± 0 | 0.0% ± 0.0 | 0.0% (optimum 0.0%) | 100% |

True slip 0.3:

| Configuration | Rovers lost while learning | Final regret | Final P(fail) | Learned |
|---|---|---|---|---|
| Q | 1,404 ± 245 | 37.6% ± 7.5 | 0.0% (optimum 0.0%) | 0% |
| SARSA | 216 ± 30 | 17.9% ± 4.7 | 0.0% (optimum 0.0%) | 20% |
| SARSA + shield | 154 ± 14 | 18.7% ± 6.1 | 0.0% (optimum 0.0%) | 13% |
| MB-struct + CMDP | 1 ± 0 | 36.6% ± 1.9 | 0.0% (optimum 0.0%) | 0% |
