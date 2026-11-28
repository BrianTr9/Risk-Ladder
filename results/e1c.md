E1c: the route a shaping potential follows decides how many rovers the learner loses.

Q-learning on 10 dilemma maps x 5 seeds, 1,000,000 steps. Same step size as E1; the scales are not tuned.

| Shaping | Rovers lost while learning | Lost in the first 8% | Final regret: median [quartiles] | Runs that loop |
|---|---|---|---|---|
| no shaping | 1,252 ± 104 | 278 | 12.0% [8.0 to 24.0] | 3 of 50 |
| shortest route, scale 1 | 2,345 ± 150 | 550 | 13.0% [9.1 to 27.2] | 3 of 50 |
| shortest route, scale 2 | 2,815 ± 250 | 598 | 12.3% [9.4 to 30.9] | 3 of 50 |
| route clear of hazards, scale 1 | 486 ± 207 | 120 | 13.3% [9.0 to 32.9] | 5 of 50 |
| route clear of hazards, scale 2 | 252 ± 119 | 104 | 12.1% [9.2 to 26.5] | 3 of 50 |
