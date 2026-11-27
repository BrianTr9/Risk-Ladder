# How RiskLadder is tested

The results of this project are only worth something if the code computing them is right, so testing is layered. Each layer answers a different question, and none of them is trusted alone.

## The layers

| Question | Technique | Where |
|---|---|---|
| Does it do what the specification says, in detail? | Independent reference implementation written from `docs/env_spec.md` with plain loops (shares no code with the package), compared on every state and action | `tests/reference.py`, `tests/test_reference_implementation.py` |
| Does that hold across the configuration space, not just two settings? | Pairwise (2-way) covering array over nine inputs, 12 maps instead of 1,728, boundary values included | `tests/test_configuration_matrix.py` |
| Could code and reference share the same misreading of the spec? | Closed forms (gambler's ruin, a corridor worked out on paper) and metamorphic relations taken from domain knowledge: mirror and rotation symmetry, cost scaling, monotonicity in the failure cost, invariance to sealed regions | `tests/test_oracle_solve.py`, `tests/test_metamorphic.py` |
| Does it hold for inputs nobody thought of? | Property-based tests: round trip, invariants, idempotence, symmetry, value iteration versus policy iteration, exact evaluation versus rollouts | `tests/test_properties.py`, `tests/test_oracle_rollout.py` |
| Does it fail cleanly on bad input? | Boundary values with the single-invalid-input rule, and fuzzing of the `.map` parser (it may only raise `MapError`) | `tests/test_boundaries.py`, `tests/test_parser_fuzz.py` |
| Are the tests themselves any good? | Branch coverage gate in CI, and a mutation check that applies deliberate bugs and requires the suite to fail | `pyproject.toml`, `tools/mutation_check.py` |

## Why the reference alone is not enough

The package and the reference were both written from the same specification, so a misreading could appear in both and the comparison would still pass. This is the pseudo-oracle problem. The closed forms and the metamorphic relations do not depend on either implementation: a mirrored map must have the same values, doubling every cost must double every value, a larger failure cost can never lower the cost or raise the failure probability. A fault that preserved all of these would have to be an unusually well-behaved fault.

A relation that sounds right can still be wrong. Scaling all costs by `k` scales every value by `k` only if the failure cost scales too; `test_the_relation_that_looks_valid_but_is_not` records that trap.

## Determinism

Randomness is seeded everywhere a test depends on it. Property tests run in a `ci` profile that replays the same examples on every run, so a failure in CI is reproducible, never flaky. The `thorough` profile (200 examples) runs weekly and on demand with the mutation check (`.github/workflows/nightly.yml`).

## What this found

Checks like these only count if they find things.

Defects in the code, found and fixed:

- the dilemma map generator quietly produced weak dilemmas on very wide, short maps (now rejected with a clear error);
- the `.map` parser raised raw `TypeError`, `AttributeError` and `JSONDecodeError` on malformed headers, and accepted NaN costs and an unbounded horizon (found by fuzzing);
- `evaluate` and `rollout` accepted a horizon of 0, which the specification forbids (found by boundary analysis).

Gaps in the tests, found and closed:

- no map had a sealed region holding a goal, so the rule that the failure cost counts only cells reachable from the start was untested (a surviving mutant);
- one assertion could never fail, and the non-convergence paths of both solvers were never run (weak-oracle review and branch coverage);
- the map size limit was only tested with input that was invalid anyway (a surviving mutant);
- a statistical test divided by a zero standard deviation when every rollout was identical.

## Running it

```bash
uv run pytest                                   # fast suite
uv run pytest --cov                             # with the branch coverage gate used in CI
uv run pytest --hypothesis-profile=thorough     # 200 examples per property
uv run python tools/mutation_check.py --jobs 4  # do the tests catch deliberate bugs?
```

## Limits

- The mutation list is curated, not exhaustive. Mutants that cannot be killed because behaviour does not change are marked `equivalent` with the reason.
- The coverage threshold is a gate, not evidence of quality; the mutation check is the evidence.
- Statistical comparisons use five standard errors, so a false alarm is vanishingly rare but not impossible.
- Learners (a later phase) are stochastic and will be tested with statistical oracles over several seeds, plus metamorphic relations.
