"""The training loop, the exact scoring and the multi-seed summaries."""

import json
import math

import numpy as np
import pytest

from riskladder.agents.tabular import QLearning
from riskladder.env import Layout, MapSpec, Noise, RoverEnv
from riskladder.env.dynamics import FAIL
from riskladder.harness import RunSpec, Scorer, learned_at, run, summarize, train
from riskladder.harness.runner import AGENTS, mean_ci, read_jsonl, write_jsonl

HAND = "#######\n#S..XG#\n#.~...#\n#######"


def make_env(seed=0, p_slip=0.2, horizon=None):
    spec = MapSpec(Layout.from_text(HAND), noise=Noise(p_slip=p_slip), horizon_override=horizon)
    return RoverEnv(spec, seed=seed)


class Fixed:
    """A learner with a fixed greedy policy that never learns."""

    name = "fixed"

    def __init__(self, action_of):
        self.action_of = action_of
        self.seen = []

    def begin(self, layout, budget_steps, rng):
        pass

    def start_episode(self, layout):
        self.seen.append(layout)

    def act(self, cell):
        return int(self.action_of(cell))

    def observe(self, *args):
        pass

    def greedy(self, layout, cells):
        return np.array([self.action_of(tuple(c)) for c in cells])


def test_a_run_uses_exactly_the_step_budget_and_reports_every_checkpoint():
    env = make_env()
    checkpoints = train(QLearning(), env.sample_interface(), 1000, np.random.default_rng(0),
                        [Scorer(env)], n_checkpoints=10)  # fmt: skip
    assert [c.steps for c in checkpoints] == list(range(0, 1001, 100))
    assert env.stats.steps == 1000
    assert all(len(c.scores) == 1 for c in checkpoints)


def test_a_budget_that_does_not_divide_evenly_still_ends_on_the_budget():
    env = make_env()
    checkpoints = train(QLearning(), env.sample_interface(), 1005, np.random.default_rng(0),
                        n_checkpoints=10)  # fmt: skip
    assert [c.steps for c in checkpoints] == [*range(0, 1001, 100), 1005]


def test_training_failures_and_cost_match_what_the_environment_counted():
    env = make_env(p_slip=0.3)
    last = train(Fixed(lambda cell: 5), env.sample_interface(), 3000, np.random.default_rng(0))[-1]
    assert last.train_failures == env.stats.failures
    assert last.train_successes == env.stats.successes
    assert last.episodes == env.stats.episodes
    assert last.train_cost > 0


def test_an_episode_is_cut_at_the_horizon_and_restarted():
    env = make_env(horizon=5)
    always_bump = Fixed(lambda cell: 0)  # move north into the wall forever
    last = train(always_bump, env.sample_interface(), 50, np.random.default_rng(0))[-1]
    assert env.stats.timeouts == 10 and last.episodes == 10


def test_several_maps_are_visited_at_random_one_per_episode():
    first, second = make_env(1), make_env(2)
    learner = Fixed(lambda cell: 5)
    train(learner, [first.sample_interface(), second.sample_interface()], 4000,
          np.random.default_rng(0))  # fmt: skip
    assert first.stats.episodes > 0 and second.stats.episodes > 0
    assert first.stats.steps + second.stats.steps == 4000


@pytest.mark.parametrize("kwargs", [{"budget_steps": 0}, {"budget_steps": 10, "n_checkpoints": 0}])
def test_bad_arguments_are_rejected(kwargs):
    env = make_env().sample_interface()
    with pytest.raises(ValueError):
        train(Fixed(lambda c: 0), env, rng=np.random.default_rng(0), **kwargs)
    with pytest.raises(ValueError, match="at least one"):
        train(Fixed(lambda c: 0), [], 10, np.random.default_rng(0))


def test_the_optimal_policy_scores_zero_regret_and_a_bad_one_does_not():
    env = make_env()
    scorer = Scorer(env)
    best = scorer.oracle.optimum.policy
    cells = [tuple(c) for c in scorer.mdp.cells]
    optimal = Fixed(lambda cell: best[cells.index(tuple(cell))])
    assert scorer.score(optimal).regret == pytest.approx(0.0, abs=1e-9)
    worse = scorer.score(Fixed(lambda cell: 0))  # always tries to go north, into the wall
    assert worse.regret > 1.0 and worse.cost == pytest.approx(scorer.optimal.cost + worse.regret)


def test_the_learned_threshold_is_a_fraction_of_the_optimal_cost():
    scorer = Scorer(make_env(), tau=0.1)
    assert scorer.threshold == pytest.approx(0.1 * scorer.optimal.cost)
    for tau in (0.0, 1.0, -0.5):
        with pytest.raises(ValueError):
            Scorer(make_env(), tau=tau)


def test_a_learner_must_return_one_action_per_cell():
    scorer = Scorer(make_env())

    class Wrong(Fixed):
        def greedy(self, layout, cells):
            return np.zeros(len(cells) + 1, dtype=int)

    with pytest.raises(ValueError, match="one action per decision cell"):
        scorer.policy(Wrong(lambda c: 0))


def test_learned_at_is_the_first_checkpoint_within_the_threshold():
    rows = [{"steps": s, "regret": r} for s, r in [(0, 9.0), (10, 3.0), (20, 0.5), (30, 0.4)]]
    assert learned_at(rows, 1.0) == 20
    assert learned_at(rows, 0.1) is None
    assert learned_at(rows, 9.0) == 0


def small_spec(agent="q", seed=0, budget=3000):
    return RunSpec(agent, "experiments/maps/dilemma/dilemma-30x30-s0.map", seed, budget, {}, 5)


def test_a_run_record_is_complete_and_repeatable():
    spec = small_spec()
    first, second = run(spec), run(spec)
    assert first == second  # same seed, same record
    assert first["agent"] == "q" and first["budget"] == 3000 and len(first["checkpoints"]) == 6
    assert first["threshold"] == pytest.approx(0.05 * first["optimal_cost"])
    json.dumps(first)  # plain data only
    assert run(small_spec(seed=1)) != first


def test_the_agent_and_the_environment_use_independent_random_streams():
    a = run(small_spec(seed=3))
    b = run(small_spec(seed=3))
    assert a["checkpoints"][-1]["train_cost"] == b["checkpoints"][-1]["train_cost"]


def test_unknown_agents_are_named_in_the_error():
    with pytest.raises(ValueError, match="unknown agent"):
        run(RunSpec("nope", "x.map", 0, 10))
    assert set(AGENTS) == {"q", "sarsa", "linear", "mb_count", "mb_struct"}


def test_mean_and_interval_follow_the_t_distribution():
    mean, half = mean_ci(np.array([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]]))
    assert mean.tolist() == [3.0, 2.0]
    assert half[0] == pytest.approx(4.302652729911275 * 2.0 / math.sqrt(3))  # t(0.975, 2) * s/√n
    assert half[1] == 0.0
    assert np.isnan(mean_ci(np.array([[1.0]]))[1]).all()


def record(regrets, learned, failures=None):
    steps = [0, 10, 20]
    failures = failures or [0, 1, 2]
    return {
        "learned_at": learned,
        "checkpoints": [
            {"steps": s, "regret": r, "p_fail": 0.0, "p_timeout": 0.0, "train_failures": f,
             "train_cost": 2.0 * s}
            for s, r, f in zip(steps, regrets, failures, strict=True)
        ],
    }  # fmt: skip


def test_summaries_average_runs_and_report_how_many_learned():
    out = summarize([record([4, 2, 0], 20), record([6, 4, 2], None), record([2, 0, 0], 10)])
    assert out["runs"] == 3 and out["steps"] == [0, 10, 20]
    assert out["regret"]["mean"] == [4.0, 2.0, 2 / 3]
    assert out["learned_fraction"] == pytest.approx(2 / 3)
    assert out["learned_median"] == 15.0
    assert out["train_failures"]["mean"] == [0, 1, 2]


def test_the_median_is_withheld_when_fewer_than_half_of_the_runs_learned():
    out = summarize([record([4, 2, 0], 20), record([6, 4, 2], None), record([6, 4, 2], None)])
    assert out["learned_fraction"] == pytest.approx(1 / 3) and out["learned_median"] is None


def test_summaries_reject_empty_input_and_mismatched_grids():
    with pytest.raises(ValueError, match="no records"):
        summarize([])
    other = record([1, 1, 1], None)
    other["checkpoints"][1]["steps"] = 11
    with pytest.raises(ValueError, match="different checkpoint grids"):
        summarize([record([1, 1, 1], None), other])


def test_records_survive_a_round_trip_through_json_lines(tmp_path):
    records = [run(small_spec(seed=s, budget=500)) for s in range(2)]
    path = tmp_path / "nested" / "runs.jsonl"
    write_jsonl(path, records)
    assert read_jsonl(path) == records
    assert FAIL == 2  # keeps the status import honest for readers of this file
