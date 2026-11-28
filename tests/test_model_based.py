"""Model-based learners: the statistics they keep, the plans they make, and that they learn."""

import numpy as np
import pytest

from riskladder.agents.model_based import MBCount, MBStruct, Replanner, _wilson_upper
from riskladder.env import Costs, Layout, MapSpec, Noise, RoverEnv
from riskladder.env.dynamics import ACTIONS
from riskladder.harness import Scorer, train

HAND = "#######\n#S..XG#\n#.~...#\n#######"
LAYOUT = Layout.from_text(HAND)
DETOUR = Layout.from_text("#########\n#S.....G#\n#..XXX..#\n#.......#\n#.......#\n#########")
ACTION = {a.name: i for i, a in enumerate(ACTIONS)}


def started(agent, layout=LAYOUT, budget=1000):
    agent.begin(layout, budget, np.random.default_rng(0))
    return agent


def test_the_replan_schedule_grows_geometrically_with_a_minimum_gap():
    schedule = Replanner(first=100, growth=1.5, gap=10)
    assert schedule.due(100) and not schedule.due(99)
    schedule.planned(100)
    assert schedule.next == 150
    schedule.planned(1000)
    assert schedule.next == 1500
    schedule.next = 0
    schedule.planned(5)
    assert schedule.next == 15  # the gap wins while the step count is small
    capped = Replanner(first=100, growth=2.0, gap=10, longest=50)
    capped.planned(1000)
    assert capped.next == 1050  # but never more than `longest` steps ahead
    for bad in ({"first": 0}, {"growth": 1.0}, {"gap": 0}, {"gap": 10, "longest": 5}):
        with pytest.raises(ValueError):
            Replanner(**bad)


def test_counts_record_each_transition_and_the_cost_of_ending():
    agent = started(MBCount())
    agent.observe((1, 1), 1, 1.0, (1, 2), False, False)
    agent.observe((1, 1), 1, 1.0, (1, 2), False, False)
    agent.observe((1, 1), 1, 251.0, (1, 4), True, False)
    s = 1 * LAYOUT.cols + 1
    assert agent.visits[s, 1] == 3 and agent.ended[s, 1] == 1
    assert agent.moved[s, 1, 1 * LAYOUT.cols + 2] == 2
    assert agent.cost_sum[s, 1] == pytest.approx(253.0)


def test_the_empirical_model_has_proper_probabilities_and_untried_actions_depend_on_the_mode():
    agent = started(MBCount())
    for _ in range(3):
        agent.observe((1, 1), 1, 1.0, (1, 2), False, False)
    agent.observe((1, 1), 1, 251.0, (1, 4), True, False)
    model = agent.model(optimistic=False)
    s = 1 * LAYOUT.cols + 1
    row = model.P_stack[1 * model.n_states + s]
    assert row.sum() + model.p_success[s, 1] == pytest.approx(1.0)
    assert model.p_success[s, 1] == pytest.approx(0.25) and model.cost[s, 1] == pytest.approx(63.5)
    assert model.p_success[s, 0] == 1.0 and model.cost[s, 0] > model.cost[s, 1]  # never tried
    hopeful = agent.model(optimistic=True)
    assert hopeful.p_success[s, 0] == 1.0 and hopeful.cost[s, 0] == 0.0  # drawn to try it
    assert hopeful.cost[s, 1] == model.cost[s, 1]  # what is known is the same in both


def test_known_after_delays_trust_in_a_pair():
    agent = started(MBCount(known_after=3))
    agent.observe((1, 1), 1, 1.0, (1, 2), False, False)
    assert agent.model(optimistic=False).p_success[1 * LAYOUT.cols + 1, 1] == 1.0  # unknown


@pytest.mark.parametrize("bad", [{"known_after": 0}, {"growth": 0.5}, {"longest": 10}])
def test_bad_count_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        MBCount(**bad)


@pytest.mark.parametrize("bad", [{"delta": 2.0}, {"p_max": 0.0}, {"z": -1.0}, {"ridge": 0.0}])
def test_bad_struct_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        MBStruct(**bad)


def test_the_slip_estimate_follows_the_branch_the_rover_ended_in():
    agent = started(MBStruct())
    # move_E from (2,2) goes to (2,3) when it works; a slip north reaches (1,2), south is a rock
    for _ in range(3):
        agent.observe((2, 2), ACTION["move_E"], 1.0, (2, 3), False, False)
    agent.observe((2, 2), ACTION["move_E"], 1.0, (1, 2), False, False)
    assert agent.trials[0] == 4 and agent.slips[0] == pytest.approx(1.0)
    assert agent.slip_estimate()[0] == pytest.approx(0.25)
    assert agent.trials[1] == 0  # no dash seen yet


def test_an_ambiguous_observation_is_shared_between_the_branches_that_fit_it():
    agent = started(MBStruct())
    agent.slips[:], agent.trials[:] = [2.0, 0.0], [10.0, 0.0]  # believes moves slip 20%
    # move_W from (1,1): the wall blocks the intended move, and the slip north is blocked too, so
    # staying in place fits two branches; the third slips south
    agent.observe((1, 1), ACTION["move_W"], 3.0, (1, 1), False, False)
    assert agent.trials[0] == 11
    assert 2.0 < agent.slips[0] < 3.0  # part of one slip, in proportion to the prior


def test_costs_are_recovered_by_regression_on_the_branch_features():
    spec = MapSpec(DETOUR, noise=Noise(p_slip=0.2), costs=Costs(1.0, 1.5, 0.7, 2.5),
                   c_fail_override=40.0)  # fmt: skip
    env = RoverEnv(spec, seed=0)
    agent = MBStruct(eps_start=1.0, eps_end=1.0)  # random actions, to see every kind of cost
    train(agent, env.sample_interface(), 20_000, np.random.default_rng(0))
    move, dash, _, collision, fail = agent.cost_estimate()
    assert (move, dash, collision, fail) == pytest.approx((1.0, 1.5, 2.5, 40.0), abs=0.15)


def test_slip_probabilities_are_recovered_with_a_confidence_bound_that_covers_the_truth():
    spec = MapSpec(DETOUR, noise=Noise(p_slip=0.15, dash_slip_mult=2.0))
    env = RoverEnv(spec, seed=1)
    agent = MBStruct(eps_start=1.0, eps_end=1.0)
    train(agent, env.sample_interface(), 30_000, np.random.default_rng(1))
    assert agent.slip_estimate() == pytest.approx([0.15, 0.30], abs=0.03)
    assert (agent.slip_bound() >= agent.slip_estimate() - 1e-9).all()
    assert (agent.slip_bound() >= [0.15, 0.30]).all()


def test_before_any_data_the_upper_bound_is_the_assumed_maximum():
    agent = started(MBStruct(p_max=0.35))
    assert agent.slip_bound().tolist() == [0.35, 0.35]
    assert agent.slip_estimate().tolist() == [0.0, 0.0]


def test_wilson_bound_properties():
    assert _wilson_upper(0, 0, 1.64) == 1.0
    for k, n in [(0, 10), (3, 10), (50, 100), (10, 10)]:
        bound = _wilson_upper(k, n, 1.64)
        assert k / n <= bound <= 1.0
    assert _wilson_upper(5, 100, 1.64) < _wilson_upper(5, 20, 1.64)  # more data, tighter
    assert _wilson_upper(5, 20, 0.0) == pytest.approx(0.25)


def test_a_model_based_learner_is_tied_to_its_map():
    agent = started(MBCount())
    with pytest.raises(ValueError, match="tied to the map"):
        agent.start_episode(Layout.from_text("#####\n#SG.#\n#####"))


def test_mb_count_reaches_the_optimum_on_a_small_deterministic_map():
    env = RoverEnv(MapSpec(LAYOUT, noise=Noise(p_slip=0.0)), seed=0)
    agent = MBCount()
    train(agent, env.sample_interface(), 20_000, np.random.default_rng(0))
    assert Scorer(env).score(agent).regret == pytest.approx(0.0, abs=1e-6)


def test_mb_struct_needs_far_fewer_samples_than_mb_count_and_than_q_learning():
    spec = MapSpec(DETOUR, noise=Noise(p_slip=0.15))
    scorer = Scorer(RoverEnv(spec))
    regret = {}
    for name, agent in (("struct", MBStruct()), ("count", MBCount())):
        env = RoverEnv(spec, seed=0)
        train(agent, env.sample_interface(), 3000, np.random.default_rng(0))
        regret[name] = scorer.score(agent).regret
    assert regret["struct"] < 0.1 * scorer.optimal.cost < regret["count"]


def test_the_constrained_planner_keeps_failures_down_while_learning():
    spec = MapSpec(DETOUR, noise=Noise(p_slip=0.15))
    failures = {}
    for cmdp in (False, True):
        env = RoverEnv(spec, seed=2)
        last = train(MBStruct(cmdp=cmdp), env.sample_interface(), 4000, np.random.default_rng(2))[
            -1
        ]
        failures[cmdp] = last.train_failures
    assert failures[True] <= failures[False]


def test_the_scorer_evaluates_the_mixture_the_constrained_planner_would_run():
    spec = MapSpec(DETOUR, noise=Noise(p_slip=0.15))
    env = RoverEnv(spec, seed=3)
    scorer = Scorer(env)
    agent = MBStruct(cmdp=True, delta=0.02)
    train(agent, env.sample_interface(), 3000, np.random.default_rng(3))
    parts = agent.mixture(spec.layout, scorer.mdp.cells)
    assert sum(w for w, _ in parts) == pytest.approx(1.0)
    assert scorer.score(agent).p_fail <= 0.02 + 1e-6
    assert agent.greedy(spec.layout, scorer.mdp.cells).shape == (scorer.mdp.n_states,)


def test_the_learners_do_not_plan_before_the_schedule_says_so():
    agent = started(MBCount(first_plan=50))
    agent.act((1, 1))  # the very first action needs a plan
    assert agent._parts is not None
    plan = agent._parts
    for _ in range(10):
        agent.observe((1, 1), 1, 1.0, (1, 2), False, False)
        agent.act((1, 1))
    assert agent._parts is plan  # still the old plan: the schedule has not fired


def test_an_action_counts_as_known_exactly_when_it_has_been_tried_known_after_times():
    agent = started(MBCount(known_after=3))
    s = 1 * LAYOUT.cols + 1
    for tries in (1, 2, 3):
        agent.observe((1, 1), 1, 1.0, (1, 2), False, False)
        known = agent.model(optimistic=False).p_success[s, 1] != 1.0
        assert known == (tries >= 3)


def test_greedy_reports_the_heavier_part_of_a_mixture():
    agent = started(MBStruct(cmdp=True))
    light, heavy = np.zeros(LAYOUT.rows * LAYOUT.cols, int), np.ones(LAYOUT.rows * LAYOUT.cols, int)
    cells = np.array([[1, 1], [1, 2]])
    agent.mixture = lambda layout, cells: [(0.2, light[:2]), (0.8, heavy[:2])]
    assert agent.greedy(LAYOUT, cells).tolist() == [1, 1]
