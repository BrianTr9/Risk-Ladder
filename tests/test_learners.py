"""Q-learning, SARSA and linear Q: the updates, the schedules and learning a tiny map."""

import numpy as np
import pytest

from riskladder.agents.linear import LinearQ
from riskladder.agents.tabular import SARSA, QLearning
from riskladder.env import Layout, MapSpec, Noise, RoverEnv
from riskladder.env.dynamics import N_ACTIONS
from riskladder.harness import Scorer, train

HAND = "#######\n#S..XG#\n#.~...#\n#######"
LAYOUT = Layout.from_text(HAND)


def started(agent, budget=1000, seed=0):
    agent.begin(LAYOUT, budget, np.random.default_rng(seed))
    return agent


@pytest.mark.parametrize("cls", [QLearning, SARSA])
@pytest.mark.parametrize(
    "bad",
    [{"alpha": 0.0}, {"alpha": 1.5}, {"alpha_decay": -1.0}, {"eps_end": 0.9, "eps_start": 0.5},
     {"eps_start": 1.5}, {"eps_fraction": 0.0}],
)  # fmt: skip
def test_bad_settings_are_rejected(cls, bad):
    with pytest.raises(ValueError):
        cls(**bad)


@pytest.mark.parametrize("bad", [{"alpha": 0.0}, {"alpha_decay": -1.0}, {"eps_fraction": 2.0}])
def test_linear_bad_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        LinearQ(**bad)


def test_exploration_falls_linearly_then_stays_at_its_floor():
    agent = started(QLearning(eps_start=1.0, eps_end=0.1, eps_fraction=0.5), budget=1000)
    assert agent.epsilon() == pytest.approx(1.0)
    agent.t = 250
    assert agent.epsilon() == pytest.approx(0.55)
    agent.t = 500
    assert agent.epsilon() == pytest.approx(0.1)
    agent.t = 900
    assert agent.epsilon() == pytest.approx(0.1)


def test_q_learning_moves_the_estimate_a_step_towards_the_sampled_cost():
    agent = started(QLearning(alpha=0.5))
    agent.observe((1, 1), 3, 4.0, (1, 2), False, False)  # next cell still has Q = 0
    assert agent.Q[1, 1, 3] == pytest.approx(2.0)
    agent.Q[1, 2] = [5, 6, 7, 8, 9, 10, 11, 12]
    agent.observe((1, 1), 3, 4.0, (1, 2), False, False)  # bootstraps from the best next action
    assert agent.Q[1, 1, 3] == pytest.approx(2.0 + 0.5 * (4.0 + 5.0 - 2.0))


def test_a_terminal_step_uses_the_cost_alone_and_a_cut_at_the_horizon_does_not():
    agent = started(QLearning(alpha=1.0))
    agent.Q[1, 2] = 100.0
    agent.observe((1, 1), 0, 7.0, (1, 2), True, False)
    assert agent.Q[1, 1, 0] == pytest.approx(7.0)
    agent.observe((1, 1), 1, 7.0, (1, 2), False, True)
    assert agent.Q[1, 1, 1] == pytest.approx(107.0)


def test_step_size_shrinks_with_the_number_of_visits():
    agent = started(QLearning(alpha=0.5, alpha_decay=1.0))
    agent.observe((1, 1), 0, 8.0, (1, 2), True, False)  # step 0.5
    assert agent.Q[1, 1, 0] == pytest.approx(4.0)
    agent.observe((1, 1), 0, 8.0, (1, 2), True, False)  # step 0.25 after one visit
    assert agent.Q[1, 1, 0] == pytest.approx(4.0 + 0.25 * 4.0)
    assert agent.visits[1, 1, 0] == 2


def test_sarsa_bootstraps_from_the_action_it_will_really_take():
    agent = started(SARSA(alpha=1.0, eps_start=0.0, eps_end=0.0))
    agent.Q[1, 2] = [9, 3, 8, 8, 8, 8, 8, 8]  # greedy next action is 1 with Q = 3
    agent.observe((1, 1), 0, 2.0, (1, 2), False, False)
    assert agent.Q[1, 1, 0] == pytest.approx(5.0)
    assert agent.act((1, 2)) == 1  # the action used in the target is the one it takes
    assert agent._next is None


def test_sarsa_forgets_its_pending_action_when_the_episode_is_cut_or_restarted():
    agent = started(SARSA(alpha=1.0, eps_start=0.0, eps_end=0.0))
    agent.observe((1, 1), 0, 2.0, (1, 2), False, True)
    assert agent._next is None
    agent.observe((1, 1), 0, 2.0, (1, 2), False, False)
    agent.start_episode(LAYOUT)
    assert agent._next is None


def test_a_tabular_learner_refuses_a_map_of_another_shape():
    agent = started(QLearning())
    with pytest.raises(ValueError, match="tied to the map"):
        agent.start_episode(Layout.from_text("#####\n#SG.#\n#####"))


def test_greedy_picks_the_lowest_cost_action_in_each_cell():
    agent = started(QLearning())
    agent.Q[1, 1] = [3, 1, 2, 5, 5, 5, 5, 5]
    agent.Q[2, 1] = [4, 4, 0.5, 4, 4, 4, 4, 4]
    assert agent.greedy(LAYOUT, np.array([[1, 1], [2, 1]])).tolist() == [1, 2]


def test_linear_update_is_a_normalised_step_towards_the_target():
    agent = started(LinearQ(alpha=1.0))
    phi = agent.table(LAYOUT)[1, 1, 0]
    agent.observe((1, 1), 0, 5.0, (1, 2), True, False)
    assert phi @ agent.w == pytest.approx(5.0, rel=1e-6)  # alpha = 1 lands on the target


def test_linear_bootstrap_never_goes_below_zero():
    agent = started(LinearQ(alpha=0.5))
    agent.w[:] = -10.0  # an extrapolation that dips below zero
    phi = agent.table(LAYOUT)[1, 1, 0]
    before = phi @ agent.w
    agent.observe((1, 1), 0, 1.0, (1, 2), False, False)
    assert phi @ agent.w == pytest.approx(before + 0.5 * (1.0 - before), rel=1e-6)


def test_linear_greedy_shares_one_weight_vector_across_maps():
    other = Layout.from_text("######\n#S.XG#\n#....#\n######")
    agent = started(LinearQ())
    agent.w[:] = 0.0
    agent.w[1] = 10.0  # cost grows with the distance still to go
    first = agent.greedy(LAYOUT, np.array([[1, 1]]))
    second = agent.greedy(other, np.array([[1, 1]]))
    assert first.shape == second.shape == (1,)
    assert agent.table(other) is agent.table(other)  # cached per layout


def deterministic_env(seed=0):
    return RoverEnv(MapSpec(LAYOUT, noise=Noise(p_slip=0.0)), seed=seed)


def test_q_learning_finds_the_optimal_policy_on_a_deterministic_map():
    env = deterministic_env()
    scorer = Scorer(env)
    agent = QLearning(alpha=0.5, eps_fraction=0.6)
    train(agent, env.sample_interface(), 30_000, np.random.default_rng(1), [])
    assert scorer.score(agent).regret == pytest.approx(0.0, abs=1e-9)


def test_sarsa_learns_a_safe_policy_that_may_cost_a_little_more():
    """SARSA values the actions of its own exploring policy, so it keeps away from the hazard
    (the classic cliff-walking behaviour) and can settle on a slightly longer route."""
    env = deterministic_env()
    scorer = Scorer(env)
    agent = SARSA(alpha=0.5, eps_fraction=0.6)
    train(agent, env.sample_interface(), 30_000, np.random.default_rng(1), [])
    result = scorer.score(agent)
    assert result.p_fail == 0.0 and result.p_timeout == 0.0
    assert 0.0 <= result.regret <= 2.0


def test_linear_q_learns_a_safe_policy_on_the_same_map():
    env = deterministic_env()
    scorer = Scorer(env)
    agent = LinearQ(alpha=0.1)
    train(agent, env.sample_interface(), 30_000, np.random.default_rng(1), [])
    assert scorer.score(agent).p_fail == 0.0
    assert scorer.score(agent).p_timeout == 0.0


def test_the_same_seed_gives_the_same_learner():
    runs = []
    for _ in range(2):
        env = deterministic_env(seed=5)
        agent = QLearning()
        train(agent, env.sample_interface(), 2000, np.random.default_rng(9), [])
        runs.append(agent.Q.copy())
    assert np.array_equal(*runs)


def test_every_action_is_tried_under_full_exploration():
    agent = started(QLearning(eps_start=1.0, eps_end=1.0))
    assert {agent.act((1, 1)) for _ in range(300)} == set(range(N_ACTIONS))
