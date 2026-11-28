"""Shield and shaping as switches on the learners; shaping must not change the best policy."""

import dataclasses

import numpy as np
import pytest

from riskladder.agents.linear import LinearQ
from riskladder.agents.tabular import SARSA, QLearning
from riskladder.env import Layout, MapSpec, Noise, RoverEnv, generate
from riskladder.harness import Scorer, train
from riskladder.oracle import value_iteration
from riskladder.safety.shaping import Shaper
from riskladder.safety.shield import Shield

HAND = "#######\n#S..XG#\n#.~...#\n#######"
LAYOUT = Layout.from_text(HAND)
DETOUR = Layout.from_text("#########\n#S.....G#\n#..XXX..#\n#.......#\n#.......#\n#########")


def shaped_model(mdp, potential):
    """The exact MDP whose costs are the shaped costs (a test-only use of the hidden model)."""
    n, a = mdp.n_states, mdp.n_actions
    phi = potential[mdp.cells[:, 0], mdp.cells[:, 1]]
    ahead = (
        (mdp.P_stack @ phi).reshape(a, n).T
    )  # expected potential of the next state, 0 if absorbed
    return dataclasses.replace(mdp, cost=mdp.cost + ahead - phi[:, None])


@pytest.mark.parametrize("route", ["free", "safe"])
@pytest.mark.parametrize("p_slip", [0.0, 0.2])
@pytest.mark.parametrize("scale", [0.5, 1.0, 3.0])
def test_shaping_shifts_every_value_by_the_potential_and_keeps_the_best_policy(
    p_slip, scale, route
):
    spec = MapSpec(DETOUR, noise=Noise(p_slip=p_slip))
    mdp = RoverEnv(spec).model()
    potential = Shaper(scale, route).potential(DETOUR)
    original = value_iteration(mdp)
    shaped = value_iteration(shaped_model(mdp, potential))
    phi = potential[mdp.cells[:, 0], mdp.cells[:, 1]]
    assert pytest.approx(original.V - phi, abs=1e-6) == shaped.V
    from riskladder.oracle import Oracle

    oracle = Oracle(mdp)
    assert oracle.regret(shaped.policy) == pytest.approx(0.0, abs=1e-6)


def test_shaping_leaves_the_best_policy_unchanged_on_a_generated_map():
    spec = generate("field", seed=3, rows=12)
    mdp = RoverEnv(spec).model()
    shaped = value_iteration(shaped_model(mdp, Shaper(1.0).potential(spec.layout)))
    from riskladder.oracle import Oracle

    assert Oracle(mdp).regret(shaped.policy) == pytest.approx(0.0, abs=1e-6)


def test_the_potential_is_the_scaled_goal_distance_and_zero_at_absorbing_outcomes():
    shaper = Shaper(2.0)
    potential = shaper.potential(LAYOUT)
    assert potential[1, 1] == 12.0 and potential[2, 5] == 2.0  # 6 and 1 steps, scale 2
    assert Shaper.cost(potential, (1, 1), (1, 2), 1.0, terminated=True) == 1.0 - potential[1, 1]
    assert Shaper.cost(potential, (1, 1), (1, 2), 1.0, terminated=False) == pytest.approx(
        1.0 + potential[1, 2] - potential[1, 1]
    )


def test_a_move_towards_the_goal_gets_cheaper_and_away_gets_dearer():
    potential = Shaper(1.0).potential(DETOUR)
    closer = Shaper.cost(potential, (1, 1), (1, 2), 1.0, False)
    away = Shaper.cost(potential, (1, 2), (1, 1), 1.0, False)
    assert closer < 1.0 < away


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_a_bad_shaping_scale_is_rejected(bad):
    with pytest.raises(ValueError):
        Shaper(bad)


@pytest.mark.parametrize("cls", [QLearning, SARSA, LinearQ])
def test_no_action_the_shield_blocks_is_ever_taken_in_training(cls):
    spec = MapSpec(DETOUR, noise=Noise(p_slip=0.15))
    env = RoverEnv(spec, seed=0)
    allowed = Shield(p_max=0.4, theta=0.05).table(DETOUR)
    taken = []

    class Spy:
        layout, horizon = env.layout, env.horizon

        def reset(self):
            self.cell = env.reset()
            return self.cell

        def step(self, action):
            taken.append(allowed[(*self.cell, action)])
            result = env.step(action)
            self.cell = result.cell
            return result

    learner = cls(shield={"p_max": 0.4, "theta": 0.05}, eps_start=1.0, eps_end=0.3)
    train(learner, [Spy()], 6000, np.random.default_rng(0))
    assert len(taken) == 6000 and all(taken)


def test_the_shield_cuts_failures_while_learning():
    spec = MapSpec(DETOUR, noise=Noise(p_slip=0.15))
    failures = {}
    for name, shield in (("off", None), ("on", {"p_max": 0.4, "theta": 0.05})):
        env = RoverEnv(spec, seed=1)
        last = train(QLearning(shield=shield), env.sample_interface(), 20_000,
                     np.random.default_rng(1))[-1]  # fmt: skip
        failures[name] = last.train_failures
    assert failures["on"] < failures["off"]


def test_the_deployed_policy_respects_the_shield_only_when_asked():
    allowed = Shield().table(LAYOUT)
    cells = np.array([[1, 3]])  # moving east here walks into the hazard
    for at_run in (True, False):
        learner = QLearning(shield={"p_max": 0.4, "theta": 0.05}, shield_at_run=at_run)
        learner.begin(LAYOUT, 100, np.random.default_rng(0))
        learner.Q[1, 3] = 5.0
        learner.Q[1, 3, 1] = -9.0  # east looks best
        action = int(learner.greedy(LAYOUT, cells)[0])
        assert (not allowed[1, 3, action]) == (not at_run)


def test_a_blocked_action_is_never_the_bootstrap_target():
    learner = QLearning(alpha=1.0, shield={"p_max": 0.4, "theta": 0.05})
    learner.begin(LAYOUT, 100, np.random.default_rng(0))
    learner.Q[1, 3] = 5.0
    learner.Q[1, 3, 1] = -9.0  # blocked, and tempting
    learner.observe((1, 2), 1, 1.0, (1, 3), False, False)
    assert learner.Q[1, 2, 1] == pytest.approx(1.0 + 5.0)


@pytest.mark.parametrize("cls", [QLearning, SARSA, LinearQ])
def test_configs_are_plain_data_that_rebuild_the_learner(cls):
    learner = cls(shield={"p_max": 0.3, "theta": 0.02}, shaping=1.5, shield_at_run=False)
    config = learner.config
    assert config["shield"] == {"p_max": 0.3, "theta": 0.02}
    assert config["shaping"] == 1.5 and config["shield_at_run"] is False
    rebuilt = cls(**config)
    assert rebuilt.config == config


def test_shaped_q_values_equal_the_original_values_minus_the_potential():
    spec = MapSpec(LAYOUT, noise=Noise(p_slip=0.0))
    env = RoverEnv(spec, seed=0)
    learner = QLearning(alpha=0.5, shaping=1.0, eps_fraction=0.6)
    train(learner, env.sample_interface(), 30_000, np.random.default_rng(0))
    solution = value_iteration(env.model())
    potential = Shaper(1.0).potential(LAYOUT)
    for s, (r, c) in enumerate(env.model().cells):
        if learner.visits[r, c].min() > 0 and np.isfinite(solution.V[s]):
            assert learner.Q[r, c].min() + potential[r, c] == pytest.approx(solution.V[s], abs=1e-6)


def test_shaping_speeds_up_learning_on_a_larger_map():
    spec = generate("field", seed=5, rows=14)
    scorer = Scorer(RoverEnv(spec))
    regret = {}
    for scale in (None, 1.0):
        env = RoverEnv(spec, seed=0)
        learner = QLearning(alpha=0.5, shaping=scale)
        train(learner, env.sample_interface(), 40_000, np.random.default_rng(0))
        regret[scale] = scorer.score(learner).regret
    assert regret[1.0] < regret[None]


def test_the_safe_route_potential_is_higher_where_the_short_route_hugs_a_hazard():
    free = Shaper(1.0, "free").potential(DETOUR)
    safe = Shaper(1.0, "safe").potential(DETOUR)
    corridor = (1, 4)  # directly above the hazard strip: on the short route, not on the safe one
    assert safe[corridor] > free[corridor]
    assert (safe >= free - 1e-9).all()  # a route with more constraints is never shorter
    assert safe[1, 7] == 0 and free[1, 7] == 0  # the goal is the same


def test_an_unknown_route_is_rejected_and_the_route_reaches_the_learner_config():
    with pytest.raises(ValueError, match="route"):
        Shaper(1.0, "scenic")
    learner = QLearning(shaping=2.0, shaping_route="safe")
    assert learner.config["shaping_route"] == "safe" and QLearning(**learner.config).config == (
        learner.config
    )
    assert QLearning().config["shaping_route"] == "free"
