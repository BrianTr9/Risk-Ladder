from collections import Counter

import pytest
from scipy.stats import chisquare

from riskladder.env import FAIL, RUNNING, SUCCESS, MapSpec, RoverEnv, outcomes

N_SAMPLES = 100_000


@pytest.mark.parametrize(
    ("cell", "action"),
    [((1, 1), 1), ((1, 2), 5), ((2, 5), 4), ((2, 2), 7)],  # move_E, dash_E, dash_N, dash_W
)
def test_sampler_matches_model(hand_spec, cell, action):
    """Key test of the phase: sampled frequencies follow the exact outcome distribution."""
    env = RoverEnv(hand_spec, seed=123)
    expected = {
        (o.cell, round(o.cost, 9), o.status): o.prob for o in outcomes(hand_spec, cell, action)
    }
    observed: Counter = Counter()
    for _ in range(N_SAMPLES):
        env.reset(start=cell)
        result = env.step(action)
        observed[(result.cell, round(result.cost, 9), result.status)] += 1
    assert set(observed) <= set(expected), "sampler produced an outcome the model does not have"
    keys = sorted(expected)
    f_obs = [observed[k] for k in keys]
    f_exp = [expected[k] * N_SAMPLES for k in keys]
    assert chisquare(f_obs, f_exp).pvalue > 0.01


def test_episode_lifecycle_and_stats(hand_spec_deterministic):
    env = RoverEnv(hand_spec_deterministic, seed=0)
    env.reset()
    for action in (2, 1, 1, 1, 1):  # S, E, E, E, E along the lower row
        result = env.step(action)
    assert result.status == RUNNING and result.cell == (2, 5)
    result = env.step(0)  # N into the goal
    assert (result.terminated, result.status, result.cell) == (True, SUCCESS, (1, 5))
    with pytest.raises(RuntimeError):
        env.step(0)

    env.reset(start=(1, 3))
    assert env.step(1).status == FAIL  # E into the hazard
    assert (env.stats.episodes, env.stats.successes, env.stats.failures, env.stats.steps) == (
        2,
        1,
        1,
        7,
    )


def test_truncation_at_horizon(hand_spec_deterministic):
    spec = MapSpec(
        hand_spec_deterministic.layout, noise=hand_spec_deterministic.noise, horizon_override=3
    )
    env = RoverEnv(spec, seed=0)
    env.reset()
    results = [env.step(0) for _ in range(3)]  # bump into the north wall three times
    assert [r.truncated for r in results] == [False, False, True]
    assert not results[-1].terminated and env.stats.timeouts == 1
    with pytest.raises(RuntimeError):
        env.step(0)


def test_invalid_inputs(hand_spec):
    env = RoverEnv(hand_spec, seed=0)
    with pytest.raises(ValueError):
        env.reset(start=(0, 0))  # rock
    with pytest.raises(ValueError):
        env.reset(start=(1, 5))  # goal is not a decision cell
    env.reset()
    with pytest.raises(ValueError):
        env.step(8)


def test_sample_interface_hides_the_model(hand_spec):
    env = RoverEnv(hand_spec, seed=0)
    learner = env.sample_interface()
    for hidden in ("model", "spec", "noise", "costs", "c_fail"):
        assert not hasattr(learner, hidden)
    assert learner.layout == hand_spec.layout and learner.horizon == hand_spec.horizon
    learner.reset()
    learner.step(1)
    snapshot = learner.stats
    snapshot.steps = 999
    assert learner.stats.steps == 1  # stats are returned as a copy


def test_seeded_sampler_is_reproducible(hand_spec):
    def trajectory(seed):
        env = RoverEnv(hand_spec, seed=seed)
        env.reset()
        out = []
        for _ in range(20):
            result = env.step(1)
            out.append(result)
            if result.terminated:
                env.reset()
        return out

    assert trajectory(5) == trajectory(5)


def test_model_interface_returns_the_exact_model(hand_spec):
    env = RoverEnv(hand_spec, seed=0)
    mdp = env.model()
    assert mdp is env.model() and mdp.n_states == 8 and mdp.spec == hand_spec
    assert not hasattr(env.sample_interface(), "model")
