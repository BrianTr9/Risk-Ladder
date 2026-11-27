import numpy as np
import pytest

from riskladder.env import build_model, render


def test_ascii_render_with_agent_and_policy(hand_spec):
    layout = hand_spec.layout
    policy = np.full((layout.rows, layout.cols), -1)
    policy[2, 1:6] = 1  # move_E along the lower row
    policy[2, 5] = 4  # dash_N into the goal
    text = render(layout, agent=(1, 2), policy=policy)
    assert text.split("\n") == ["#######", "#S@.XG#", "#→→→→⇑#", "#######"]


def test_plot_map_smoke(hand_spec):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from riskladder.viz import plot_map

    mdp = build_model(hand_spec)
    values = mdp.to_grid(np.arange(mdp.n_states, dtype=float))
    policy = mdp.to_grid(np.ones(mdp.n_states, dtype=int), fill=-1)
    ax = plot_map(hand_spec.layout, values=values, policy=policy, title="hand", value_label="value")
    assert ax.get_title() == "hand"
    matplotlib.pyplot.close(ax.figure)
