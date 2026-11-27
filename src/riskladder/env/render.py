"""Plain-text rendering of a map, an agent position and a policy."""

from __future__ import annotations

import numpy as np

from riskladder.env.layout import Layout

# Arrows for move_N, move_E, move_S, move_W, dash_N, dash_E, dash_S, dash_W.
ARROWS = ("↑", "→", "↓", "←", "⇑", "⇒", "⇓", "⇐")
AGENT = "@"


def render(
    layout: Layout,
    agent: tuple[int, int] | None = None,
    policy: np.ndarray | None = None,
) -> str:
    """Render the grid. `policy` is a (rows, cols) array of action indices, negative where unset."""
    lines = []
    for r, line in enumerate(layout.grid):
        chars = list(line)
        if policy is not None:
            for c in range(layout.cols):
                if policy[r, c] >= 0:
                    chars[c] = ARROWS[int(policy[r, c])]
        if agent is not None and agent[0] == r:
            chars[agent[1]] = AGENT
        lines.append("".join(chars))
    return "\n".join(lines)
