"""Matplotlib plotting shared by experiment figures and the demo (needs the `exp` extra)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from riskladder.env.dynamics import ACTIONS, DELTA
from riskladder.env.layout import GOAL, HAZARD, ROCK, ROUGH, START, Layout

if TYPE_CHECKING:
    from matplotlib.axes import Axes

TILE_COLOURS = {
    ".": "#f4f1ea",
    START: "#f4f1ea",
    ROUGH: "#d9c7a3",
    ROCK: "#4a4a4a",
    HAZARD: "#d1495b",
    GOAL: "#3c9d5d",
}


def plot_map(
    layout: Layout,
    values: np.ndarray | None = None,
    policy: np.ndarray | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    value_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Axes:
    """Draw tiles, an optional per-cell value heatmap and optional policy arrows.

    `values` is a (rows, cols) float array (NaN where unset); `policy` is a (rows, cols) array
    of action indices (negative where unset). Dash arrows are drawn thicker than move arrows.
    Pass the same `vmin` and `vmax` to several panels so that one colour means one value.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    if ax is None:
        _, ax = plt.subplots(figsize=(0.25 * layout.cols + 1, 0.25 * layout.rows + 1))
    rgb = np.array([[to_rgb(TILE_COLOURS[ch]) for ch in line] for line in layout.grid])
    ax.imshow(rgb, interpolation="nearest")

    if values is not None:
        image = ax.imshow(
            np.ma.masked_invalid(values),
            cmap=cmap,
            alpha=0.85,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        colourbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        if value_label:
            colourbar.set_label(value_label)

    if policy is not None:
        for cells, width in ((1, 0.004), (2, 0.008)):
            rr, cc, uu, vv = [], [], [], []
            for r, c in zip(*np.nonzero(policy >= 0), strict=True):
                act = ACTIONS[int(policy[r, c])]
                if act.cells == cells:
                    dr, dc = DELTA[act.direction]
                    rr.append(r)
                    cc.append(c)
                    uu.append(dc)
                    vv.append(dr)  # data coordinates: rows grow downward
            if rr:
                ax.quiver(
                    cc,
                    rr,
                    uu,
                    vv,
                    angles="xy",
                    scale_units="xy",
                    scale=1.4,
                    width=width,
                    pivot="middle",
                )

    r, c = layout.start
    ax.scatter([c], [r], marker="o", s=60, facecolors="none", edgecolors="#2b59c3", linewidths=2)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax
