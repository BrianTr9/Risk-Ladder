from pathlib import Path

import pytest

from riskladder.env import MapError, generate, load_map, steps_to_goal
from riskladder.env.generate import (
    MIN_STRIPS,
    TEMPLATES,
    count_strips,
    has_safe_route,
    is_dilemma,
)

MAPS = Path(__file__).parents[1] / "experiments" / "maps"


@pytest.mark.parametrize("template", ["dilemma", "field", "ridge"])
def test_generation_is_deterministic(template):
    assert generate(template, 20, seed=11) == generate(template, 20, seed=11)
    assert generate(template, 20, seed=11).layout != generate(template, 20, seed=12).layout


@pytest.mark.parametrize("template", ["dilemma", "field"])
@pytest.mark.parametrize("size", [10, 30])
def test_many_seeds_are_valid(template, size):
    for seed in range(25):
        generate(template, size, seed=seed)  # Layout validation raises on an invalid map


@pytest.mark.parametrize("size", [10, 30])
def test_dilemma_offers_a_real_choice(size):
    """Short corridor beside the hazards versus a hazard-free detour that is clearly longer."""
    for seed in range(25):
        layout = generate("dilemma", size, seed=seed).layout
        start = layout.start
        short = steps_to_goal(layout)[start]
        safe = steps_to_goal(layout, avoid_hazard_adjacent=True)[start]
        assert has_safe_route(layout)
        assert safe >= 1.4 * short, f"seed {seed}: detour {safe} vs corridor {short}"


def test_unknown_template():
    with pytest.raises(ValueError):
        generate("maze", 10)
    with pytest.raises(ValueError):
        generate("dilemma", 8)


def test_committed_map_sets_match_the_generator():
    """experiments/maps must be reproducible from experiments/make_maps.py."""
    files = sorted(MAPS.rglob("*.map"))
    assert len(files) == 62
    for path in files:
        spec = load_map(path)
        template = spec.name.split("-")[0]
        regenerated = generate(
            template,
            spec.layout.rows,
            spec.layout.cols,
            seed=spec.seed,
            noise=spec.noise,
            costs=spec.costs,
        )
        assert spec == regenerated, path.name


def test_train_and_test_maps_are_disjoint():
    train = {load_map(p).layout for p in (MAPS / "field" / "train").glob("*.map")}
    test = {load_map(p).layout for p in (MAPS / "field" / "test").glob("*.map")}
    assert len(train) == 20 and len(test) == 20 and not train & test


def test_map_error_is_a_value_error():
    assert issubclass(MapError, ValueError)


@pytest.mark.parametrize("template", ["dilemma", "field"])
@pytest.mark.parametrize(("rows", "cols"), [(10, 20), (15, 30), (40, 12), (25, 25)])
def test_non_square_maps_are_valid(template, rows, cols):
    for seed in range(5):
        layout = generate(template, rows, cols, seed=seed).layout
        assert (layout.rows, layout.cols) == (rows, cols)
        if template == "dilemma":
            assert is_dilemma(layout)


def test_dilemma_that_cannot_be_built_is_rejected_loudly():
    """A very wide, short map has no room for a long detour: fail, do not return a weak map."""
    with pytest.raises(MapError, match="squarer"):
        generate("dilemma", 10, 45, seed=0)


def test_generation_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setitem(
        TEMPLATES, "broken", lambda rows, cols, rng: [["#"] * cols for _ in range(rows)]
    )
    with pytest.raises(MapError, match="could not generate"):
        generate("broken", 10)


def test_ridge_maps_have_a_staircase_of_independent_choices():
    for seed in range(30):
        layout = generate("ridge", 6, 30, seed=seed).layout
        assert layout.rows == 6 and layout.cols == 30
        assert count_strips(layout) >= MIN_STRIPS
        # every hazard strip has an alcove above the lane, and rock separates the alcoves
        lane, hazards = layout.rows - 3, layout.grid[layout.rows - 2]
        for c, ch in enumerate(hazards):
            if ch == "X":
                assert layout.grid[lane - 1][c] == "."
        assert layout.grid[lane][1] == "S" and layout.grid[lane][layout.cols - 2] == "G"


@pytest.mark.parametrize(("rows", "cols"), [(4, 30), (6, 19), (3, 10)])
def test_ridge_rejects_too_small_maps(rows, cols):
    with pytest.raises(ValueError, match="at least 5 rows"):
        generate("ridge", rows, cols, seed=0)


def test_taller_ridges_have_taller_alcoves():
    layout = generate("ridge", 9, 40, seed=1).layout
    heights = {
        sum(1 for r in range(layout.rows - 3) if layout.grid[r][c] == ".")
        for c in range(layout.cols)
        if layout.grid[layout.rows - 2][c] == "X"
    }
    assert max(heights) > 1  # alcove heights vary up to rows - 4
