from pathlib import Path

import pytest

from riskladder.env import MapError, generate, load_map, steps_to_goal
from riskladder.env.generate import has_safe_route

MAPS = Path(__file__).parents[1] / "experiments" / "maps"


@pytest.mark.parametrize("template", ["dilemma", "field"])
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
    assert len(files) == 50
    for path in files:
        spec = load_map(path)
        template = spec.name.split("-")[0]
        regenerated = generate(template, spec.layout.rows, spec.layout.cols, seed=spec.seed)
        assert spec == regenerated, path.name


def test_train_and_test_maps_are_disjoint():
    train = {load_map(p).layout for p in (MAPS / "field" / "train").glob("*.map")}
    test = {load_map(p).layout for p in (MAPS / "field" / "test").glob("*.map")}
    assert len(train) == 20 and len(test) == 20 and not train & test


def test_map_error_is_a_value_error():
    assert issubclass(MapError, ValueError)
