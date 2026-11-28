import json

import pytest
from experiments import e1

from riskladder.harness.runner import AGENTS

slow = pytest.mark.slow  # experiments: run in CI and nightly, skipped by default


@pytest.fixture(scope="module")
def quick(tmp_path_factory):
    return e1.run_all(quick=True, raw=tmp_path_factory.mktemp("raw"))


def test_there_are_eight_configurations_and_each_builds_its_learner():
    named = e1.configs(1.0)
    assert len(named) == 8 and set(e1.MAIN) <= set(named)
    for agent, settings in named.values():
        AGENTS[agent](**settings)  # raises on an invalid setting


def test_the_ablation_switches_are_independent():
    named = e1.configs(1.5)
    assert "shield" not in named["Q"][1] and "shaping" not in named["Q"][1]
    assert named["Q + shaping"][1]["shaping"] == 1.5 and "shield" not in named["Q + shaping"][1]
    assert named["Q + shield"][1]["shield"] == e1.SHIELD and "shaping" not in named["Q + shield"][1]
    both = named["Q + shield + shaping"][1]
    assert both["shield"] == e1.SHIELD and both["shaping"] == 1.5
    assert named["MB-struct + CMDP"][1]["cmdp"] is True and "cmdp" not in named["MB-struct"][1]


def test_all_configurations_use_the_same_tuned_step_sizes_for_the_tabular_learner():
    named = e1.configs(1.0)
    tuned = json.loads(e1.e4.CONFIG.read_text())["q"]
    for label in ("Q", "Q + shaping", "Q + shield", "Q + shield + shaping"):
        assert all(named[label][1][k] == v for k, v in tuned.items())


def test_the_evaluation_maps_exclude_the_tuning_maps():
    assert len(e1.MAPS) == 10
    seeds = {int(p.stem.rsplit("-s", 1)[1]) for p in e1.MAPS}
    assert not seeds & set(e1.TUNE_SEEDS)


def test_the_committed_shaping_scale_is_one_of_the_tried_values():
    assert json.loads(e1.CONFIG.read_text())["shaping"] in e1.SHAPING_GRID
    assert e1.load_scale() in e1.SHAPING_GRID


@slow
def test_every_configuration_is_summarised_over_the_same_runs(quick):
    assert set(quick["configs"]) == set(e1.configs(1.0))
    for c in quick["configs"].values():
        assert c["runs"] == quick["maps"] * quick["seeds_per_map"]
        assert c["train_failures"]["mean"] >= 0 and 0.0 <= c["p_fail"]["mean"] <= 1.0


@slow
def test_the_structured_model_with_the_constraint_loses_far_fewer_rovers_than_plain_q(quick):
    configs = quick["configs"]
    assert (
        configs["MB-struct + CMDP"]["train_failures"]["mean"]
        < configs["Q"]["train_failures"]["mean"]
    )
    assert configs["MB-struct"]["regret"]["mean"] < configs["Q"]["regret"]["mean"]


@slow
def test_main_writes_tables_and_figures_and_never_touches_the_committed_settings(tmp_path):
    before = e1.CONFIG.read_text()
    assert e1.default_out(True) == e1.RESULTS / "smoke" and e1.default_out(False) == e1.RESULTS
    e1.main(["--quick", "--tune", "--out", str(tmp_path)])
    assert (tmp_path / "e1_config.json").exists() and e1.CONFIG.read_text() == before
    e1.main(["--quick", "--out", str(tmp_path)])
    for name in ("e1.json", "e1.md", "figures/e1_pareto.png", "figures/e1_curves.png"):
        assert (tmp_path / name).stat().st_size > 0
