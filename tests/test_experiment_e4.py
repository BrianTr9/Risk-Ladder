import json

import pytest
from experiments import e4

from riskladder.harness.runner import AGENTS

slow = pytest.mark.slow  # experiments: run in CI and nightly, skipped by default


@pytest.fixture(scope="module")
def quick(tmp_path_factory):
    return e4.run_all(quick=True, raw=tmp_path_factory.mktemp("raw"))


def test_steps_to_match_is_the_first_checkpoint_at_or_below_the_target():
    curves = {"a": [9.0, 4.0, 1.0, 0.5], "b": [9.0, 8.0, 7.0, 6.0]}
    out = e4.steps_to_match(curves, [0, 10, 20, 30], [1.0, 1.0], ["a", "b"])
    assert out == [20, None]


def test_the_match_table_reports_the_median_only_when_half_the_maps_matched():
    curve = {"q": {"steps": [0, 10, 20], "per_map_regret": {"a": [5, 1, 0], "b": [5, 5, 5]}}}
    general = {"maps": ["a", "b"], "final_regret_per_map": [1.0, 1.0]}
    table = e4.match_table(curve, general)
    assert table["per_map"] == {"a": 10, "b": None}
    assert table["matched_maps"] == 1 and table["median_steps"] == 10.0 and table["budget"] == 20
    curve["q"]["per_map_regret"]["c"] = [5, 5, 5]
    general = {"maps": ["a", "b", "c"], "final_regret_per_map": [1.0, 1.0, 1.0]}
    table = e4.match_table(curve, general)
    assert table["matched_maps"] == 1 and table["median_steps"] is None  # fewer than half


def test_missing_settings_fall_back_to_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(e4, "CONFIG", tmp_path / "none.json")
    assert e4.load_config() == {"q": {}, "sarsa": {}, "linear": {}}
    (tmp_path / "set.json").write_text(json.dumps({"q": {"alpha": 0.2}}))
    monkeypatch.setattr(e4, "CONFIG", tmp_path / "set.json")
    assert e4.load_config()["q"] == {"alpha": 0.2}


def test_the_committed_settings_only_name_known_learners_and_valid_values():
    config = json.loads(e4.CONFIG.read_text())
    for agent in e4.AGENTS:
        AGENTS[agent](**config[agent])  # raises on an invalid setting
    assert config["tuned_on"] == "field/train"


def test_held_out_maps_are_not_the_training_maps():
    assert e4.TRAIN_MAPS and e4.TEST_MAPS
    assert not {p.name for p in e4.TRAIN_MAPS} & {p.name for p in e4.TEST_MAPS}


@slow
def test_every_agent_gets_a_summary_with_the_same_checkpoint_grid(quick):
    grids = {tuple(quick["curves"][a]["steps"]) for a in e4.AGENTS}
    assert len(grids) == 1 and next(iter(grids))[0] == 0
    for agent in e4.AGENTS:
        s = quick["curves"][agent]
        assert s["runs"] == 3 * quick["seeds_per_map"]
        assert len(s["regret"]["mean"]) == len(s["steps"])
        assert all(r >= -1e-9 for r in s["regret"]["mean"])  # regret against the optimum
        assert 0.0 <= s["learned_fraction"] <= 1.0


@slow
def test_training_failures_only_grow_and_start_at_zero(quick):
    for agent in e4.AGENTS:
        failures = quick["curves"][agent]["train_failures"]["mean"]
        assert failures[0] == 0 and failures == sorted(failures)


@slow
def test_generalisation_is_scored_on_every_held_out_map(quick):
    g = quick["generalisation"]
    assert len(g["maps"]) == 3 == len(g["final_regret_per_map"]) == len(g["final_p_fail_per_map"])
    assert len(g["regret_mean"]) == len(g["steps"]) and g["steps"][0] == 0
    assert all(0.0 <= p <= 1.0 for p in g["final_p_fail_per_map"])


@slow
def test_the_raw_records_are_written_one_line_per_run(quick, tmp_path):
    raw = tmp_path / "raw"
    e4.curves(quick["config"], e4.TEST_MAPS[:1], 1, 3000, raw / "curves.jsonl")
    lines = (raw / "curves.jsonl").read_text().splitlines()
    assert len(lines) == len(e4.AGENTS) and json.loads(lines[0])["agent"] == "q"


@slow
def test_main_writes_tables_and_figures_and_tuning_never_touches_the_committed_settings(tmp_path):
    before = e4.CONFIG.read_text()
    assert e4.default_out(True) == e4.RESULTS / "smoke" and e4.default_out(False) == e4.RESULTS
    e4.main(["--quick", "--tune", "--out", str(tmp_path)])
    assert (tmp_path / "e4_config.json").exists() and e4.CONFIG.read_text() == before
    e4.main(["--quick", "--out", str(tmp_path)])
    for name in ("e4.json", "e4.md", "figures/e4_curves.png", "figures/e4_generalisation.png"):
        assert (tmp_path / name).stat().st_size > 0
