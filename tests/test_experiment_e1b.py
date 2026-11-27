import json

import pytest
from experiments import e1, e1b

from riskladder.harness.runner import AGENTS

slow = pytest.mark.slow  # experiments: run in CI and nightly, skipped by default

TUNED = {"q": {"alpha": 0.3}, "sarsa": {"alpha": 0.2}, "shaping": 1.5}


@pytest.fixture(scope="module")
def quick(tmp_path_factory):
    return e1b.run_all(quick=True, raw=tmp_path_factory.mktemp("raw"))


def test_the_shield_and_shaping_are_applied_to_sarsa_as_independent_switches():
    named = e1b.configs(TUNED)
    assert named["SARSA"][1] == {"alpha": 0.2}
    assert named["SARSA + shaping"][1] == {"alpha": 0.2, "shaping": 1.5}
    assert named["SARSA + shield"][1] == {"alpha": 0.2, "shield": e1.SHIELD}
    assert named["SARSA + shield + shaping"][1]["shaping"] == 1.5
    assert named["Q"][1] == {"alpha": 0.3} and named["Q + shield"][1]["shield"] == e1.SHIELD
    for agent, settings in named.values():
        AGENTS[agent](**settings)


def test_the_committed_settings_cover_every_family_and_use_values_from_the_grids():
    tuned = json.loads(e1b.CONFIG.read_text())
    for family in e1b.FAMILIES:
        assert tuned[family]["q"]["alpha"] in e1b.ALPHAS
        assert tuned[family]["sarsa"]["alpha"] in e1b.ALPHAS
        assert tuned[family]["shaping"] in e1b.SHAPINGS


def test_tuning_maps_are_not_evaluation_maps():
    for family, spec in e1b.FAMILIES.items():
        _, seeds, _ = spec["tune"]
        used = {int(p.stem.rsplit("-s", 1)[1]) for p in spec["maps"]}
        assert not used & set(seeds), family


@slow
def test_both_families_and_the_sweep_report_every_configuration(quick):
    assert set(quick["families"]) == {"dilemma", "ridge"}
    for data in quick["families"].values():
        assert set(data["configs"]) == set(e1b.configs(TUNED))
    for configs in quick["slip_sweep"]["slips"].values():
        assert {"Q", "SARSA", "SARSA + shield", "MB-struct + CMDP"} == set(configs)


@slow
def test_main_writes_outputs_and_tuning_never_touches_the_committed_settings(tmp_path):
    before = e1b.CONFIG.read_text()
    e1b.main(["--quick", "--tune", "--out", str(tmp_path)])
    assert (tmp_path / "e1b_config.json").exists() and e1b.CONFIG.read_text() == before
    e1b.main(["--quick", "--out", str(tmp_path)])
    for name in ("e1b.json", "e1b.md", "figures/e1b_ablation.png"):
        assert (tmp_path / name).stat().st_size > 0
