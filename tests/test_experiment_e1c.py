import pytest
from experiments import e1c

from riskladder.harness.runner import AGENTS


def test_the_shaping_configurations_build_their_learners_and_differ_only_in_the_potential():
    for scale, route in e1c.CONFIGS.values():
        settings = {} if scale is None else {"shaping": scale, "shaping_route": route}
        AGENTS["q"](**settings)
    assert {r for _, r in e1c.CONFIGS.values() if _ is not None} == {"free", "safe"}


@pytest.fixture(scope="module")
def quick():
    return e1c.run(quick=True)


@pytest.mark.slow
def test_every_configuration_is_summarised_over_the_same_runs(quick):
    assert set(quick["configs"]) == set(e1c.CONFIGS) == set(quick["early_failures"])
    assert {c["runs"] for c in quick["configs"].values()} == {
        quick["maps"] * quick["seeds_per_map"]
    }


@pytest.mark.slow
def test_main_writes_the_table_and_the_figure(tmp_path):
    e1c.main(["--quick", "--out", str(tmp_path)])
    for name in ("e1c.json", "e1c.md", "figures/e1c_potential.png"):
        assert (tmp_path / name).stat().st_size > 0
    assert e1c.default_out(True) == e1c.RESULTS / "smoke"
