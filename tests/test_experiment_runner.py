import json
import re

import pytest
from experiments import run as runner
from experiments import summary

RESULTS = summary.RESULTS


def test_every_experiment_is_registered_with_a_smoke_mode_and_a_description(capsys):
    runner.main(["list"])
    listing = capsys.readouterr().out
    for name, (what, main, quick) in runner.EXPERIMENTS.items():
        assert name in listing and what in listing and callable(main) and isinstance(quick, list)
    assert "summary" in listing


def test_an_unknown_experiment_is_refused_by_name(capsys):
    with pytest.raises(SystemExit):
        runner.main(["e9"])
    assert "e9" in capsys.readouterr().err


def test_the_committed_summary_has_every_headline_figure_and_all_links_resolve():
    text = summary.build(RESULTS)
    for figure in ("e3_risk_spread", "e2_misspecification", "e1_pareto", "e4_generalisation"):
        assert f"figures/{figure}.png" in text
    links = re.findall(r"\(figures/([\w.]+)\)", text)
    assert links and all((RESULTS / "figures" / name).exists() for name in links)
    assert "Not generated" not in text


def test_summary_numbers_come_from_the_result_files():
    text = summary.build(RESULTS)
    e5 = json.loads((RESULTS / "e5.json").read_text())
    near = min(e5["curve"], key=lambda c: abs(c["p_slip"] - 0.15))
    assert f"{100 * near['p_fail']['mean']:.1f}%" in text
    e1 = json.loads((RESULTS / "e1.json").read_text())
    assert (
        f"MB-struct + CMDP {e1['configs']['MB-struct + CMDP']['train_failures']['mean']:,.0f}"
        in text
    )
    e4 = json.loads((RESULTS / "e4.json").read_text())
    assert f"{100 * e4['generalisation']['final']['regret']['median']:.1f}%" in text


def test_the_summary_says_which_result_is_missing(tmp_path):
    (tmp_path / "figures").mkdir()
    text = summary.build(tmp_path)
    assert text.count("Not generated") == len(summary.SECTIONS)


def test_a_single_experiment_runs_through_the_runner_into_the_folder_asked_for(tmp_path):
    folder = runner.run(["e5", "summary"], smoke=True, out=tmp_path)
    assert folder == tmp_path
    assert (tmp_path / "e5.json").exists() and (tmp_path / "SUMMARY.md").exists()


@pytest.mark.slow
def test_the_whole_smoke_run_writes_every_result_and_a_complete_summary(tmp_path):
    runner.run([*runner.EXPERIMENTS, "summary"], smoke=True, out=tmp_path)
    for name in runner.EXPERIMENTS:
        assert (tmp_path / f"{name}.json").stat().st_size > 0
    assert "Not generated" not in (tmp_path / "SUMMARY.md").read_text()
