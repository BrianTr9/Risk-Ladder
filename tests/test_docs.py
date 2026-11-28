"""The public documents: links resolve, claims quote the results, the write-up keeps its style."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
README = (ROOT / "README.md").read_text()
WRITEUP = (ROOT / "docs" / "writeup.md").read_text()
DECISIONS = (ROOT / "docs" / "decisions.md").read_text()


def anchors(text: str) -> set[str]:
    """GitHub's anchor for each heading: lower case, punctuation dropped, spaces to hyphens."""
    out = set()
    for heading in re.findall(r"^#+ (.+)$", text, re.M):
        slug = re.sub(r"[^\w\s-]", "", heading.lower()).strip().replace(" ", "-")
        out.add(slug)
    return out


def test_every_local_link_and_image_in_the_readme_exists():
    for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", README):
        if not target.startswith(("http", "mailto")):
            assert (ROOT / target).exists(), target


def test_every_anchor_link_in_the_readme_points_at_a_heading():
    known = anchors(README)
    for anchor in re.findall(r"\]\(#([^)]+)\)", README):
        assert anchor in known, anchor


def test_the_headline_numbers_in_the_claims_come_from_the_results():
    e0 = min(
        json.loads((ROOT / "results/e0.json").read_text())["rows"], key=lambda r: r["p_fail_exact"]
    )
    e1 = json.loads((ROOT / "results/e1.json").read_text())["configs"]
    e3 = json.loads((ROOT / "results/e3.json").read_text())["spread"]["summary"]["0.005"]
    e4 = json.loads((ROOT / "results/e4.json").read_text())["generalisation"]["final"]["regret"]
    claims = README.split("## Claims")[1].split("## Quickstart")[0]
    assert f"{e0['rollouts_needed']:,}" in claims and f"{e0['speedup']:,.0f}" in claims
    assert f"{e1['Q']['train_failures']['mean']:,.0f}" in claims
    assert f"{e1['MB-struct']['train_failures']['mean']:,.0f} rovers" in claims
    assert f"{e3['penalty_violates_bound']} of {e3['maps']}" in claims
    assert f"{100 * e3['penalty_max_p_fail']:.2f}%" in claims
    assert f"{100 * e4['median']:.1f}%" in claims


def test_the_writeup_follows_the_style_rules():
    assert "—" not in WRITEUP and "—" not in README  # no em dashes
    assert not re.search(r"\byet\b", WRITEUP, re.I)
    assert "**" not in WRITEUP  # no bold in running text
    banned = ["delve", "pivotal", "crucial", "furthermore", "moreover", "notably", "landscape"]
    assert not [w for w in banned if re.search(rf"\b{w}\b", WRITEUP, re.I)]
    assert 1100 <= len(WRITEUP.split()) <= 1800


def test_the_writeup_numbers_match_the_results():
    e5 = json.loads((ROOT / "results/e5.json").read_text())["curve"]
    near = min(e5, key=lambda c: abs(c["p_slip"] - 0.15))
    assert f"{100 * near['p_fail']['mean']:.0f}%" in WRITEUP
    e1 = json.loads((ROOT / "results/e1.json").read_text())["configs"]
    for name in ("Q", "SARSA", "MB-count", "MB-struct"):
        assert f"{e1[name]['train_failures']['mean']:,.0f}" in WRITEUP, name


def test_the_decision_log_follows_the_same_style_and_its_numbers_match_the_results():
    assert "—" not in DECISIONS and "**" not in DECISIONS
    assert not re.search(r"\byet\b", DECISIONS, re.I)
    e1 = json.loads((ROOT / "results/e1.json").read_text())["configs"]
    assert f"{e1['MB-count']['train_failures']['mean']:,.0f} against" in DECISIONS
    e1c = json.loads((ROOT / "results/e1c.json").read_text())["configs"]
    assert (
        f"{e1c['route clear of hazards, scale 2']['train_failures']['mean']:,.0f} rovers"
        in DECISIONS
    )
