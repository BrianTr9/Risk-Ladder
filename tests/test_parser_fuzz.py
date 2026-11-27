"""The `.map` parser must turn every bad input into `MapError` (an API maps that to a 4xx)."""

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from riskladder.env import MapError, format_map, parse_map
from riskladder.env.spec import MAX_HORIZON, MAX_MAP_CHARS

GRID = "#S..G#\n#....#"
BASELINE = {
    "name": "t",
    "seed": 1,
    "noise": {"p_slip": 0.1, "dash_slip_mult": 2.0},
    "costs": {"move": 1.0, "dash": 1.5, "rough_extra": 1.0, "collision": 2.0},
    "fail_multiplier": 5.0,
    "horizon": 100,
}


def text_with(**changes) -> str:
    """The baseline header with exactly one field replaced (the single-invalid-input rule)."""
    header = {**BASELINE, **changes}
    return json.dumps(header) + "\n---\n" + GRID


def test_baseline_is_valid():
    spec = parse_map(text_with())
    assert spec.horizon == 100 and spec.seed == 1 and spec.noise.p_slip == 0.1


@pytest.mark.parametrize(
    "changes",
    [
        {"noise": 3},
        {"noise": None},
        {"noise": [0.1]},
        {"noise": {"speed": 1}},
        {"noise": {"p_slip": "high"}},
        {"noise": {"p_slip": None}},
        {"noise": {"p_slip": True}},
        {"noise": {"p_slip": float("nan")}},
        {"noise": {"p_slip": float("inf")}},
        {"noise": {"dash_slip_mult": float("nan")}},
        {"costs": None},
        {"costs": [1, 2]},
        {"costs": {"move": float("nan")}},
        {"costs": {"move": float("inf")}},
        {"costs": {"dash": "1.5"}},
        {"costs": {"collision": None}},
        {"fail_multiplier": "x"},
        {"fail_multiplier": None},
        {"fail_multiplier": float("nan")},
        {"fail_multiplier": 0},
        {"c_fail": "9"},
        {"c_fail": float("inf")},
        {"horizon": 2.5},
        {"horizon": 50.0},
        {"horizon": "9"},
        {"horizon": True},
        {"horizon": 0},
        {"horizon": MAX_HORIZON + 1},
        {"horizon": 10**30},
        {"seed": [1]},
        {"seed": 2**63},
        {"seed": -(2**63) - 1},
        {"fail_multiplier": 10**400},  # integers too large for a float
        {"c_fail": -(10**400)},
        {"costs": {"move": 10**400}},
        {"noise": {"p_slip": 10**400}},
        {"seed": 1.5},
        {"name": {"a": 1}},
        {"unknown_key": 1},
    ],
    ids=lambda c: json.dumps(c)[:60],
)
def test_each_invalid_header_field_is_a_map_error(changes):
    with pytest.raises(MapError):
        parse_map(text_with(**changes))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "---",
        "{not json\n---\n" + GRID,
        "[]\n---\n" + GRID,
        "null\n---\n" + GRID,
        "123\n---\n" + GRID,
        '"text"\n---\n' + GRID,
        "{}\n" + GRID,  # no separator
        "{}\n---\n",  # no grid
        "{}\n---\n#S?G#",
        "[" * 100_000 + "\n---\n" + GRID,  # nesting that would overflow the recursion limit
        "{}\n---\n" + "#" * (MAX_MAP_CHARS + 1),
    ],
    ids=lambda t: repr(t[:20]),
)
def test_structural_problems_are_map_errors(text):
    with pytest.raises(MapError):
        parse_map(text)


def test_non_string_input_is_a_map_error():
    with pytest.raises(MapError):
        parse_map(None)  # type: ignore[arg-type]


json_values = st.recursive(
    st.none() | st.booleans() | st.integers(-(10**12), 10**12) | st.floats() | st.text(max_size=6),
    lambda children: (
        st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=8), children, max_size=3)
    ),
    max_leaves=8,
)
header_keys = ["name", "seed", "noise", "costs", "fail_multiplier", "c_fail", "horizon"]
headers = st.fixed_dictionaries({}, optional={k: json_values for k in header_keys})
nested_headers = st.fixed_dictionaries(
    {},
    optional={
        "noise": st.dictionaries(st.sampled_from(["p_slip", "dash_slip_mult", "x"]), json_values),
        "costs": st.dictionaries(
            st.sampled_from(["move", "dash", "rough_extra", "collision", "x"]), json_values
        ),
    },
)


def check(text: str) -> None:
    """Accept or raise MapError; whatever is accepted must round-trip through `format_map`."""
    try:
        spec = parse_map(text)
    except MapError:
        return
    assert parse_map(format_map(spec)) == spec
    assert spec.c_fail >= 0 and spec.horizon >= 1


@given(header=headers)
def test_arbitrary_headers_never_crash_the_parser(header):
    check(json.dumps(header) + "\n---\n" + GRID)


@given(header=nested_headers)
def test_arbitrary_noise_and_cost_sections_never_crash_the_parser(header):
    check(json.dumps(header) + "\n---\n" + GRID)


@given(text=st.text(alphabet=list('.S~#XG \n-{}[]":,0123456789e'), max_size=80))
def test_arbitrary_text_never_crashes_the_parser(text):
    check(text)
    check(text + "\n---\n" + GRID)
    check("{}\n---\n" + text)
