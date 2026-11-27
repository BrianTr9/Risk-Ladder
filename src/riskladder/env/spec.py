"""Map specification: layout plus hidden parameters (noise, costs) and the `.map` file format."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from functools import cached_property
from numbers import Integral, Real
from pathlib import Path

from riskladder.env.distances import max_reachable_distance
from riskladder.env.layout import Layout, MapError

SEPARATOR = "---"
_HEADER_KEYS = {"name", "seed", "noise", "costs", "fail_multiplier", "c_fail", "horizon"}
MAX_HORIZON = 10_000_000  # longest episode a spec may request (guards evaluation loops)
MAX_MAP_CHARS = 4_000_000  # largest `.map` text `parse_map` accepts (about a 2000 x 2000 grid)


def _number(value: object, name: str) -> float:
    """A finite real number; booleans, strings, None, NaN and infinities are rejected."""
    try:
        finite = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
    except OverflowError:  # an integer too large for a float
        finite = False
    if not finite:
        raise MapError(f"{name} must be a finite number, got {str(value)[:40]!r}")
    return value  # type: ignore[return-value]


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise MapError(f"{name} must be an integer, got {str(value)[:40]!r}")
    return int(value)


@dataclass(frozen=True)
class Noise:
    """Slip model: the rover slips sideways with probability `p`, split evenly between sides."""

    p_slip: float = 0.15
    dash_slip_mult: float = 2.0

    def __post_init__(self) -> None:
        _number(self.p_slip, "p_slip")
        _number(self.dash_slip_mult, "dash_slip_mult")
        if not 0.0 <= self.p_slip <= 1.0:
            raise MapError(f"p_slip must be in [0, 1], got {self.p_slip}")
        if self.dash_slip_mult < 0.0 or self.p_slip * self.dash_slip_mult > 1.0:
            raise MapError("dash slip probability p_slip * dash_slip_mult must be in [0, 1]")

    def slip(self, cells: int) -> float:
        """Slip probability of a move (1 cell) or a dash (2 cells)."""
        return self.p_slip if cells == 1 else self.p_slip * self.dash_slip_mult


@dataclass(frozen=True)
class Costs:
    move: float = 1.0
    dash: float = 1.5
    rough_extra: float = 1.0
    collision: float = 2.0

    def __post_init__(self) -> None:
        for name in ("move", "dash", "rough_extra", "collision"):
            _number(getattr(self, name), name)
        if self.move <= 0 or self.dash <= 0:
            raise MapError("move and dash costs must be strictly positive")
        if self.rough_extra < 0 or self.collision < 0:
            raise MapError("rough_extra and collision costs must be non-negative")


@dataclass(frozen=True)
class MapSpec:
    """Everything needed to build the environment. Planners see all of it; learners see `layout`."""

    layout: Layout
    noise: Noise = field(default_factory=Noise)
    costs: Costs = field(default_factory=Costs)
    fail_multiplier: float = 5.0
    c_fail_override: float | None = None
    horizon_override: int | None = None
    name: str = ""
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layout, Layout):
            raise MapError("layout must be a Layout")
        _number(self.fail_multiplier, "fail_multiplier")
        if self.fail_multiplier <= 0:
            raise MapError("fail_multiplier must be positive")
        if self.c_fail_override is not None:
            _number(self.c_fail_override, "c_fail")
            if self.c_fail_override < 0:
                raise MapError("c_fail must be non-negative")
        if self.horizon_override is not None:
            _integer(self.horizon_override, "horizon")
            if not 1 <= self.horizon_override <= MAX_HORIZON:
                raise MapError(f"horizon must be between 1 and {MAX_HORIZON}")
        if not isinstance(self.name, str):
            raise MapError("name must be a string")
        if self.seed is not None and not -(2**63) <= _integer(self.seed, "seed") < 2**63:
            raise MapError("seed must fit in 64 bits")

    @cached_property
    def c_fail(self) -> float:
        """Cost of losing the rover: `fail_multiplier * D_max * move` unless overridden."""
        if self.c_fail_override is not None:
            return float(self.c_fail_override)
        return self.fail_multiplier * max_reachable_distance(self.layout) * self.costs.move

    @property
    def horizon(self) -> int:
        """Episode length `H`; default `10 * (rows + cols)`."""
        if self.horizon_override is not None:
            return self.horizon_override
        return 10 * (self.layout.rows + self.layout.cols)

    def with_noise(self, noise: Noise) -> MapSpec:
        """Same map under different dynamics (used for model-misspecification experiments)."""
        return replace(self, noise=noise)


def _section(header: dict, key: str, cls: type) -> object:
    """Build the `noise` or `costs` dataclass from its header object, rejecting anything else."""
    value = header.get(key, {})
    if not isinstance(value, dict):
        raise MapError(f"header key {key!r} must be an object")
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise MapError(f"unknown keys in {key!r}: {sorted(unknown)}")
    return cls(**value)


def parse_map(text: str) -> MapSpec:
    """Parse the `.map` format: a JSON header, a line `---`, then the grid.

    Any malformed input raises `MapError` (never another exception type), so callers such as an API
    can turn every failure into a clean client error.
    """
    if not isinstance(text, str):
        raise MapError("map text must be a string")
    if len(text) > MAX_MAP_CHARS:
        raise MapError(f"map text is longer than {MAX_MAP_CHARS:,} characters")
    lines = text.strip("\n").split("\n")
    try:
        cut = next(i for i, line in enumerate(lines) if line.strip() == SEPARATOR)
    except StopIteration:
        raise MapError(f"missing '{SEPARATOR}' line between header and grid") from None
    try:
        header = json.loads("\n".join(lines[:cut]) or "{}")
    except json.JSONDecodeError as error:
        raise MapError(f"invalid JSON header: {error}") from None
    except RecursionError:
        raise MapError("header is nested too deeply") from None
    if not isinstance(header, dict):
        raise MapError("header must be a JSON object")
    unknown = set(header) - _HEADER_KEYS
    if unknown:
        raise MapError(f"unknown header keys: {sorted(unknown)}")
    return MapSpec(
        layout=Layout(tuple(lines[cut + 1 :])),
        noise=_section(header, "noise", Noise),  # type: ignore[arg-type]
        costs=_section(header, "costs", Costs),  # type: ignore[arg-type]
        fail_multiplier=header.get("fail_multiplier", 5.0),
        c_fail_override=header.get("c_fail"),
        horizon_override=header.get("horizon"),
        name=header.get("name", ""),
        seed=header.get("seed"),
    )


def format_map(spec: MapSpec) -> str:
    header: dict[str, object] = {"name": spec.name, "seed": spec.seed}
    header["noise"] = asdict(spec.noise)
    header["costs"] = asdict(spec.costs)
    header["fail_multiplier"] = spec.fail_multiplier
    if spec.c_fail_override is not None:
        header["c_fail"] = spec.c_fail_override
    if spec.horizon_override is not None:
        header["horizon"] = spec.horizon_override
    return f"{json.dumps(header, indent=2)}\n{SEPARATOR}\n{spec.layout}\n"


def load_map(path: str | Path) -> MapSpec:
    return parse_map(Path(path).read_text())


def save_map(spec: MapSpec, path: str | Path) -> None:
    Path(path).write_text(format_map(spec))
