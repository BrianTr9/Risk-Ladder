"""Draw the demo's Plan tab for the first example map as a sharp image for the README.

    uv run python scripts/render_demo_image.py [--out docs/img/demo.png]

The image is the demo's own output (the same function the Space calls), so it cannot drift from
it. A screenshot of the live page is limited to the size of the browser window and comes out soft.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from app import demo  # noqa: E402


def render(out: Path, dpi: int = 200) -> str:
    """Write the figure to `out` and return the table the demo shows with it."""
    text, slip, _ = demo.example_view(next(iter(demo.EXAMPLES)))
    figure, table = demo.plan_view(text, slip, 0.01, 50.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=dpi)
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "img" / "demo.png")
    args = parser.parse_args(argv)
    print(render(args.out))


if __name__ == "__main__":
    main()
