"""Write the fixed, seeded map sets used by every experiment.

    uv run python -m experiments.make_maps

- dilemma/    10 dilemma maps (headline experiments E1-E3)
- field/train 20 field maps for training generalising agents (E4)
- field/test  20 held-out field maps, never used for training or tuning (E4)

Seeds are disjoint between sets, so a map can never appear in both train and test.
"""

from __future__ import annotations

from pathlib import Path

from riskladder.env import generate, save_map

ROOT = Path(__file__).parent / "maps"
SIZE = 30
SETS = {
    "dilemma": ("dilemma", range(0, 10)),
    "field/train": ("field", range(1000, 1020)),
    "field/test": ("field", range(2000, 2020)),
}


def main() -> None:
    for folder, (template, seeds) in SETS.items():
        out = ROOT / folder
        out.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            spec = generate(template, SIZE, seed=seed)
            save_map(spec, out / f"{spec.name}.map")
        print(f"{folder}: {len(seeds)} maps -> {out}")


if __name__ == "__main__":
    main()
