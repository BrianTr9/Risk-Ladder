"""Write the fixed, seeded map sets used by every experiment.

    uv run python -m experiments.make_maps

- dilemma/    10 dilemma maps (a corridor beside a hazard strip versus a long detour)
- ridge/      12 ridge maps for the risk experiments (E2, E3): hazard strips of different
              lengths, each with its own alcove, at a slip probability of 0.01 (env_spec section 9)
- field/train 20 field maps for training generalising agents (E4)
- field/test  20 held-out field maps, never used for training or tuning (E4)

Seeds are disjoint between sets, so a map can never appear in both train and test.
"""

from __future__ import annotations

from pathlib import Path

from riskladder.env import Noise, generate, save_map

ROOT = Path(__file__).parent / "maps"
# folder -> (template, seeds, generator arguments)
SETS = {
    "dilemma": ("dilemma", range(0, 10), {"rows": 30}),
    "ridge": ("ridge", range(3000, 3012), {"rows": 6, "cols": 30, "noise": Noise(p_slip=0.01)}),
    "field/train": ("field", range(1000, 1020), {"rows": 30}),
    "field/test": ("field", range(2000, 2020), {"rows": 30}),
}


def main() -> None:
    for folder, (template, seeds, arguments) in SETS.items():
        out = ROOT / folder
        out.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            spec = generate(template, seed=seed, **arguments)
            save_map(spec, out / f"{spec.name}.map")
        print(f"{folder}: {len(seeds)} maps -> {out}")


if __name__ == "__main__":
    main()
