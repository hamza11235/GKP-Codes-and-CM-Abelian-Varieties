"""Validate the generated Phase-5 population ledger."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(WORKSPACE / "src"))

from gkp_passive_cliffords import load_phase5_population_ledger


EXPECTED_COUNTS = {
    (1, 3): 876,
    (1, 5): 915,
    (1, 1, 2): 1051,
    (1, 1, 3): 1070,
    (1, 2, 2): 253,
}

EXPECTED_BEST = {
    (1, 3): "4/sqrt(24)",
    (1, 5): "4/sqrt(55)",
    (1, 1, 2): "2/sqrt(4)",
    (1, 1, 3): "2/sqrt(3)",
    (1, 2, 2): "2/sqrt(4)",
}


def main() -> None:
    rows, summaries = load_phase5_population_ledger(PROJECT / "data")
    counts = Counter(tuple(row["polarization_type"]) for row in rows)
    if counts != Counter(EXPECTED_COUNTS):
        raise AssertionError(f"population counts changed: {counts}")
    if len(rows) != 4165:
        raise AssertionError(f"expected 4165 records, found {len(rows)}")
    if not all(row["pairing_verified"] for row in rows):
        raise AssertionError("a logical action failed pairing verification")
    if not all(
        row["polarized_automorphism_order"]
        == row["logical_image_order"] * row["action_kernel_order"]
        for row in rows
    ):
        raise AssertionError("an action failed the image-kernel identity")
    lookup = {tuple(row["polarization_type"]): row for row in summaries}
    for polarization_type, expected in EXPECTED_BEST.items():
        if lookup[polarization_type]["best_ell_squared_exact"] != expected:
            raise AssertionError(
                f"type {polarization_type} best value changed: "
                f"{lookup[polarization_type]['best_ell_squared_exact']}"
            )

    print("type | candidates | enhanced | enhanced fraction | max image | Pareto")
    print("--- | ---: | ---: | ---: | ---: | ---:")
    for row in summaries:
        print(
            f"{tuple(row['polarization_type'])} | {row['candidate_count']} | "
            f"{row['extra_passive_symmetry_count']} | "
            f"{row['extra_passive_symmetry_fraction']:.6f} | "
            f"{row['maximum_logical_image_order']} | {row['pareto_candidate_count']}"
        )
    print("\nPhase 5 population assertions passed.")


if __name__ == "__main__":
    main()
