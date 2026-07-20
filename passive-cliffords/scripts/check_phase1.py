from __future__ import annotations

import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(WORKSPACE / "src"))

from gkp_passive_cliffords import phase1_benchmark_table


def main() -> None:
    columns = (
        "benchmark",
        "level",
        "kernel_order",
        "polarized_automorphism_order",
        "logical_image_order",
        "action_kernel_order",
    )
    print(" | ".join(columns))
    print(" | ".join("---" for _ in columns))
    rows = phase1_benchmark_table()
    for row in rows:
        print(" | ".join(str(row[column]) for column in columns))

    lookup = {(row["benchmark"], row["level"]): row for row in rows}
    expected = {
        ("generic rectangular", 2): (2, 1, 2),
        ("generic rectangular", 3): (2, 2, 1),
        ("square CM", 2): (4, 2, 2),
        ("square CM", 3): (4, 4, 1),
        ("hexagonal CM", 2): (6, 3, 2),
        ("hexagonal CM", 3): (6, 6, 1),
    }
    for key, orders in expected.items():
        row = lookup[key]
        observed = (
            row["polarized_automorphism_order"],
            row["logical_image_order"],
            row["action_kernel_order"],
        )
        if observed != orders:
            raise AssertionError(f"benchmark {key} returned {observed}, expected {orders}")
    if not all(row["pairing_verified"] for row in rows):
        raise AssertionError("a benchmark action failed pairing preservation")
    print("\nPhase 1 benchmark assertions passed.")


if __name__ == "__main__":
    main()
