from __future__ import annotations

import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(WORKSPACE / "src"))

from gkp_passive_cliffords import phase2_benchmark_table


def main() -> None:
    columns = (
        "model",
        "level",
        "polarization_type",
        "kernel_order",
        "polarized_automorphism_order",
        "logical_image_order",
        "action_kernel_order",
    )
    rows = phase2_benchmark_table()
    print(" | ".join(columns))
    print(" | ".join("---" for _ in columns))
    for row in rows:
        print(" | ".join(str(row[column]) for column in columns))

    lookup = {(row["model"], row["level"]): row for row in rows}
    expected = {
        ("D4 principally polarized abelian surface", 2): (48, 24, 2),
        ("D4 principally polarized abelian surface", 3): (48, 48, 1),
        ("Jacobian of the Klein quartic", 2): (336, 168, 2),
        ("Jacobian of the Klein quartic", 3): (336, 336, 1),
    }
    for key, answer in expected.items():
        row = lookup[key]
        observed = (
            row["polarized_automorphism_order"],
            row["logical_image_order"],
            row["action_kernel_order"],
        )
        if observed != answer:
            raise AssertionError(f"benchmark {key} returned {observed}, expected {answer}")
    if not all(row["pairing_verified"] for row in rows):
        raise AssertionError("a benchmark action failed pairing preservation")
    print("\nPhase 2 benchmark assertions passed.")


if __name__ == "__main__":
    main()
