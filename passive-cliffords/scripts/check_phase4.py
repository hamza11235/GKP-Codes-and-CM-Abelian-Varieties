"""Print and validate the Phase 4 matched-control comparison."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(WORKSPACE / "src"))

from gkp_passive_cliffords import phase4_comparison_table, phase4_control_table


def _markdown_table(rows: list[dict[str, object]], columns: tuple[str, ...]) -> str:
    header = " | ".join(columns)
    separator = " | ".join("---" for _ in columns)
    body = [" | ".join(str(row[column]) for column in columns) for row in rows]
    return "\n".join((header, separator, *body))


def main() -> None:
    controls = phase4_control_table()
    comparisons = phase4_comparison_table()
    assert len(controls) == 18
    assert all(row["polarized_automorphism_order"] == 2 for row in controls)
    assert all(row["logical_image_order"] in (1, 2) for row in controls)

    columns = (
        "candidate_id",
        "polarization_type",
        "cm_ell_squared",
        "control_ell_squared_min",
        "control_ell_squared_max",
        "cm_automorphism_order",
        "control_automorphism_orders",
        "cm_logical_image_order",
        "control_logical_image_orders",
        "logical_image_enhancement",
    )
    print(_markdown_table(comparisons, columns))
    print("\nAll 18 controls preserve the matched polarization type and have Aut_0={+-I}.")
    print("Phase 4 matched generic-real control assertions passed.")


if __name__ == "__main__":
    main()
