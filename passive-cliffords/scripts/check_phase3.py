from __future__ import annotations

import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(WORKSPACE / "src"))

from gkp_passive_cliffords import phase3_cm_action_table


EXPECTED = {
    "g2_type_13_delta_24": (24, 24, 1, 24),
    "g2_type_15_reconstructed": (24, 24, 1, 120),
    "g3_type_112_reconstructed": (12, 3, 4, 6),
    "g3_type_112_gaussian_bounded": (384, 6, 64, 6),
    "g3_type_113_eisenstein_bounded": (1296, 24, 54, 24),
    "g3_type_122_gaussian_bounded": (384, 48, 8, 720),
}


def main() -> None:
    columns = (
        "candidate_id",
        "polarization_type",
        "polarized_automorphism_order",
        "logical_image_order",
        "action_kernel_order",
        "full_symplectic_target_order",
        "target_coverage",
        "ell_squared_exact",
    )
    rows = phase3_cm_action_table()
    print(" | ".join(columns))
    print(" | ".join("---" for _ in columns))
    for row in rows:
        print(" | ".join(str(row[column]) for column in columns))

    for row in rows:
        expected = EXPECTED[row["candidate_id"]]
        observed = (
            row["polarized_automorphism_order"],
            row["logical_image_order"],
            row["action_kernel_order"],
            row["full_symplectic_target_order"],
        )
        if observed != expected:
            raise AssertionError(
                f"candidate {row['candidate_id']} returned {observed}, expected {expected}"
            )
        if not row["pairing_verified"]:
            raise AssertionError(f"candidate {row['candidate_id']} failed pairing verification")
    print("\nPhase 3 nonuniform CM assertions passed.")


if __name__ == "__main__":
    main()
