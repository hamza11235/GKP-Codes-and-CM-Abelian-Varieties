from fractions import Fraction

from gkp_passive_cliffords import (
    PHASE3_CM_CANDIDATES,
    elementary_prime_symplectic_order,
    phase3_cm_action_table,
)


EXPECTED = {
    "g2_type_13_delta_24": (24, 24, 1, 24),
    "g2_type_15_reconstructed": (24, 24, 1, 120),
    "g3_type_112_reconstructed": (12, 3, 4, 6),
    "g3_type_112_gaussian_bounded": (384, 6, 64, 6),
    "g3_type_113_eisenstein_bounded": (1296, 24, 54, 24),
    "g3_type_122_gaussian_bounded": (384, 48, 8, 720),
}


def test_elementary_prime_target_orders() -> None:
    assert elementary_prime_symplectic_order((1, 3)) == 24
    assert elementary_prime_symplectic_order((1, 5)) == 120
    assert elementary_prime_symplectic_order((1, 1, 2)) == 6
    assert elementary_prime_symplectic_order((1, 2, 2)) == 720


def test_phase3_nonuniform_cm_actions() -> None:
    rows = phase3_cm_action_table()
    assert len(rows) == len(PHASE3_CM_CANDIDATES) == 6
    for row in rows:
        expected = EXPECTED[row["candidate_id"]]
        observed = (
            row["polarized_automorphism_order"],
            row["logical_image_order"],
            row["action_kernel_order"],
            row["full_symplectic_target_order"],
        )
        assert observed == expected
        assert row["polarized_automorphism_order"] == (
            row["logical_image_order"] * row["action_kernel_order"]
        )
        assert row["pairing_verified"] is True


def test_phase3_target_coverage() -> None:
    rows = {row["candidate_id"]: row for row in phase3_cm_action_table()}
    assert rows["g2_type_13_delta_24"]["target_coverage"] == Fraction(1)
    assert rows["g2_type_15_reconstructed"]["target_coverage"] == Fraction(1, 5)
    assert rows["g3_type_112_reconstructed"]["target_coverage"] == Fraction(1, 2)
    assert rows["g3_type_112_gaussian_bounded"]["target_coverage"] == Fraction(1)
    assert rows["g3_type_113_eisenstein_bounded"]["target_coverage"] == Fraction(1)
    assert rows["g3_type_122_gaussian_bounded"]["target_coverage"] == Fraction(1, 15)
