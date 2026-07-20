from fractions import Fraction

from gkp_systole import Polarization

from gkp_passive_cliffords import (
    GENERIC_RECTANGULAR,
    HEXAGONAL,
    PHASE3_CM_CANDIDATES,
    SQUARE,
    NumericalPolarizedAutomorphismProblem,
    enumerate_numerical_polarized_automorphisms,
    phase4_comparison_table,
    phase4_control_table,
)


def test_numerical_automorphism_enumerator_reproduces_exact_one_mode_orders() -> None:
    polarization = Polarization(((0, 2), (-2, 0)))
    expected = {
        GENERIC_RECTANGULAR.name: 2,
        SQUARE.name: 4,
        HEXAGONAL.name: 6,
    }
    for benchmark in (GENERIC_RECTANGULAR, SQUARE, HEXAGONAL):
        group = enumerate_numerical_polarized_automorphisms(
            NumericalPolarizedAutomorphismProblem(polarization, benchmark.metric)
        )
        assert group.order == expected[benchmark.name]
        assert group.maximum_metric_residual == 0.0


def test_phase4_controls_are_fixed_type_and_have_generic_minimal_symmetry() -> None:
    rows = phase4_control_table()
    candidate_types = {
        candidate.candidate_id: candidate.polarization.type
        for candidate in PHASE3_CM_CANDIDATES
    }
    assert len(rows) == 3 * len(PHASE3_CM_CANDIDATES) == 18
    for row in rows:
        assert row["polarization_type"] == candidate_types[row["candidate_id"]]
        assert row["polarized_automorphism_order"] == 2
        assert row["logical_image_order"] in (1, 2)
        assert row["action_kernel_order"] * row["logical_image_order"] == 2
        assert row["maximum_metric_residual"] < 1e-10
        assert 0.0 < row["relative_metric_displacement"] < 0.2
        assert "not individually certified" in row["control_status"]


def test_phase4_matched_cm_image_enhancement() -> None:
    rows = {row["candidate_id"]: row for row in phase4_comparison_table()}
    expected = {
        "g2_type_13_delta_24": (Fraction(12), (2, 2, 2)),
        "g2_type_15_reconstructed": (Fraction(12), (2, 2, 2)),
        "g3_type_112_reconstructed": (Fraction(3), (1, 1, 1)),
        "g3_type_112_gaussian_bounded": (Fraction(6), (1, 1, 1)),
        "g3_type_113_eisenstein_bounded": (Fraction(12), (2, 2, 2)),
        "g3_type_122_gaussian_bounded": (Fraction(48), (1, 1, 1)),
    }
    assert set(rows) == set(expected)
    for candidate_id, (enhancement, control_images) in expected.items():
        assert rows[candidate_id]["logical_image_enhancement"] == enhancement
        assert rows[candidate_id]["control_logical_image_orders"] == control_images
