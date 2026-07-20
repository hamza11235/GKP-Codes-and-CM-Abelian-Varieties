from fractions import Fraction

from gkp_systole import ImaginaryQuadraticOrder, QuadraticHermitianForm

from gkp_passive_cliffords import (
    Phase5PopulationSpec,
    compute_logical_action,
    compute_logical_action_orders,
    enumerate_hermitian_cm_automorphisms,
    evaluate_phase5_form,
    phase5_candidate_forms,
    phase5_population_summary,
)


def test_lightweight_action_orders_match_full_action() -> None:
    form = QuadraticHermitianForm(ImaginaryQuadraticOrder(-4), 2, 2, 1, 1)
    group = enumerate_hermitian_cm_automorphisms(form)
    full = compute_logical_action(group)
    light = compute_logical_action_orders(group)
    assert (light.automorphism_order, light.image_order, light.action_kernel_order) == (
        full.automorphism_order,
        full.image_order,
        full.action_kernel_order,
    )


def test_small_population_is_predefined_and_exact() -> None:
    spec = Phase5PopulationSpec(2, (1, 3), 3, 4, 6)
    forms = phase5_candidate_forms(spec)
    assert forms
    records = tuple(evaluate_phase5_form(form) for form in forms)
    assert all(record.polarization_type == (1, 3) for record in records)
    assert all(record.pairing_verified for record in records)
    assert all(
        record.polarized_automorphism_order
        == record.logical_image_order * record.action_kernel_order
        for record in records
    )
    summary = phase5_population_summary(records)[0]
    assert summary["candidate_count"] == len(records)
    assert 0 <= summary["extra_passive_symmetry_fraction"] <= 1


def test_population_record_keeps_exact_systole_coefficient() -> None:
    form = QuadraticHermitianForm(ImaginaryQuadraticOrder(-24), 6, 6, 3, -2)
    record = evaluate_phase5_form(form)
    assert record.ell_squared_coefficient == "4"
    assert record.ell_squared_exact == "4/sqrt(24)"
    assert record.logical_image_order == 24
    assert Fraction(record.target_coverage) == 1
