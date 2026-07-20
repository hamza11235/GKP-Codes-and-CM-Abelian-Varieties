from gkp_passive_cliffords import (
    PHASE6_PROTOCOL_VERSION,
    Phase6ControlRegime,
    evaluate_phase6_candidate_controls,
    load_phase5_population_ledger,
    phase6_control_summary,
    phase6_seed,
)


def _one_population_row():
    from pathlib import Path

    project = Path(__file__).resolve().parents[1]
    rows, _ = load_phase5_population_ledger(project / "data")
    row = dict(next(row for row in rows if row["polarization_type"] == [1, 3]))
    row["_phase6_gate_audit_regimes"] = ["test-local", "test"]
    return row


def test_phase6_seed_is_stable_and_regime_specific() -> None:
    candidate = "g2_type_1_3_delta_m24_h_6_6_3_2"
    assert phase6_seed(candidate, "local") == phase6_seed(candidate, "local")
    assert phase6_seed(candidate, "local") != phase6_seed(candidate, "broad")
    assert PHASE6_PROTOCOL_VERSION == "phase6-v1-preregistered"


def test_small_preregistered_control_batch() -> None:
    row = _one_population_row()
    regimes = (
        Phase6ControlRegime("test-local", 2, 0.001, 2, 1),
    )
    controls = evaluate_phase6_candidate_controls(row, regimes)
    assert len(controls) == 2
    assert all(control.candidate_id == row["candidate_id"] for control in controls)
    assert all(control.regime == "test-local" for control in controls)
    audited = [control for control in controls if control.gate_audited]
    assert len(audited) == 1
    assert audited[0].gate_audit_status == "certified"
    assert all(control.control_automorphism_order == 2 for control in audited)
    assert all(
        control.control_automorphism_order
        == control.control_logical_image_order * control.control_action_kernel_order
        for control in audited
    )
    assert all(control.maximum_metric_residual < 1e-10 for control in audited)


def test_phase6_summary_preserves_paired_counts() -> None:
    row = _one_population_row()
    regime = Phase6ControlRegime("test", 3, 0.001, 2, 1)
    controls = evaluate_phase6_candidate_controls(row, (regime,))
    summary = phase6_control_summary(controls)[0]
    assert summary["candidate_count"] == 1
    assert summary["control_count"] == 3
    assert summary["control_extra_passive_symmetry_count"] == 0
    assert 0 < summary["median_control_to_cm_ratio"] < 2
