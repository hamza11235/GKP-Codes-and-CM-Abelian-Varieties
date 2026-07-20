from pathlib import Path

import numpy as np

from gkp_passive_cliffords import (
    PHASE7_PROTOCOL_VERSION,
    Phase7Radius,
    evaluate_phase7_candidate_controls,
    load_phase5_population_ledger,
    phase7_control_summary,
    phase7_seed,
    rms_affine_invariant_distance,
)


def _one_population_row():
    project = Path(__file__).resolve().parents[1]
    rows, _ = load_phase5_population_ledger(project / "data")
    return next(row for row in rows if row["polarization_type"] == [1, 3])


def test_affine_invariant_metric_distance_is_congruence_invariant() -> None:
    baseline = np.asarray([[2.0, 0.3], [0.3, 1.2]])
    comparison = np.asarray([[1.5, -0.1], [-0.1, 1.8]])
    change = np.asarray([[2.0, 1.0], [-1.0, 1.0]])
    original = rms_affine_invariant_distance(baseline, comparison)
    transformed = rms_affine_invariant_distance(
        change.T @ baseline @ change,
        change.T @ comparison @ change,
    )
    assert abs(original - transformed) < 1e-13


def test_phase7_equal_distance_controls_hit_frozen_radius() -> None:
    row = _one_population_row()
    radius = Phase7Radius("test", 0.015)
    controls = evaluate_phase7_candidate_controls(row, (radius,))
    assert PHASE7_PROTOCOL_VERSION == "phase7-v1-equal-geodesic-distance"
    assert len(controls) == 3
    assert all(abs(record.achieved_rms_geodesic_distance - 0.015) < 2e-11 for record in controls)
    assert all(record.polarization_type == (1, 3) for record in controls)
    assert all(record.polarization_residual < 1e-11 for record in controls)
    assert all(record.log_volume_residual < 1e-11 for record in controls)


def test_phase7_radii_share_directions_and_summary_is_candidate_level() -> None:
    row = _one_population_row()
    radii = (Phase7Radius("near-test", 0.01), Phase7Radius("far-test", 0.03))
    controls = evaluate_phase7_candidate_controls(row, radii)
    assert phase7_seed(str(row["candidate_id"])) == controls[0].seed
    for index in range(3):
        pair = [record for record in controls if record.direction_index == index]
        assert len(pair) == 2
        assert pair[0].vectors == pair[1].vectors
        assert pair[0].weights == pair[1].weights
        near = next(record for record in pair if record.radius_name == "near-test")
        far = next(record for record in pair if record.radius_name == "far-test")
        assert near.radial_scale < far.radial_scale
    summaries = phase7_control_summary(controls)
    assert len(summaries) == 2
    assert all(summary["candidate_count"] == 1 for summary in summaries)
    assert all(summary["control_count"] == 3 for summary in summaries)
