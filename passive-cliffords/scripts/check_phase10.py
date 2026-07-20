"""Dependency-light validation for Phase 10."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(WORKSPACE / "src"))

from gkp_passive_cliffords import (
    PHASE10_METHOD_BUDGET,
    PHASE10_POLARIZATION_TYPES,
    PHASE10_RADII,
    canonical_product_family,
    intrinsic_coordinate_radius,
    load_json_artifact,
    rms_affine_invariant_distance,
)


def main() -> None:
    data = PROJECT / "data"
    summaries = load_json_artifact(data / "phase10_blind_search_summary.json")
    evaluations = load_json_artifact(data / "phase10_blind_search_evaluations.json")
    comparisons = load_json_artifact(data / "phase10_posthoc_cm_comparison.json")
    expected_searches = len(PHASE10_POLARIZATION_TYPES) * len(PHASE10_RADII)
    assert len(summaries) == expected_searches * 3
    assert len(evaluations) == expected_searches * 3 * PHASE10_METHOD_BUDGET
    assert len(comparisons) == expected_searches
    assert max(row["achieved_rms_distance"] - row["radius"] for row in evaluations) < 2e-9
    assert all(not row["blind_beats_cm"] for row in comparisons)
    audits = load_json_artifact(data / "phase10_high_precision_audit.json")
    assert len(audits) == len(summaries)
    assert max(row["ell_absolute_discrepancy"] for row in audits) < 2e-12

    for polarization_type in PHASE10_POLARIZATION_TYPES:
        family = canonical_product_family(polarization_type)
        coordinate_radius = intrinsic_coordinate_radius(len(polarization_type), 0.37)
        coordinates = np.zeros(family.coordinate_dimension)
        coordinates[0] = coordinate_radius
        metric = family.metric(coordinates)
        assert abs(
            rms_affine_invariant_distance(family.reference_metric, metric) - 0.37
        ) < 3e-12
        family.validate_coordinates(coordinates)
    print("Phase 10 checks passed")


if __name__ == "__main__":
    main()
