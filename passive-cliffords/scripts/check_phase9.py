"""Standalone checks for the Phase-9 gate-robustness experiment."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(WORKSPACE / "src"))

from gkp_passive_cliffords import PHASE9_RADII, load_json_artifact


def main() -> None:
    summaries = load_json_artifact(PROJECT / "data/phase9_gate_robustness_summary.json")
    evaluations = load_json_artifact(PROJECT / "data/phase9_gate_robustness_evaluations.json")
    audits = load_json_artifact(PROJECT / "data/phase9_high_precision_audit.json")
    assert len(summaries) == 5 * len(PHASE9_RADII) == 25
    assert len(evaluations) == 25 * 128 == 3200
    by_search = Counter((row["candidate_id"], float(row["radius"])) for row in evaluations)
    assert set(by_search.values()) == {128}
    for summary in summaries:
        assert int(summary["tangent_dimension"]) == int(summary["dimension_g"]) * (
            int(summary["dimension_g"]) + 1
        )
        assert float(summary["maximum_distance_error"]) < 2.1e-11
        assert float(summary["maximum_polarization_residual"]) < 2e-9
        assert float(summary["maximum_log_volume_residual"]) < 2e-9
    assert not any(int(row["exact_retained_enhanced_actions"]) for row in evaluations)
    summary_by_search = {
        (row["candidate_id"], float(row["radius"])): row for row in summaries
    }
    assert all(
        int(row["exact_retained_logical_actions"])
        == int(summary_by_search[(row["candidate_id"], float(row["radius"]))]["generic_minimal_image_order"])
        for row in evaluations
    )
    assert len(audits) == 50
    assert max(float(row["maximum_scalar_vectorized_defect_discrepancy"]) for row in audits) < 2e-12
    assert max(float(row["ell_absolute_discrepancy"]) for row in audits) < 2e-12
    assert not any(int(row["enhanced_exact_action_count"]) for row in audits)
    print("Phase-9 checks passed: 25 searches, 3,200 evaluations, 50 audits")


if __name__ == "__main__":
    main()
