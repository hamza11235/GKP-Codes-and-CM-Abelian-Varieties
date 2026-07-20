"""Standalone Phase-8 geometry and ledger checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(WORKSPACE / "src"))

from gkp_passive_cliffords import (
    PHASE8_RADII,
    compatible_tangent_model,
    fixed_radius_deformation,
    load_phase5_population_ledger,
    phase8_champion_rows,
    sobol_sphere,
)
from gkp_passive_cliffords.preregistered_controls import form_from_population_row


def main() -> None:
    population, _summary = load_phase5_population_ledger(PROJECT / "data")
    champions = phase8_champion_rows(population)
    assert len(champions) == 5
    for champion in champions:
        form = form_from_population_row(champion)
        metric = np.asarray(form.metric_core, dtype=float) / np.sqrt(form.order.radicand)
        model = compatible_tangent_model(form.alternating, metric)
        dimension = int(champion["dimension_g"])
        assert model.tangent_dimension == dimension * (dimension + 1)
        assert np.max(np.abs(model.tangent_gram - np.eye(model.tangent_dimension))) < 2e-10
        direction = sobol_sphere(model.tangent_dimension, 1, seed=dimension)[0]
        deformation = fixed_radius_deformation(model, direction, 0.02)
        assert abs(deformation.achieved_distance - 0.02) < 2e-11
        assert deformation.polarization_residual < 2e-10
        assert deformation.log_volume_residual < 2e-10

    summary_path = PROJECT / "data" / "phase8_adversarial_search_summary.json"
    if summary_path.exists():
        rows = json.loads(summary_path.read_text())
        assert len(rows) == len(champions) * len(PHASE8_RADII)
        for row in rows:
            assert int(row["tangent_dimension"]) == int(row["dimension_g"]) * (
                int(row["dimension_g"]) + 1
            )
            assert int(row["initial_sobol_evaluations"]) == 32
            assert int(row["sobol_holdout_evaluations"]) == 32
            assert int(row["bayesian_evaluations"]) == 32
            assert float(row["maximum_distance_error"]) < 2e-11
            assert float(row["maximum_polarization_residual"]) < 2e-9
            assert float(row["maximum_log_volume_residual"]) < 2e-9
        print(f"Phase-8 ledger checks passed for {len(rows)} searches")
        audit_path = PROJECT / "data" / "phase8_high_precision_audit.json"
        if audit_path.exists():
            audits = json.loads(audit_path.read_text())
            assert len(audits) == len(rows)
            assert not any(row["counterexample_after_audit"] for row in audits)
            assert max(float(row["absolute_discrepancy"]) for row in audits) < 2e-12
            print(f"Phase-8 high-precision checks passed for {len(audits)} winners")
    else:
        print("Phase-8 geometry checks passed; result ledger not generated yet")


if __name__ == "__main__":
    main()
