"""Audit Phase-9 winning/losing metrics with scalar defects and mpmath CVP."""

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
    compatible_tangent_model,
    fixed_radius_deformation,
    load_phase5_population_ledger,
    logical_gate_defect_model,
    measure_gate_robustness,
)
from gkp_passive_cliffords.preregistered_controls import form_from_population_row
from gkp_systole import high_precision_metric_systole


def scalar_action_defects(model, metric) -> tuple[float, ...]:
    g = np.asarray(metric, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(g)
    inverse_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    per_automorphism = []
    for transformation in model.automorphisms:
        difference = transformation.T @ g @ transformation - g
        whitened = inverse_sqrt @ difference @ inverse_sqrt
        per_automorphism.append(float(np.linalg.norm(whitened) / np.sqrt(g.shape[0])))
    return tuple(
        min(per_automorphism[index] for index in group)
        for group in model.action_groups
    )


def main() -> None:
    population, _summary = load_phase5_population_ledger(PROJECT / "data")
    by_id = {str(row["candidate_id"]): row for row in population}
    summaries = json.loads(
        (PROJECT / "data/phase9_gate_robustness_summary.json").read_text()
    )
    audits = []
    for summary in summaries:
        row = by_id[str(summary["candidate_id"])]
        form = form_from_population_row(row)
        metric = np.asarray(form.metric_core, dtype=float) / np.sqrt(form.order.radicand)
        tangent = compatible_tangent_model(form.alternating, metric)
        gates = logical_gate_defect_model(form)
        for extremum in ("best", "worst"):
            direction = summary[f"{extremum}_retention_direction"]
            deformation = fixed_radius_deformation(tangent, direction, float(summary["radius"]))
            vectorized = measure_gate_robustness(gates, deformation.metric)
            scalar = scalar_action_defects(gates, deformation.metric)
            high_precision_ell = high_precision_metric_systole(
                form.alternating,
                [
                    [format(float(value), ".17g") for value in metric_row]
                    for metric_row in deformation.metric
                ],
                decimal_places=60,
            )
            stored_ell = float(summary[f"{extremum}_retention_ell_ratio"]) * float(
                summary["cm_ell_squared"]
            )
            audits.append(
                {
                    "candidate_id": summary["candidate_id"],
                    "polarization_type": summary["polarization_type"],
                    "radius": summary["radius"],
                    "extremum": extremum,
                    "maximum_scalar_vectorized_defect_discrepancy": max(
                        abs(left - right)
                        for left, right in zip(scalar, vectorized.action_defects)
                    ),
                    "stored_ell_squared": stored_ell,
                    "high_precision_ell_squared": high_precision_ell,
                    "ell_absolute_discrepancy": abs(stored_ell - float(high_precision_ell)),
                    "enhanced_exact_action_count": vectorized.enhanced_epsilon_count(1e-8),
                }
            )
            print(
                tuple(summary["polarization_type"]),
                summary["radius"],
                extremum,
                f"defect={audits[-1]['maximum_scalar_vectorized_defect_discrepancy']:.2e}",
                f"ell={audits[-1]['ell_absolute_discrepancy']:.2e}",
                flush=True,
            )
    output = PROJECT / "data" / "phase9_high_precision_audit.json"
    output.write_text(json.dumps(audits, indent=2) + "\n")
    print(f"Wrote {len(audits)} Phase-9 audits")


if __name__ == "__main__":
    main()
