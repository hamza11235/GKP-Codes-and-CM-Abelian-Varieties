"""Independently audit every Phase-8 winning metric with mpmath CVP."""

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
)
from gkp_passive_cliffords.preregistered_controls import form_from_population_row
from gkp_systole import high_precision_metric_systole


def main() -> None:
    population, _population_summary = load_phase5_population_ledger(PROJECT / "data")
    by_id = {str(row["candidate_id"]): row for row in population}
    summaries = json.loads(
        (PROJECT / "data" / "phase8_adversarial_search_summary.json").read_text()
    )
    audits = []
    for summary in summaries:
        population_row = by_id[str(summary["candidate_id"])]
        form = form_from_population_row(population_row)
        metric = np.asarray(form.metric_core, dtype=float) / np.sqrt(form.order.radicand)
        model = compatible_tangent_model(form.alternating, metric)
        deformation = fixed_radius_deformation(
            model,
            summary["best_direction"],
            float(summary["radius"]),
        )
        high_precision = high_precision_metric_systole(
            form.alternating,
            [[format(float(value), ".17g") for value in row] for row in deformation.metric],
            decimal_places=60,
        )
        stored = float(summary["overall_best_ell_squared"])
        recomputed = float(high_precision)
        audits.append(
            {
                "candidate_id": summary["candidate_id"],
                "polarization_type": summary["polarization_type"],
                "radius": summary["radius"],
                "stored_ell_squared": stored,
                "high_precision_ell_squared": high_precision,
                "absolute_discrepancy": abs(stored - recomputed),
                "counterexample_after_audit": (
                    recomputed
                    > float(summary["cm_ell_squared"])
                    + 2e-10 * max(1.0, abs(float(summary["cm_ell_squared"])))
                ),
            }
        )
        print(
            tuple(summary["polarization_type"]),
            summary["radius"],
            f"discrepancy={audits[-1]['absolute_discrepancy']:.3e}",
            flush=True,
        )
    output = PROJECT / "data" / "phase8_high_precision_audit.json"
    output.write_text(json.dumps(audits, indent=2) + "\n")
    print(
        f"Wrote {len(audits)} audits; maximum discrepancy "
        f"{max(row['absolute_discrepancy'] for row in audits):.3e}"
    )


if __name__ == "__main__":
    main()
