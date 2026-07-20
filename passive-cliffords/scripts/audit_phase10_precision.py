"""High-precision audit of every Phase-10 method winner."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(WORKSPACE / "src"))

from gkp_passive_cliffords import canonical_product_family, rms_affine_invariant_distance
from gkp_systole import high_precision_metric_systole


def main() -> None:
    summary_path = PROJECT / "data" / "phase10_blind_search_summary.json"
    summaries = json.loads(summary_path.read_text())
    audits = []
    for row in summaries:
        polarization_type = tuple(int(value) for value in row["polarization_type"])
        family = canonical_product_family(polarization_type)
        metric = family.metric(row["best_coordinates"])
        high_precision = high_precision_metric_systole(
            family.alternating,
            [[format(float(value), ".17g") for value in metric_row] for metric_row in metric],
            decimal_places=70,
        )
        stored = float(row["best_ell_squared"])
        distance = rms_affine_invariant_distance(family.reference_metric, metric)
        audit = {
            "polarization_type": list(polarization_type),
            "radius": row["radius"],
            "method": row["method"],
            "stored_ell_squared": stored,
            "high_precision_ell_squared": high_precision,
            "ell_absolute_discrepancy": abs(stored - float(high_precision)),
            "stored_rms_distance": row["best_achieved_rms_distance"],
            "recomputed_rms_distance": distance,
            "distance_absolute_discrepancy": abs(
                float(row["best_achieved_rms_distance"]) - distance
            ),
            "within_preregistered_ball": distance <= float(row["radius"]) + 2e-10,
        }
        audits.append(audit)
        print(
            polarization_type,
            row["radius"],
            row["method"],
            f"ell discrepancy={audit['ell_absolute_discrepancy']:.2e}",
            flush=True,
        )
    output = PROJECT / "data" / "phase10_high_precision_audit.json"
    output.write_text(json.dumps(audits, indent=2) + "\n")
    print(f"Wrote {len(audits)} Phase-10 audits")


if __name__ == "__main__":
    main()
