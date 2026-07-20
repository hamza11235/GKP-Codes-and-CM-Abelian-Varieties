"""High-precision audit of ten extremal Phase-7 controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(WORKSPACE / "src"))

from gkp_passive_cliffords import (
    high_precision_phase7_control,
    load_phase5_population_ledger,
    load_phase7_control_ledger,
)


def main() -> None:
    population, _ = load_phase5_population_ledger(PROJECT / "data")
    controls, _, _ = load_phase7_control_ledger(PROJECT / "data")
    lookup = {row["candidate_id"]: row for row in population}
    types = sorted({tuple(row["polarization_type"]) for row in controls})
    audits = []
    for polarization_type in types:
        for radius_name in ("near", "far"):
            subset = [
                row
                for row in controls
                if tuple(row["polarization_type"]) == polarization_type
                and row["radius_name"] == radius_name
            ]
            selected = max(subset, key=lambda row: row["ell_ratio_control_to_cm"])
            high_precision = high_precision_phase7_control(
                lookup[selected["candidate_id"]],
                selected,
                decimal_places=60,
            )
            audits.append(
                {
                    "polarization_type": list(polarization_type),
                    "radius_name": radius_name,
                    "candidate_id": selected["candidate_id"],
                    "direction_index": selected["direction_index"],
                    "screened_ell_squared": selected["control_ell_squared"],
                    "high_precision_ell_squared": high_precision,
                    "absolute_difference": abs(
                        float(high_precision) - selected["control_ell_squared"]
                    ),
                }
            )
    output = PROJECT / "data" / "phase7_high_precision_audit.json"
    output.write_text(json.dumps(audits, indent=2) + "\n")
    print(f"Wrote {len(audits)} high-precision audits; max difference = "
          f"{max(row['absolute_difference'] for row in audits):.3e}")


if __name__ == "__main__":
    main()
