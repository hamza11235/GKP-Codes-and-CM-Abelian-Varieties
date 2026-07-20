"""Generate the complete preregistered Phase-6 generic-control ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(WORKSPACE / "src"))

from gkp_passive_cliffords import (
    PHASE6_CONTROL_REGIMES,
    high_precision_phase6_control,
    load_phase5_population_ledger,
    phase6_control_summary,
    prepare_phase6_population_rows,
    survey_phase6_controls,
    write_phase6_control_ledger,
)


EXPECTED_CANDIDATES = {
    (1, 3): 876,
    (1, 5): 915,
    (1, 1, 2): 1051,
    (1, 1, 3): 1070,
    (1, 2, 2): 253,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--data", type=Path, default=PROJECT / "data")
    args = parser.parse_args()

    population_rows, _ = load_phase5_population_ledger(args.data)
    population_rows = prepare_phase6_population_rows(population_rows)
    all_controls = []
    for polarization_type, expected in EXPECTED_CANDIDATES.items():
        population = [
            row for row in population_rows if tuple(row["polarization_type"]) == polarization_type
        ]
        if len(population) != expected:
            raise AssertionError(
                f"type {polarization_type} has {len(population)} candidates, expected {expected}"
            )
        expected_controls = expected * sum(
            regime.samples_per_candidate for regime in PHASE6_CONTROL_REGIMES
        )
        print(
            f"Evaluating type {polarization_type}: {expected_controls} preregistered controls",
            flush=True,
        )
        controls = survey_phase6_controls(population, workers=args.workers)
        all_controls.extend(controls)
        write_phase6_control_ledger(all_controls, args.data)
        print(f"Completed type {polarization_type}", flush=True)

    # Independently recheck the largest control/CM ratio in each type and regime.
    control_rows = [record.as_dict() for record in all_controls]
    population_lookup = {row["candidate_id"]: row for row in population_rows}
    audits = []
    for polarization_type in EXPECTED_CANDIDATES:
        for regime in (item.name for item in PHASE6_CONTROL_REGIMES):
            subset = [
                row
                for row in control_rows
                if tuple(row["polarization_type"]) == polarization_type
                and row["regime"] == regime
            ]
            selected = max(subset, key=lambda row: row["ell_ratio_control_to_cm"])
            high_precision = high_precision_phase6_control(
                population_lookup[selected["candidate_id"]],
                selected,
                decimal_places=60,
            )
            audits.append(
                {
                    "polarization_type": list(polarization_type),
                    "regime": regime,
                    "candidate_id": selected["candidate_id"],
                    "control_index": selected["control_index"],
                    "screened_ell_squared": selected["control_ell_squared"],
                    "high_precision_ell_squared": high_precision,
                    "absolute_difference": abs(
                        float(high_precision) - float(selected["control_ell_squared"])
                    ),
                }
            )
    audit_path = args.data / "phase6_high_precision_audit.json"
    audit_path.write_text(json.dumps(audits, indent=2) + "\n")

    print("\ntype | regime | controls | beat CM | mean difference | median ratio | CM image wins")
    print("--- | --- | ---: | ---: | ---: | ---: | ---:")
    for row in phase6_control_summary(all_controls):
        gate_result = row["cm_image_exceeds_control_fraction"]
        gate_text = f"{gate_result:.6f}" if gate_result is not None else "unresolved"
        print(
            f"{tuple(row['polarization_type'])} | {row['regime']} | "
            f"{row['control_count']} | {row['control_beats_cm_fraction']:.6f} | "
            f"{row['mean_paired_ell_difference']:.6f} | "
            f"{row['median_control_to_cm_ratio']:.6f} | "
            f"{gate_text}"
        )
    print(f"\nWrote {len(all_controls)} preregistered controls to {args.data}")


if __name__ == "__main__":
    main()
