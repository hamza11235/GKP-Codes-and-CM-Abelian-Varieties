"""Generate the complete preregistered Phase-7 equal-distance controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(WORKSPACE / "src"))

from gkp_passive_cliffords import (
    PHASE7_RADII,
    PHASE7_SAMPLES_PER_CANDIDATE,
    load_phase5_population_ledger,
    phase7_control_summary,
    survey_phase7_controls,
    write_phase7_control_ledger,
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
    parser.add_argument("--data", type=Path, default=PROJECT / "data")
    args = parser.parse_args()

    population_rows, _ = load_phase5_population_ledger(args.data)
    all_controls = []
    for polarization_type, expected in EXPECTED_CANDIDATES.items():
        population = [
            row
            for row in population_rows
            if tuple(row["polarization_type"]) == polarization_type
        ]
        if len(population) != expected:
            raise AssertionError(
                f"type {polarization_type} has {len(population)} candidates, expected {expected}"
            )
        expected_controls = expected * PHASE7_SAMPLES_PER_CANDIDATE * len(PHASE7_RADII)
        print(
            f"Evaluating type {polarization_type}: {expected_controls} equal-distance controls",
            flush=True,
        )
        controls = survey_phase7_controls(population)
        all_controls.extend(controls)
        write_phase7_control_ledger(all_controls, args.data)
        print(f"Completed type {polarization_type}", flush=True)

    print("\ntype | radius | candidates | mean control-CM | 95% interval | mean ratio")
    print("--- | --- | ---: | ---: | --- | ---:")
    for row in phase7_control_summary(all_controls):
        interval = (
            f"[{row['paired_difference_ci95_low']:.6f}, "
            f"{row['paired_difference_ci95_high']:.6f}]"
        )
        print(
            f"{tuple(row['polarization_type'])} | {row['radius_name']} | "
            f"{row['candidate_count']} | {row['mean_paired_ell_difference']:.6f} | "
            f"{interval} | {row['mean_candidate_control_to_cm_ratio']:.6f}"
        )
    print(f"\nWrote {len(all_controls)} equal-distance controls to {args.data}")


if __name__ == "__main__":
    main()
