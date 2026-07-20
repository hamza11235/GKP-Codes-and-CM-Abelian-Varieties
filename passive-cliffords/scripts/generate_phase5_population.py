"""Generate the complete bounded Phase-5 CM population ledger."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(WORKSPACE / "src"))

from gkp_passive_cliffords import (
    PHASE5_POPULATION_SPECS,
    phase5_candidate_forms,
    phase5_population_summary,
    survey_phase5_population,
    write_phase5_population_ledger,
)


EXPECTED_COUNTS = {
    (1, 3): 876,
    (1, 5): 915,
    (1, 1, 2): 1051,
    (1, 1, 3): 1070,
    (1, 2, 2): 253,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=PROJECT / "data")
    args = parser.parse_args()

    records = []
    for spec in PHASE5_POPULATION_SPECS:
        candidate_count = len(phase5_candidate_forms(spec))
        expected = EXPECTED_COUNTS[spec.polarization_type]
        if candidate_count != expected:
            raise AssertionError(
                f"type {spec.polarization_type} generated {candidate_count}, expected {expected}"
            )
        print(f"Evaluating type {spec.polarization_type}: {candidate_count} candidates", flush=True)
        batch = survey_phase5_population((spec,), workers=args.workers)
        records.extend(batch)
        write_phase5_population_ledger(records, args.output)
        print(f"Completed type {spec.polarization_type}", flush=True)

    summaries = phase5_population_summary(records)
    print("\ntype | candidates | enhanced | fraction | best ell^2 | max image | correlation")
    print("--- | ---: | ---: | ---: | --- | ---: | ---:")
    for row in summaries:
        print(
            f"{tuple(row['polarization_type'])} | {row['candidate_count']} | "
            f"{row['extra_passive_symmetry_count']} | "
            f"{row['extra_passive_symmetry_fraction']:.6f} | "
            f"{row['best_ell_squared_exact']} | "
            f"{row['maximum_logical_image_order']} | "
            f"{row['distance_log_image_correlation']}"
        )
    print(f"\nWrote {len(records)} exact records to {args.output}")


if __name__ == "__main__":
    main()
