"""Run Phase 10 blind searches and only then reveal the CM comparison."""

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
    PHASE10_POLARIZATION_TYPES,
    PHASE10_RADII,
    compare_phase10_with_cm,
    load_phase5_population_ledger,
    phase8_champion_rows,
    run_phase10_blind_search,
    write_phase10_results,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT / "data")
    args = parser.parse_args()
    summaries = []
    evaluations = []
    checkpoint = args.data / "phase10_blind_search_checkpoint.json"

    # Blind stage: no population or CM ledger is loaded anywhere above this line.
    for polarization_type in PHASE10_POLARIZATION_TYPES:
        print(f"Blind type D={polarization_type}", flush=True)
        for radius in PHASE10_RADII:
            method_summaries, records = run_phase10_blind_search(
                polarization_type, radius
            )
            summaries.extend(method_summaries)
            evaluations.extend(records)
            winners = sorted(
                (
                    (item.method, item.best_ell_squared)
                    for item in method_summaries
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            print(
                f"  R={radius:.2f}: "
                + ", ".join(f"{name}={value:.9f}" for name, value in winners),
                flush=True,
            )
            checkpoint.write_text(
                json.dumps(
                    {
                        "stage": "blind; CM data not loaded",
                        "completed_searches": len(summaries),
                        "completed_evaluations": len(evaluations),
                        "summaries": [item.as_dict() for item in summaries],
                    },
                    indent=2,
                )
                + "\n"
            )

    # Reveal stage: the CM population is loaded only after all blind queries finish.
    population, _population_summary = load_phase5_population_ledger(args.data)
    champions = phase8_champion_rows(population)
    comparisons = compare_phase10_with_cm(summaries, champions)
    write_phase10_results(summaries, evaluations, comparisons, args.data)
    print(
        f"Wrote {len(summaries)} method summaries, {len(evaluations)} blind "
        f"evaluations, and {len(comparisons)} post-hoc comparisons",
        flush=True,
    )


if __name__ == "__main__":
    main()
