"""Run Phase 8 on the five within-type Phase-5 CM champions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(WORKSPACE / "src"))

from gkp_passive_cliffords import (
    PHASE8_RADII,
    evaluate_phase8_search,
    load_phase5_population_ledger,
    phase8_champion_rows,
    write_phase8_results,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT / "data")
    args = parser.parse_args()

    population, _summary = load_phase5_population_ledger(args.data)
    champions = phase8_champion_rows(population)
    summaries = []
    evaluations = []
    for champion in champions:
        polarization_type = tuple(champion["polarization_type"])
        print(
            f"Champion {polarization_type}: {champion['candidate_id']} "
            f"(ell^2={champion['ell_squared_numeric']:.12g})",
            flush=True,
        )
        for radius in PHASE8_RADII:
            summary, rows = evaluate_phase8_search(champion, radius)
            summaries.append(summary)
            evaluations.extend(rows)
            print(
                f"  r={radius:.3f} sobol={summary.sobol_best_ratio:.9f} "
                f"BO={summary.bayesian_best_ratio:.9f} "
                f"counterexample={summary.counterexample_found}",
                flush=True,
            )
            write_phase8_results(summaries, evaluations, args.data)
    print(f"Wrote {len(summaries)} searches and {len(evaluations)} evaluations")


if __name__ == "__main__":
    main()
