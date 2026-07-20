"""Run the Phase-9 passive-gate radial robustness experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(WORKSPACE / "src"))

from gkp_passive_cliffords import (
    PHASE9_RADII,
    evaluate_phase9_search,
    load_phase5_population_ledger,
    phase9_champion_rows,
    write_phase9_results,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT / "data")
    args = parser.parse_args()
    population, _summary = load_phase5_population_ledger(args.data)
    champions = phase9_champion_rows(population)
    summaries = []
    evaluations = []
    for champion in champions:
        polarization_type = tuple(champion["polarization_type"])
        print(
            f"Champion {polarization_type}: image={champion['logical_image_order']}, "
            f"Aut={champion['polarized_automorphism_order']}",
            flush=True,
        )
        for radius in PHASE9_RADII:
            summary, rows = evaluate_phase9_search(champion, radius)
            summaries.append(summary)
            evaluations.extend(rows)
            print(
                f"  r={radius:.3f}: best={summary.overall_best_retention:.6f}, "
                f"worst={summary.overall_worst_retention:.6f}, "
                f"exact-enhanced={summary.best_retention_exact_enhanced_action_count}",
                flush=True,
            )
            write_phase9_results(summaries, evaluations, args.data)
    print(f"Wrote {len(summaries)} searches and {len(evaluations)} evaluations")


if __name__ == "__main__":
    main()
