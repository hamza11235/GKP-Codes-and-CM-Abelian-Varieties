"""Validate the complete Phase-7 equal-distance control ledger."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(WORKSPACE / "src"))

from gkp_passive_cliffords import (
    PHASE7_DISTANCE_TOLERANCE,
    PHASE7_PROTOCOL_VERSION,
    load_phase7_control_ledger,
)


EXPECTED_CANDIDATES = {
    (1, 3): 876,
    (1, 5): 915,
    (1, 1, 2): 1051,
    (1, 1, 3): 1070,
    (1, 2, 2): 253,
}


def main() -> None:
    rows, summaries, protocol = load_phase7_control_ledger(PROJECT / "data")
    if protocol["protocol_version"] != PHASE7_PROTOCOL_VERSION:
        raise AssertionError("Phase-7 protocol version changed")
    if protocol["adaptive_resampling"] is not False:
        raise AssertionError("Phase 7 must not adaptively resample")
    if len(rows) != 4165 * 6:
        raise AssertionError(f"expected 24,990 controls, found {len(rows)}")
    if max(
        abs(row["achieved_rms_geodesic_distance"] - row["target_rms_geodesic_distance"])
        for row in rows
    ) > PHASE7_DISTANCE_TOLERANCE:
        raise AssertionError("a control missed its target metric radius")
    if max(row["polarization_residual"] for row in rows) > 2e-9:
        raise AssertionError("a control failed polarization preservation")
    if max(row["log_volume_residual"] for row in rows) > 2e-9:
        raise AssertionError("a control failed volume preservation")
    for summary in summaries:
        polarization_type = tuple(summary["polarization_type"])
        if summary["candidate_count"] != EXPECTED_CANDIDATES[polarization_type]:
            raise AssertionError("candidate count changed")
        if summary["control_count"] != 3 * EXPECTED_CANDIDATES[polarization_type]:
            raise AssertionError("control count changed")

    by_candidate_direction = defaultdict(list)
    for row in rows:
        by_candidate_direction[(row["candidate_id"], row["direction_index"])].append(row)
    if not all(len(pair) == 2 for pair in by_candidate_direction.values()):
        raise AssertionError("each direction must appear at both radii")
    for pair in by_candidate_direction.values():
        if pair[0]["vectors"] != pair[1]["vectors"] or pair[0]["weights"] != pair[1]["weights"]:
            raise AssertionError("radii did not share the same direction")

    audits = json.loads(
        (PROJECT / "data" / "phase7_high_precision_audit.json").read_text()
    )
    if len(audits) != 10 or max(row["absolute_difference"] for row in audits) > 2e-11:
        raise AssertionError("Phase-7 high-precision audit failed")

    print("type | radius | candidates | mean control-CM | 95% interval | mean ratio")
    print("--- | --- | ---: | ---: | --- | ---:")
    for row in summaries:
        interval = (
            f"[{row['paired_difference_ci95_low']:.6f}, "
            f"{row['paired_difference_ci95_high']:.6f}]"
        )
        print(
            f"{tuple(row['polarization_type'])} | {row['radius_name']} | "
            f"{row['candidate_count']} | {row['mean_paired_ell_difference']:.6f} | "
            f"{interval} | {row['mean_candidate_control_to_cm_ratio']:.6f}"
        )
    print("\nPhase 7 equal-distance-control assertions passed.")


if __name__ == "__main__":
    main()
