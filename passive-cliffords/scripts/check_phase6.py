"""Validate the complete Phase-6 generic-control ledger."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(WORKSPACE / "src"))

from gkp_passive_cliffords import (
    PHASE6_PROTOCOL_VERSION,
    load_phase6_control_ledger,
)


EXPECTED_CANDIDATES = {
    (1, 3): 876,
    (1, 5): 915,
    (1, 1, 2): 1051,
    (1, 1, 3): 1070,
    (1, 2, 2): 253,
}


def main() -> None:
    rows, summaries, protocol = load_phase6_control_ledger(PROJECT / "data")
    if protocol["protocol_version"] != PHASE6_PROTOCOL_VERSION:
        raise AssertionError("protocol version changed")
    if protocol["adaptive_resampling"] is not False:
        raise AssertionError("Phase 6 must not adaptively resample")
    if len(rows) != 4165 * 6:
        raise AssertionError(f"expected 24,990 controls, found {len(rows)}")
    for row in summaries:
        polarization_type = tuple(row["polarization_type"])
        if row["candidate_count"] != EXPECTED_CANDIDATES[polarization_type]:
            raise AssertionError("candidate count changed")
        if row["control_count"] != 3 * EXPECTED_CANDIDATES[polarization_type]:
            raise AssertionError("control count changed")
    audited = [row for row in rows if row["gate_audited"]]
    if len(audited) != 250:
        raise AssertionError(f"expected 250 gate audits, found {len(audited)}")
    certified = [row for row in audited if row["gate_audit_status"] == "certified"]
    unresolved = [row for row in audited if row["gate_audit_status"] == "bounded_unresolved"]
    if len(certified) + len(unresolved) != len(audited):
        raise AssertionError("unknown gate-audit status")
    if not all(
        row["control_automorphism_order"]
        == row["control_logical_image_order"] * row["control_action_kernel_order"]
        for row in certified
    ):
        raise AssertionError("a control failed the image-kernel identity")
    if not all(row["maximum_metric_residual"] < 1e-8 for row in certified):
        raise AssertionError("a numerical automorphism failed the metric check")
    audits = json.loads((PROJECT / "data" / "phase6_high_precision_audit.json").read_text())
    if len(audits) != 10 or not all(row["absolute_difference"] < 2e-12 for row in audits):
        raise AssertionError("high-precision audit failed")

    print("type | regime | controls | beat CM | mean difference | generic extra gates")
    print("--- | --- | ---: | ---: | ---: | ---:")
    for row in summaries:
        print(
            f"{tuple(row['polarization_type'])} | {row['regime']} | "
            f"{row['control_count']} | {row['control_beats_cm_fraction']:.6f} | "
            f"{row['mean_paired_ell_difference']:.6f} | "
            f"{row['control_extra_passive_symmetry_count']}"
        )
    print(
        f"\nPhase 6 preregistered-control assertions passed "
        f"({len(certified)} certified gate audits, {len(unresolved)} bounded unresolved)."
    )


if __name__ == "__main__":
    main()
