"""Phase 6 preregistered generic-real controls for the CM population ledger.

The protocol constants below are intentionally part of the source code and
are fixed before the result ledger is generated.  Every Phase-5 candidate
receives the same number of local and broad controls.  Seeds are derived from
SHA-256 of the stable candidate identifier and regime, so no adaptive
resampling or seed selection occurs after results are observed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import signal
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from fractions import Fraction
from math import sqrt
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

import numpy as np

from gkp_systole import (
    ImaginaryQuadraticOrder,
    QuadraticHermitianForm,
    TernaryQuadraticHermitianForm,
    high_precision_pi_systole,
    scan_pi_symplectic_deformations,
)

from .kernel_action import compute_logical_action_orders
from .numerical_automorphisms import (
    NumericalPolarizedAutomorphismProblem,
    enumerate_numerical_polarized_automorphisms,
)
from .release_io import load_json_artifact


PHASE6_PROTOCOL_VERSION = "phase6-v1-preregistered"
PHASE6_GATE_AUDIT_CANDIDATES_PER_TYPE_REGIME = 25
PHASE6_GATE_AUDIT_TIME_LIMIT_SECONDS = 1.0


class _GateAuditTimeout(TimeoutError):
    pass


def _gate_audit_timeout_handler(_signum, _frame) -> None:
    raise _GateAuditTimeout


@dataclass(frozen=True)
class Phase6ControlRegime:
    name: str
    samples_per_candidate: int
    amplitude: float
    steps: int
    vector_bound: int
    coefficient_denominator: int = 10_000_000


PHASE6_CONTROL_REGIMES = (
    Phase6ControlRegime("local", 3, 0.002, 4, 2),
    Phase6ControlRegime("broad", 3, 0.05, 4, 1),
)


def phase6_seed(candidate_id: str, regime: str) -> int:
    digest = hashlib.sha256(
        f"{PHASE6_PROTOCOL_VERSION}|{candidate_id}|{regime}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def form_from_population_row(
    row: dict[str, object],
) -> QuadraticHermitianForm | TernaryQuadraticHermitianForm:
    order = ImaginaryQuadraticOrder(int(row["discriminant"]))
    parameters = tuple(int(value) for value in row["form_parameters"])
    dimension = int(row["dimension_g"])
    if dimension == 2:
        return QuadraticHermitianForm(order, *parameters)
    if dimension == 3:
        return TernaryQuadraticHermitianForm(order, *parameters)
    raise ValueError(f"unsupported Phase-6 dimension {dimension}")


def _physical_metric(form) -> tuple[tuple[float, ...], ...]:
    scale = sqrt(form.order.radicand)
    return tuple(tuple(float(value) / scale for value in row) for row in form.metric_core)


@dataclass(frozen=True)
class Phase6ControlRecord:
    protocol_version: str
    candidate_id: str
    dimension_g: int
    polarization_type: tuple[int, ...]
    discriminant: int
    regime: str
    control_index: int
    seed: int
    amplitude: float
    steps: int
    vector_bound: int
    parameters: tuple[tuple[tuple[int, ...], Fraction], ...]
    cm_ell_squared: float
    control_ell_squared: float
    ell_ratio_control_to_cm: float
    control_beats_cm: bool
    control_ties_cm: bool
    relative_metric_displacement: float
    cm_logical_image_order: int
    generic_minimal_image_order: int
    gate_audited: bool
    gate_audit_status: str
    control_automorphism_order: int | None
    control_logical_image_order: int | None
    control_action_kernel_order: int | None
    control_has_extra_passive_symmetry: bool | None
    cm_image_exceeds_control: bool | None
    maximum_metric_residual: float | None
    control_status: str = (
        "generic-real pi deformation; non-CM almost surely, not individually certified"
    )

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["polarization_type"] = list(self.polarization_type)
        result["parameters"] = [
            {
                "vector": list(vector),
                "coefficient_numerator": coefficient.numerator,
                "coefficient_denominator": coefficient.denominator,
            }
            for vector, coefficient in self.parameters
        ]
        return result


def evaluate_phase6_candidate_controls(
    population_row: dict[str, object],
    regimes: Sequence[Phase6ControlRegime] = PHASE6_CONTROL_REGIMES,
) -> tuple[Phase6ControlRecord, ...]:
    form = form_from_population_row(population_row)
    metric = _physical_metric(form)
    baseline_array = np.asarray(metric, dtype=float)
    baseline_norm = float(np.linalg.norm(baseline_array))
    cm_ell = float(population_row["ell_squared_numeric"])
    cm_image = int(population_row["logical_image_order"])
    generic_image = int(population_row["generic_minimal_image_order"])
    audit_regimes = set(population_row.get("_phase6_gate_audit_regimes", ()))
    results: list[Phase6ControlRecord] = []

    for regime in regimes:
        seed = phase6_seed(str(population_row["candidate_id"]), regime.name)
        scan = scan_pi_symplectic_deformations(
            form.alternating,
            metric,
            sample_count=regime.samples_per_candidate,
            seed=seed,
            amplitude=regime.amplitude,
            steps=regime.steps,
            vector_bound=regime.vector_bound,
            coefficient_denominator=regime.coefficient_denominator,
        )
        if abs(float(scan.baseline_result.squared_systole) - cm_ell) > 2e-10:
            raise ArithmeticError("Phase-6 baseline disagrees with the Phase-5 ledger")
        for sample in scan.samples:
            gate_audited = regime.name in audit_regimes and sample.index == 0
            if gate_audited:
                previous_handler = signal.signal(signal.SIGALRM, _gate_audit_timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, PHASE6_GATE_AUDIT_TIME_LIMIT_SECONDS)
                try:
                    problem = NumericalPolarizedAutomorphismProblem(
                        polarization=scan.polarization,
                        metric=sample.metric,
                        tolerance=1e-8,
                    )
                    group = enumerate_numerical_polarized_automorphisms(problem)
                    action = compute_logical_action_orders(group)
                    automorphism_order = action.automorphism_order
                    logical_image_order = action.image_order
                    action_kernel_order = action.action_kernel_order
                    extra_symmetry = action.image_order > generic_image
                    cm_image_exceeds = cm_image > action.image_order
                    metric_residual = group.maximum_metric_residual
                    gate_audit_status = "certified"
                except _GateAuditTimeout:
                    automorphism_order = None
                    logical_image_order = None
                    action_kernel_order = None
                    extra_symmetry = None
                    cm_image_exceeds = None
                    metric_residual = None
                    gate_audit_status = "bounded_unresolved"
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0.0)
                    signal.signal(signal.SIGALRM, previous_handler)
            else:
                automorphism_order = None
                logical_image_order = None
                action_kernel_order = None
                extra_symmetry = None
                cm_image_exceeds = None
                metric_residual = None
                gate_audit_status = "not_selected"
            control_ell = float(sample.systole_result.squared_systole)
            displacement = float(
                np.linalg.norm(np.asarray(sample.metric, dtype=float) - baseline_array)
                / baseline_norm
            )
            tolerance = 1e-11 * max(1.0, abs(cm_ell), abs(control_ell))
            results.append(
                Phase6ControlRecord(
                    protocol_version=PHASE6_PROTOCOL_VERSION,
                    candidate_id=str(population_row["candidate_id"]),
                    dimension_g=int(population_row["dimension_g"]),
                    polarization_type=tuple(int(value) for value in population_row["polarization_type"]),
                    discriminant=int(population_row["discriminant"]),
                    regime=regime.name,
                    control_index=sample.index,
                    seed=seed,
                    amplitude=regime.amplitude,
                    steps=regime.steps,
                    vector_bound=regime.vector_bound,
                    parameters=sample.parameters,
                    cm_ell_squared=cm_ell,
                    control_ell_squared=control_ell,
                    ell_ratio_control_to_cm=control_ell / cm_ell,
                    control_beats_cm=control_ell > cm_ell + tolerance,
                    control_ties_cm=abs(control_ell - cm_ell) <= tolerance,
                    relative_metric_displacement=displacement,
                    cm_logical_image_order=cm_image,
                    generic_minimal_image_order=generic_image,
                    gate_audited=gate_audited,
                    gate_audit_status=gate_audit_status,
                    control_automorphism_order=automorphism_order,
                    control_logical_image_order=logical_image_order,
                    control_action_kernel_order=action_kernel_order,
                    control_has_extra_passive_symmetry=extra_symmetry,
                    cm_image_exceeds_control=cm_image_exceeds,
                    maximum_metric_residual=metric_residual,
                )
            )
    return tuple(results)


def _gate_audit_score(candidate_id: str, regime: str) -> bytes:
    return hashlib.sha256(
        f"{PHASE6_PROTOCOL_VERSION}|gate-audit|{candidate_id}|{regime}".encode("utf-8")
    ).digest()


def prepare_phase6_population_rows(
    population_rows: Sequence[dict[str, object]],
    *,
    audit_candidates_per_type_regime: int = PHASE6_GATE_AUDIT_CANDIDATES_PER_TYPE_REGIME,
) -> list[dict[str, object]]:
    """Attach outcome-independent gate-audit assignments to population rows."""

    prepared = [dict(row) for row in population_rows]
    assignments: dict[str, set[str]] = {str(row["candidate_id"]): set() for row in prepared}
    types = sorted({tuple(int(value) for value in row["polarization_type"]) for row in prepared})
    for polarization_type in types:
        group = [
            row
            for row in prepared
            if tuple(int(value) for value in row["polarization_type"]) == polarization_type
        ]
        if audit_candidates_per_type_regime > len(group):
            raise ValueError("gate-audit sample exceeds the candidate population")
        for regime in PHASE6_CONTROL_REGIMES:
            selected = sorted(
                group,
                key=lambda row: _gate_audit_score(str(row["candidate_id"]), regime.name),
            )[:audit_candidates_per_type_regime]
            for row in selected:
                assignments[str(row["candidate_id"])].add(regime.name)
    for row in prepared:
        row["_phase6_gate_audit_regimes"] = sorted(assignments[str(row["candidate_id"])])
    return prepared


def survey_phase6_controls(
    population_rows: Sequence[dict[str, object]],
    *,
    workers: int = 1,
    regimes: Sequence[Phase6ControlRegime] = PHASE6_CONTROL_REGIMES,
) -> tuple[Phase6ControlRecord, ...]:
    if workers <= 1:
        batches = (evaluate_phase6_candidate_controls(row, regimes) for row in population_rows)
        records = tuple(record for batch in batches for record in batch)
    else:
        # Regimes are fixed module-level protocol data for the parallel full run.
        if tuple(regimes) != PHASE6_CONTROL_REGIMES:
            raise ValueError("parallel Phase-6 runs use the preregistered module regimes")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            batches = executor.map(evaluate_phase6_candidate_controls, population_rows, chunksize=2)
            records = tuple(record for batch in batches for record in batch)
    return tuple(
        sorted(
            records,
            key=lambda record: (
                record.dimension_g,
                record.polarization_type,
                record.candidate_id,
                record.regime,
                record.control_index,
            ),
        )
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires nonempty data")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def phase6_control_summary(
    records: Sequence[Phase6ControlRecord],
) -> list[dict[str, object]]:
    groups: dict[tuple[tuple[int, ...], str], list[Phase6ControlRecord]] = {}
    for record in records:
        groups.setdefault((record.polarization_type, record.regime), []).append(record)
    summaries = []
    for (polarization_type, regime), group in sorted(groups.items()):
        ratios = [record.ell_ratio_control_to_cm for record in group]
        differences = [record.control_ell_squared - record.cm_ell_squared for record in group]
        audited = [record for record in group if record.gate_audited]
        certified = [record for record in audited if record.gate_audit_status == "certified"]
        summaries.append(
            {
                "polarization_type": list(polarization_type),
                "regime": regime,
                "candidate_count": len({record.candidate_id for record in group}),
                "control_count": len(group),
                "mean_cm_ell_squared": mean(record.cm_ell_squared for record in group),
                "mean_control_ell_squared": mean(record.control_ell_squared for record in group),
                "mean_paired_ell_difference": mean(differences),
                "median_control_to_cm_ratio": median(ratios),
                "ratio_quantile_05": _quantile(ratios, 0.05),
                "ratio_quantile_95": _quantile(ratios, 0.95),
                "control_beats_cm_count": sum(record.control_beats_cm for record in group),
                "control_beats_cm_fraction": sum(record.control_beats_cm for record in group)
                / len(group),
                "control_ties_cm_count": sum(record.control_ties_cm for record in group),
                "gate_audit_count": len(audited),
                "gate_audit_certified_count": len(certified),
                "gate_audit_unresolved_count": len(audited) - len(certified),
                "control_extra_passive_symmetry_count": sum(
                    bool(record.control_has_extra_passive_symmetry) for record in certified
                ),
                "control_logical_image_histogram": {
                    str(order): count
                    for order, count in sorted(
                        Counter(record.control_logical_image_order for record in certified).items()
                    )
                },
                "cm_image_exceeds_control_fraction": sum(
                    bool(record.cm_image_exceeds_control) for record in certified
                )
                / len(certified)
                if certified
                else None,
                "mean_relative_metric_displacement": mean(
                    record.relative_metric_displacement for record in group
                ),
                "maximum_metric_residual": (
                    max(float(record.maximum_metric_residual) for record in certified)
                    if certified
                    else None
                ),
            }
        )
    return summaries


def write_phase6_control_ledger(
    records: Sequence[Phase6ControlRecord],
    output_directory: str | Path,
) -> tuple[Path, Path, Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rows = [record.as_dict() for record in records]
    summary = phase6_control_summary(records)
    protocol = {
        "protocol_version": PHASE6_PROTOCOL_VERSION,
        "seed_derivation": "SHA256(protocol_version|candidate_id|regime), first 8 bytes mod (2^31-1)",
        "adaptive_resampling": False,
        "gate_audit": {
            "candidates_per_type_regime": PHASE6_GATE_AUDIT_CANDIDATES_PER_TYPE_REGIME,
            "control_index": 0,
            "selection": "smallest SHA256(protocol_version|gate-audit|candidate_id|regime)",
            "total_audited_controls": (
                PHASE6_GATE_AUDIT_CANDIDATES_PER_TYPE_REGIME
                * 5
                * len(PHASE6_CONTROL_REGIMES)
            ),
            "enumeration_time_limit_seconds": PHASE6_GATE_AUDIT_TIME_LIMIT_SECONDS,
            "timeout_status": "bounded_unresolved",
            "computational_amendment": (
                "Fixed before scientific outcomes after an exhaustive enumeration was observed "
                "to have pathological runtime; distance sampling and gate-audit selection unchanged."
            ),
        },
        "regimes": [asdict(regime) for regime in PHASE6_CONTROL_REGIMES],
        "control_status": (
            "generic-real pi deformations; non-CM almost surely; individual endomorphism rings not certified"
        ),
    }
    json_path = output / "phase6_generic_controls.json"
    csv_path = output / "phase6_generic_controls.csv"
    summary_path = output / "phase6_generic_controls_summary.json"
    protocol_path = output / "phase6_preregistered_protocol.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    fieldnames = tuple(rows[0]) if rows else ()
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )
    return json_path, csv_path, summary_path, protocol_path


def load_phase6_control_ledger(
    data_directory: str | Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    data = Path(data_directory)
    rows = load_json_artifact(data / "phase6_generic_controls.json")
    summary = load_json_artifact(data / "phase6_generic_controls_summary.json")
    protocol = load_json_artifact(data / "phase6_preregistered_protocol.json")
    return rows, summary, protocol


def high_precision_phase6_control(
    population_row: dict[str, object],
    control_row: dict[str, object],
    *,
    decimal_places: int = 60,
) -> str:
    """Recheck one selected control from its stored pi-deformation parameters."""

    import mpmath as mp

    form = form_from_population_row(population_row)
    parameters = tuple(
        (
            tuple(int(value) for value in item["vector"]),
            Fraction(
                int(item["coefficient_numerator"]),
                int(item["coefficient_denominator"]),
            ),
        )
        for item in control_row["parameters"]
    )
    core_value = high_precision_pi_systole(
        form.alternating,
        form.metric_core,
        parameters,
        decimal_places=decimal_places,
    )
    mp.mp.dps = decimal_places
    physical = mp.mpf(core_value) / mp.sqrt(form.order.radicand)
    return mp.nstr(physical, decimal_places)
