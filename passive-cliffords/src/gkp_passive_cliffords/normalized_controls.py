"""Phase 7 generic controls at fixed affine-invariant metric distance.

The protocol samples deterministic symplectic directions and moves every CM
baseline to the same two intrinsic radii in the positive-definite metric
space.  This removes the main scale ambiguity exposed by Phase 6: fixed
transvection coefficients need not produce comparable metric displacement on
ill-conditioned inputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from math import log, pi, sqrt
from pathlib import Path
from random import Random
from statistics import mean, median, stdev
from typing import Sequence

import numpy as np

from gkp_systole import KernelGroup, MetricConvention, Polarization, compute_relative_systole

from .preregistered_controls import form_from_population_row
from .release_io import load_json_artifact


PHASE7_PROTOCOL_VERSION = "phase7-v1-equal-geodesic-distance"
PHASE7_SAMPLES_PER_CANDIDATE = 3
PHASE7_STEPS = 4
PHASE7_VECTOR_BOUND = 2
PHASE7_WEIGHT_DENOMINATOR = 10_000
PHASE7_DISTANCE_TOLERANCE = 2e-11


@dataclass(frozen=True)
class Phase7Radius:
    name: str
    target_rms_geodesic_distance: float


PHASE7_RADII = (
    Phase7Radius("near", 0.02),
    Phase7Radius("far", 0.10),
)


@dataclass(frozen=True)
class Phase7Direction:
    index: int
    vectors: tuple[tuple[int, ...], ...]
    weights: tuple[Fraction, ...]


@dataclass(frozen=True)
class Phase7ControlRecord:
    protocol_version: str
    candidate_id: str
    dimension_g: int
    polarization_type: tuple[int, ...]
    discriminant: int
    radius_name: str
    target_rms_geodesic_distance: float
    achieved_rms_geodesic_distance: float
    direction_index: int
    seed: int
    vectors: tuple[tuple[int, ...], ...]
    weights: tuple[Fraction, ...]
    radial_scale: float
    cm_ell_squared: float
    control_ell_squared: float
    ell_ratio_control_to_cm: float
    control_beats_cm: bool
    control_ties_cm: bool
    cm_logical_image_order: int
    generic_minimal_image_order: int
    cm_has_enhanced_passive_symmetry: bool
    polarization_residual: float
    log_volume_residual: float
    control_status: str = (
        "equal-distance generic-real symplectic deformation; non-CM intended, "
        "endomorphism ring not individually certified"
    )

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["polarization_type"] = list(self.polarization_type)
        result["vectors"] = [list(vector) for vector in self.vectors]
        result["weights"] = [
            {"numerator": value.numerator, "denominator": value.denominator}
            for value in self.weights
        ]
        return result


def phase7_seed(candidate_id: str) -> int:
    digest = hashlib.sha256(
        f"{PHASE7_PROTOCOL_VERSION}|{candidate_id}|direction".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def rms_affine_invariant_distance(
    baseline: Sequence[Sequence[float]],
    comparison: Sequence[Sequence[float]],
) -> float:
    """Return ||log(G0^-1/2 G1 G0^-1/2)||_F / sqrt(n).

    The normalization makes the displacement scale comparable between real
    dimensions four and six.  The distance is invariant under simultaneous
    real changes of basis by congruence.
    """

    g0 = np.asarray(baseline, dtype=float)
    g1 = np.asarray(comparison, dtype=float)
    if g0.shape != g1.shape or g0.ndim != 2 or g0.shape[0] != g0.shape[1]:
        raise ValueError("metrics must be square matrices of the same size")
    lower = np.linalg.cholesky(g0)
    left = np.linalg.solve(lower, g1)
    relative = np.linalg.solve(lower, left.T).T
    relative = (relative + relative.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(relative)
    if float(np.min(eigenvalues)) <= 0:
        raise ArithmeticError("relative metric lost positive definiteness")
    return sqrt(mean(log(float(value)) ** 2 for value in eigenvalues))


def _transvection(
    alternating: np.ndarray,
    vector: tuple[int, ...],
    parameter: float,
) -> np.ndarray:
    v = np.asarray(vector, dtype=float).reshape((-1, 1))
    return np.eye(alternating.shape[0]) + parameter * v @ (v.T @ alternating)


def transformation_on_direction(
    alternating: Sequence[Sequence[int]],
    direction: Phase7Direction,
    radial_scale: float,
) -> np.ndarray:
    a = np.asarray(alternating, dtype=float)
    transformation = np.eye(a.shape[0])
    for vector, weight in zip(direction.vectors, direction.weights):
        parameter = radial_scale * pi * float(weight)
        transformation = transformation @ _transvection(a, vector, parameter)
    return transformation


def deform_on_direction(
    alternating: Sequence[Sequence[int]],
    metric: Sequence[Sequence[float]],
    direction: Phase7Direction,
    radial_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    transformation = transformation_on_direction(alternating, direction, radial_scale)
    g = np.asarray(metric, dtype=float)
    deformed = transformation.T @ g @ transformation
    deformed = (deformed + deformed.T) / 2.0
    return transformation, deformed


def phase7_directions(
    candidate_id: str,
    real_dimension: int,
) -> tuple[int, tuple[Phase7Direction, ...]]:
    seed = phase7_seed(candidate_id)
    random = Random(seed)
    directions = []
    for index in range(PHASE7_SAMPLES_PER_CANDIDATE):
        vectors = []
        weights = []
        for _ in range(PHASE7_STEPS):
            vector = tuple(
                random.randint(-PHASE7_VECTOR_BOUND, PHASE7_VECTOR_BOUND)
                for _ in range(real_dimension)
            )
            while not any(vector):
                vector = tuple(
                    random.randint(-PHASE7_VECTOR_BOUND, PHASE7_VECTOR_BOUND)
                    for _ in range(real_dimension)
                )
            numerator = random.randint(-PHASE7_WEIGHT_DENOMINATOR, PHASE7_WEIGHT_DENOMINATOR)
            while numerator == 0:
                numerator = random.randint(
                    -PHASE7_WEIGHT_DENOMINATOR,
                    PHASE7_WEIGHT_DENOMINATOR,
                )
            vectors.append(vector)
            weights.append(Fraction(numerator, PHASE7_WEIGHT_DENOMINATOR))
        directions.append(
            Phase7Direction(index=index, vectors=tuple(vectors), weights=tuple(weights))
        )
    return seed, tuple(directions)


def equal_distance_deformation(
    alternating: Sequence[Sequence[int]],
    metric: Sequence[Sequence[float]],
    direction: Phase7Direction,
    target_distance: float,
    *,
    tolerance: float = PHASE7_DISTANCE_TOLERANCE,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Scale one fixed symplectic direction to an intrinsic target radius."""

    if target_distance <= 0 or tolerance <= 0:
        raise ValueError("target distance and tolerance must be positive")
    baseline = np.asarray(metric, dtype=float)

    def evaluate(scale: float) -> tuple[np.ndarray, np.ndarray, float]:
        transformation, deformed = deform_on_direction(
            alternating,
            baseline,
            direction,
            scale,
        )
        distance = rms_affine_invariant_distance(baseline, deformed)
        return transformation, deformed, distance

    lower = 0.0
    # Begin close to the identity.  Starting at scale one can create a very
    # ill-conditioned matrix before the small target radius is bracketed.
    upper = 1e-6
    transformation, deformed, distance = evaluate(upper)
    while distance < target_distance:
        lower = upper
        upper *= 2.0
        if upper > 2**20:
            raise ArithmeticError("failed to bracket the target metric distance")
        transformation, deformed, distance = evaluate(upper)

    best = (upper, transformation, deformed, distance)
    for _ in range(90):
        midpoint = (lower + upper) / 2.0
        transformation, deformed, distance = evaluate(midpoint)
        if abs(distance - target_distance) < abs(best[3] - target_distance):
            best = (midpoint, transformation, deformed, distance)
        if distance < target_distance:
            lower = midpoint
        else:
            upper = midpoint
        if abs(best[3] - target_distance) <= tolerance:
            break
    if abs(best[3] - target_distance) > tolerance:
        raise ArithmeticError("target metric distance was not reached to tolerance")
    return best


def _physical_metric(form) -> tuple[tuple[float, ...], ...]:
    scale = sqrt(form.order.radicand)
    return tuple(tuple(float(value) / scale for value in row) for row in form.metric_core)


def evaluate_phase7_candidate_controls(
    population_row: dict[str, object],
    radii: Sequence[Phase7Radius] = PHASE7_RADII,
) -> tuple[Phase7ControlRecord, ...]:
    form = form_from_population_row(population_row)
    polarization = Polarization(form.alternating)
    metric = _physical_metric(form)
    baseline = compute_relative_systole(
        polarization,
        metric,
        metric_convention=MetricConvention.POLARIZATION_SCALED,
    )
    cm_ell = float(population_row["ell_squared_numeric"])
    if abs(float(baseline.squared_systole) - cm_ell) > 2e-10:
        raise ArithmeticError("Phase-7 baseline disagrees with the Phase-5 ledger")
    seed, directions = phase7_directions(
        str(population_row["candidate_id"]),
        len(metric),
    )
    baseline_array = np.asarray(metric, dtype=float)
    a = np.asarray(form.alternating, dtype=float)
    baseline_logdet = float(np.linalg.slogdet(baseline_array)[1])
    cm_image = int(population_row["logical_image_order"])
    generic_image = int(population_row["generic_minimal_image_order"])
    records = []
    for direction in directions:
        for radius in radii:
            scale, transformation, deformed, achieved = equal_distance_deformation(
                form.alternating,
                metric,
                direction,
                radius.target_rms_geodesic_distance,
            )
            result = compute_relative_systole(
                polarization,
                deformed,
                metric_convention=MetricConvention.POLARIZATION_SCALED,
            )
            control_ell = float(result.squared_systole)
            tolerance = 1e-11 * max(1.0, abs(cm_ell), abs(control_ell))
            polarization_residual = float(
                np.max(np.abs(transformation.T @ a @ transformation - a))
            )
            log_volume_residual = abs(
                float(np.linalg.slogdet(deformed)[1]) - baseline_logdet
            )
            records.append(
                Phase7ControlRecord(
                    protocol_version=PHASE7_PROTOCOL_VERSION,
                    candidate_id=str(population_row["candidate_id"]),
                    dimension_g=int(population_row["dimension_g"]),
                    polarization_type=tuple(
                        int(value) for value in population_row["polarization_type"]
                    ),
                    discriminant=int(population_row["discriminant"]),
                    radius_name=radius.name,
                    target_rms_geodesic_distance=radius.target_rms_geodesic_distance,
                    achieved_rms_geodesic_distance=achieved,
                    direction_index=direction.index,
                    seed=seed,
                    vectors=direction.vectors,
                    weights=direction.weights,
                    radial_scale=scale,
                    cm_ell_squared=cm_ell,
                    control_ell_squared=control_ell,
                    ell_ratio_control_to_cm=control_ell / cm_ell,
                    control_beats_cm=control_ell > cm_ell + tolerance,
                    control_ties_cm=abs(control_ell - cm_ell) <= tolerance,
                    cm_logical_image_order=cm_image,
                    generic_minimal_image_order=generic_image,
                    cm_has_enhanced_passive_symmetry=cm_image > generic_image,
                    polarization_residual=polarization_residual,
                    log_volume_residual=log_volume_residual,
                )
            )
    return tuple(records)


def survey_phase7_controls(
    population_rows: Sequence[dict[str, object]],
    radii: Sequence[Phase7Radius] = PHASE7_RADII,
) -> tuple[Phase7ControlRecord, ...]:
    records = [
        record
        for row in population_rows
        for record in evaluate_phase7_candidate_controls(row, radii)
    ]
    return tuple(
        sorted(
            records,
            key=lambda record: (
                record.dimension_g,
                record.polarization_type,
                record.candidate_id,
                record.radius_name,
                record.direction_index,
            ),
        )
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def phase7_control_summary(
    records: Sequence[Phase7ControlRecord],
) -> list[dict[str, object]]:
    groups: dict[tuple[tuple[int, ...], str], list[Phase7ControlRecord]] = {}
    for record in records:
        groups.setdefault((record.polarization_type, record.radius_name), []).append(record)
    summaries = []
    for (polarization_type, radius_name), group in sorted(groups.items()):
        by_candidate: dict[str, list[Phase7ControlRecord]] = {}
        for record in group:
            by_candidate.setdefault(record.candidate_id, []).append(record)
        differences = []
        ratios = []
        wins = []
        enhanced_differences = []
        minimal_differences = []
        for candidate_group in by_candidate.values():
            cm = candidate_group[0].cm_ell_squared
            control = mean(record.control_ell_squared for record in candidate_group)
            difference = control - cm
            differences.append(difference)
            ratios.append(control / cm)
            wins.append(control > cm)
            target = (
                enhanced_differences
                if candidate_group[0].cm_has_enhanced_passive_symmetry
                else minimal_differences
            )
            target.append(difference)
        average = mean(differences)
        standard_error = (
            stdev(differences) / sqrt(len(differences))
            if len(differences) > 1
            else 0.0
        )
        summaries.append(
            {
                "polarization_type": list(polarization_type),
                "radius_name": radius_name,
                "target_rms_geodesic_distance": group[0].target_rms_geodesic_distance,
                "candidate_count": len(by_candidate),
                "control_count": len(group),
                "mean_cm_ell_squared": mean(record.cm_ell_squared for record in group),
                "mean_control_ell_squared": mean(
                    record.control_ell_squared for record in group
                ),
                "mean_paired_ell_difference": average,
                "paired_difference_ci95_low": average - 1.96 * standard_error,
                "paired_difference_ci95_high": average + 1.96 * standard_error,
                "mean_candidate_control_to_cm_ratio": mean(ratios),
                "median_candidate_control_to_cm_ratio": median(ratios),
                "candidate_control_mean_beats_cm_fraction": mean(wins),
                "individual_control_beats_cm_fraction": mean(
                    record.control_beats_cm for record in group
                ),
                "ratio_quantile_05": _quantile(ratios, 0.05),
                "ratio_quantile_95": _quantile(ratios, 0.95),
                "mean_paired_difference_enhanced_cm": (
                    mean(enhanced_differences) if enhanced_differences else None
                ),
                "mean_paired_difference_minimal_cm": (
                    mean(minimal_differences) if minimal_differences else None
                ),
                "enhanced_cm_candidate_count": len(enhanced_differences),
                "maximum_distance_error": max(
                    abs(
                        record.achieved_rms_geodesic_distance
                        - record.target_rms_geodesic_distance
                    )
                    for record in group
                ),
                "maximum_polarization_residual": max(
                    record.polarization_residual for record in group
                ),
                "maximum_log_volume_residual": max(
                    record.log_volume_residual for record in group
                ),
            }
        )
    return summaries


def write_phase7_control_ledger(
    records: Sequence[Phase7ControlRecord],
    output_directory: str | Path,
) -> tuple[Path, Path, Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rows = [record.as_dict() for record in records]
    summaries = phase7_control_summary(records)
    protocol = {
        "protocol_version": PHASE7_PROTOCOL_VERSION,
        "distance": (
            "RMS affine-invariant SPD distance: "
            "||log(G0^-1/2 G1 G0^-1/2)||_F / sqrt(2g)"
        ),
        "radii": [asdict(radius) for radius in PHASE7_RADII],
        "samples_per_candidate": PHASE7_SAMPLES_PER_CANDIDATE,
        "steps_per_direction": PHASE7_STEPS,
        "vector_bound": PHASE7_VECTOR_BOUND,
        "weight_denominator": PHASE7_WEIGHT_DENOMINATOR,
        "direction_seed": "SHA256(protocol_version|candidate_id|direction)",
        "same_directions_at_both_radii": True,
        "adaptive_resampling": False,
        "root_tolerance": PHASE7_DISTANCE_TOLERANCE,
        "primary_response": "candidate-level mean(control ell^2) - CM ell^2",
        "gate_audit": False,
        "control_status": (
            "generic-real equal-distance controls; individual endomorphism rings not certified"
        ),
    }
    json_path = output / "phase7_equal_distance_controls.json"
    csv_path = output / "phase7_equal_distance_controls.csv"
    summary_path = output / "phase7_equal_distance_controls_summary.json"
    protocol_path = output / "phase7_preregistered_protocol.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
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


def load_phase7_control_ledger(
    data_directory: str | Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    data = Path(data_directory)
    rows = load_json_artifact(data / "phase7_equal_distance_controls.json")
    summaries = load_json_artifact(data / "phase7_equal_distance_controls_summary.json")
    protocol = load_json_artifact(data / "phase7_preregistered_protocol.json")
    return rows, summaries, protocol


def high_precision_phase7_control(
    population_row: dict[str, object],
    control_row: dict[str, object],
    *,
    decimal_places: int = 60,
) -> str:
    """Independently recompute one stored control with mpmath CVP."""

    import mpmath as mp

    mp.mp.dps = decimal_places
    form = form_from_population_row(population_row)
    polarization = Polarization(form.alternating)
    size = len(form.metric_core)
    a = mp.matrix([[mp.mpf(value) for value in row] for row in form.alternating])
    metric_scale = mp.sqrt(form.order.radicand)
    g = mp.matrix(
        [
            [mp.mpf(value) / metric_scale for value in row]
            for row in form.metric_core
        ]
    )
    s = mp.eye(size)
    radial_scale = mp.mpf(str(control_row["radial_scale"]))
    for vector, weight_record in zip(control_row["vectors"], control_row["weights"]):
        v = mp.matrix([[mp.mpf(value)] for value in vector])
        weight = mp.mpf(weight_record["numerator"]) / weight_record["denominator"]
        elementary = mp.eye(size) + radial_scale * mp.pi * weight * v * (v.T * a)
        s = s * elementary
    deformed = s.T * g * s
    upper = mp.cholesky(deformed).T
    kernel = KernelGroup.from_polarization(polarization)
    global_best = None
    for element in kernel.nonzero_elements:
        coordinates = [
            mp.mpf(value.numerator) / value.denominator for value in element.coordinates
        ]
        current = [0 for _ in range(size)]
        initial = [int(mp.floor(-value + mp.mpf("0.5"))) for value in coordinates]
        x0 = mp.matrix([[coordinates[i] + initial[i]] for i in range(size)])
        class_best = (x0.T * deformed * x0)[0]

        def recurse(index, partial):
            nonlocal class_best
            if index < 0:
                class_best = min(class_best, partial)
                return
            tail = mp.fsum(
                upper[index, column] * (coordinates[column] + current[column])
                for column in range(index + 1, size)
            )
            diagonal = upper[index, index]
            remaining = max(mp.mpf("0"), class_best - partial)
            center = -coordinates[index] - tail / diagonal
            radius = mp.sqrt(remaining) / abs(diagonal)
            lower = int(mp.ceil(center - radius - mp.mpf("1e-50")))
            upper_integer = int(mp.floor(center + radius + mp.mpf("1e-50")))
            candidates = sorted(
                range(lower, upper_integer + 1),
                key=lambda value: abs(mp.mpf(value) - center),
            )
            for integer in candidates:
                current[index] = integer
                row_value = diagonal * (coordinates[index] + integer) + tail
                new_partial = partial + row_value * row_value
                if new_partial <= class_best + mp.mpf("1e-50"):
                    recurse(index - 1, new_partial)

        recurse(size - 1, mp.mpf("0"))
        global_best = class_best if global_best is None else min(global_best, class_best)
    return mp.nstr(global_best, decimal_places)
