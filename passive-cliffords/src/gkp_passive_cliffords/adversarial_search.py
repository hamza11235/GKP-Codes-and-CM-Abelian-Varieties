"""Phase 8 adversarial searches on fixed-radius tangent spheres.

Phase 7 sampled three fixed rays around every CM record.  This module asks a
strictly harder question for selected records: after fixing an intrinsic
radius, can an optimizer find *any* direction whose relative systole exceeds
the CM baseline?

The search coordinates are intrinsic to the compatible-metric orbit.  Given
an alternating polarization matrix ``A`` and a compatible metric ``G``, the
symplectic Lie algebra is parameterized by

    K = A^{-1} H,  H = H^T.

The induced metric tangent is ``K.T @ G + G @ K``.  Its null directions are
the infinitesimal stabilizer of ``G`` and are removed by an SVD.  The resulting
orthonormal tangent basis has dimension ``g(g+1)``.  A unit vector in these
coordinates is exponentiated and rescaled to the requested affine-invariant
metric radius.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from math import ceil, log2, sqrt
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from gkp_systole import MetricConvention, Polarization, compute_relative_systole

from .normalized_controls import rms_affine_invariant_distance
from .preregistered_controls import form_from_population_row


PHASE8_PROTOCOL_VERSION = "phase8-v1-adversarial-tangent-sphere"
PHASE8_RADII = (0.005, 0.01, 0.02, 0.05, 0.10)
PHASE8_INITIAL_SOBOL = 32
PHASE8_SOBOL_HOLDOUT = 32
PHASE8_BO_STEPS = 32
PHASE8_ACQUISITION_POOL = 4096
PHASE8_UCB_KAPPA = 2.0
PHASE8_DISTANCE_TOLERANCE = 2e-11


@dataclass(frozen=True)
class CompatibleTangentModel:
    """Orthonormal effective tangent generators at one compatible metric."""

    alternating: np.ndarray
    metric: np.ndarray
    generators: tuple[np.ndarray, ...]
    tangent_gram: np.ndarray
    expected_dimension: int

    @property
    def tangent_dimension(self) -> int:
        return len(self.generators)


@dataclass(frozen=True)
class FixedRadiusDeformation:
    direction: tuple[float, ...]
    radial_scale: float
    transformation: np.ndarray
    metric: np.ndarray
    achieved_distance: float
    polarization_residual: float
    log_volume_residual: float


@dataclass(frozen=True)
class Phase8Evaluation:
    candidate_id: str
    polarization_type: tuple[int, ...]
    radius: float
    arm: str
    step: int
    direction: tuple[float, ...]
    ell_squared: float
    ell_ratio_to_cm: float
    radial_scale: float
    achieved_distance: float
    polarization_residual: float
    log_volume_residual: float

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["polarization_type"] = list(self.polarization_type)
        result["direction"] = list(self.direction)
        return result


@dataclass(frozen=True)
class Phase8SearchResult:
    protocol_version: str
    candidate_id: str
    dimension_g: int
    polarization_type: tuple[int, ...]
    discriminant: int
    radius: float
    tangent_dimension: int
    cm_ell_squared: float
    cm_logical_image_order: int
    generic_minimal_image_order: int
    initial_sobol_evaluations: int
    sobol_holdout_evaluations: int
    bayesian_evaluations: int
    sobol_best_ratio: float
    bayesian_best_ratio: float
    overall_best_ratio: float
    overall_best_ell_squared: float
    best_arm: str
    best_direction: tuple[float, ...]
    counterexample_found: bool
    maximum_distance_error: float
    maximum_polarization_residual: float
    maximum_log_volume_residual: float

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["polarization_type"] = list(self.polarization_type)
        result["best_direction"] = list(self.best_direction)
        return result


def _physical_metric(form) -> np.ndarray:
    scale = sqrt(form.order.radicand)
    return np.asarray(form.metric_core, dtype=float) / scale


def _symmetric_vector(matrix: np.ndarray) -> np.ndarray:
    """Vectorize a symmetric matrix while preserving its Frobenius norm."""

    values: list[float] = []
    for row in range(matrix.shape[0]):
        values.append(float(matrix[row, row]))
        for column in range(row + 1, matrix.shape[0]):
            values.append(sqrt(2.0) * float(matrix[row, column]))
    return np.asarray(values, dtype=float)


def compatible_tangent_model(
    alternating: Sequence[Sequence[int]],
    metric: Sequence[Sequence[float]],
    *,
    rank_tolerance: float = 1e-10,
) -> CompatibleTangentModel:
    """Construct the effective ``Sp/U`` tangent basis at ``metric``.

    The returned generators are normalized so a Euclidean unit coordinate
    vector has unit RMS affine-invariant tangent norm.
    """

    a = np.asarray(alternating, dtype=float)
    g = np.asarray(metric, dtype=float)
    if a.shape != g.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("alternating form and metric must be equally sized square matrices")
    if a.shape[0] % 2:
        raise ValueError("real phase-space dimension must be even")
    n = a.shape[0]
    complex_dimension = n // 2

    eigenvalues, eigenvectors = np.linalg.eigh(g)
    if float(np.min(eigenvalues)) <= 0:
        raise ValueError("metric must be positive definite")
    inverse_sqrt = (
        eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    )

    raw_generators: list[np.ndarray] = []
    tangent_columns: list[np.ndarray] = []
    for row in range(n):
        for column in range(row, n):
            h = np.zeros((n, n), dtype=float)
            h[row, column] = 1.0
            h[column, row] = 1.0
            if row == column:
                h[row, column] = 1.0
            k = np.linalg.solve(a, h)
            tangent = k.T @ g + g @ k
            whitened = inverse_sqrt @ tangent @ inverse_sqrt
            whitened = (whitened + whitened.T) / 2.0
            raw_generators.append(k)
            tangent_columns.append(_symmetric_vector(whitened) / sqrt(n))

    tangent_map = np.column_stack(tangent_columns)
    _u, singular_values, v_transpose = np.linalg.svd(tangent_map, full_matrices=False)
    cutoff = rank_tolerance * float(singular_values[0])
    rank = int(np.sum(singular_values > cutoff))
    expected = complex_dimension * (complex_dimension + 1)
    if rank != expected:
        raise ArithmeticError(f"effective tangent rank {rank}, expected {expected}")

    coefficient_basis = v_transpose[:rank].T / singular_values[:rank]
    generators = []
    for index in range(rank):
        generator = sum(
            coefficient_basis[raw_index, index] * raw_generators[raw_index]
            for raw_index in range(len(raw_generators))
        )
        generators.append(np.asarray(generator, dtype=float))

    # Recompute the effective tangent Gram matrix as an explicit numerical
    # certificate that the SVD normalization produced an orthonormal basis.
    effective_vectors = []
    for generator in generators:
        tangent = generator.T @ g + g @ generator
        whitened = inverse_sqrt @ tangent @ inverse_sqrt
        effective_vectors.append(_symmetric_vector(whitened) / sqrt(n))
    effective_map = np.column_stack(effective_vectors)
    tangent_gram = effective_map.T @ effective_map

    return CompatibleTangentModel(
        alternating=a,
        metric=g,
        generators=tuple(generators),
        tangent_gram=tangent_gram,
        expected_dimension=expected,
    )


def _unit_direction(direction: Sequence[float], dimension: int) -> np.ndarray:
    vector = np.asarray(direction, dtype=float)
    if vector.shape != (dimension,):
        raise ValueError(f"direction must have shape ({dimension},)")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-14:
        raise ValueError("direction must be nonzero")
    return vector / norm


def fixed_radius_deformation(
    model: CompatibleTangentModel,
    direction: Sequence[float],
    radius: float,
    *,
    tolerance: float = PHASE8_DISTANCE_TOLERANCE,
) -> FixedRadiusDeformation:
    """Exponentiate one unit tangent direction to an exact intrinsic radius."""

    from scipy.linalg import expm

    if radius <= 0:
        raise ValueError("radius must be positive")
    unit = _unit_direction(direction, model.tangent_dimension)
    generator = sum(
        coefficient * basis
        for coefficient, basis in zip(unit, model.generators)
    )
    baseline_logdet = float(np.linalg.slogdet(model.metric)[1])

    def evaluate(scale: float) -> tuple[np.ndarray, np.ndarray, float]:
        transformation = expm(scale * generator)
        deformed = transformation.T @ model.metric @ transformation
        deformed = (deformed + deformed.T) / 2.0
        distance = rms_affine_invariant_distance(model.metric, deformed)
        return transformation, deformed, distance

    lower = 0.0
    upper = max(radius, 1e-6)
    transformation, deformed, achieved = evaluate(upper)
    while achieved < radius:
        lower = upper
        upper *= 2.0
        if upper > 64.0:
            raise ArithmeticError("failed to bracket requested tangent-sphere radius")
        transformation, deformed, achieved = evaluate(upper)

    best = (upper, transformation, deformed, achieved)
    for _ in range(90):
        midpoint = (lower + upper) / 2.0
        transformation, deformed, achieved = evaluate(midpoint)
        if abs(achieved - radius) < abs(best[3] - radius):
            best = (midpoint, transformation, deformed, achieved)
        if achieved < radius:
            lower = midpoint
        else:
            upper = midpoint
        if abs(best[3] - radius) <= tolerance:
            break

    scale, transformation, deformed, achieved = best
    if abs(achieved - radius) > tolerance:
        raise ArithmeticError("requested tangent-sphere radius was not reached")
    polarization_residual = float(
        np.max(
            np.abs(
                transformation.T @ model.alternating @ transformation
                - model.alternating
            )
        )
    )
    log_volume_residual = abs(
        float(np.linalg.slogdet(deformed)[1]) - baseline_logdet
    )
    return FixedRadiusDeformation(
        direction=tuple(float(value) for value in unit),
        radial_scale=float(scale),
        transformation=transformation,
        metric=deformed,
        achieved_distance=float(achieved),
        polarization_residual=polarization_residual,
        log_volume_residual=log_volume_residual,
    )


def phase8_seed(candidate_id: str, radius: float, component: str) -> int:
    digest = hashlib.sha256(
        f"{PHASE8_PROTOCOL_VERSION}|{candidate_id}|{radius:.12g}|{component}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def sobol_sphere(dimension: int, count: int, seed: int) -> np.ndarray:
    """Return deterministic scrambled-Sobol directions on a unit sphere."""

    from scipy.stats import norm, qmc

    if dimension <= 0 or count <= 0:
        raise ValueError("dimension and count must be positive")
    power = int(ceil(log2(count)))
    points = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(power)
    gaussian = norm.ppf(np.clip(points[:count], 1e-12, 1.0 - 1e-12))
    norms = np.linalg.norm(gaussian, axis=1)
    return gaussian / norms[:, None]


def _bayesian_directions(
    initial_x: np.ndarray,
    initial_y: np.ndarray,
    acquisition_pool: np.ndarray,
    evaluate: Callable[[np.ndarray], float],
    steps: int,
    kappa: float,
) -> tuple[list[np.ndarray], list[float]]:
    """Sequential fixed-kernel Matérn-GP upper-confidence-bound search."""

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    x = [np.asarray(row, dtype=float) for row in initial_x]
    y = [float(value) for value in initial_y]
    available = np.ones(len(acquisition_pool), dtype=bool)
    selected: list[np.ndarray] = []
    values: list[float] = []
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
        length_scale=0.65,
        length_scale_bounds="fixed",
        nu=2.5,
    )
    for _ in range(steps):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=True,
            optimizer=None,
        )
        gp.fit(np.asarray(x), np.asarray(y))
        pool_indices = np.flatnonzero(available)
        candidates = acquisition_pool[pool_indices]
        posterior_mean, posterior_std = gp.predict(candidates, return_std=True)
        acquisition = posterior_mean + kappa * posterior_std
        chosen_local = int(np.argmax(acquisition))
        chosen_index = int(pool_indices[chosen_local])
        direction = acquisition_pool[chosen_index]
        value = float(evaluate(direction))
        available[chosen_index] = False
        x.append(direction)
        y.append(value)
        selected.append(direction)
        values.append(value)
    return selected, values


def evaluate_phase8_search(
    population_row: dict[str, object],
    radius: float,
    *,
    initial_sobol: int = PHASE8_INITIAL_SOBOL,
    sobol_holdout: int = PHASE8_SOBOL_HOLDOUT,
    bayesian_steps: int = PHASE8_BO_STEPS,
    acquisition_pool_size: int = PHASE8_ACQUISITION_POOL,
    ucb_kappa: float = PHASE8_UCB_KAPPA,
) -> tuple[Phase8SearchResult, tuple[Phase8Evaluation, ...]]:
    """Run matched-budget Sobol and Bayesian adversarial searches."""

    form = form_from_population_row(population_row)
    polarization = Polarization(form.alternating)
    baseline_metric = _physical_metric(form)
    model = compatible_tangent_model(form.alternating, baseline_metric)
    candidate_id = str(population_row["candidate_id"])
    polarization_type = tuple(int(v) for v in population_row["polarization_type"])
    cm_ell = float(population_row["ell_squared_numeric"])
    baseline = compute_relative_systole(
        polarization,
        baseline_metric,
        metric_convention=MetricConvention.POLARIZATION_SCALED,
    )
    if abs(float(baseline.squared_systole) - cm_ell) > 2e-10:
        raise ArithmeticError("Phase-8 baseline disagrees with Phase-5 ledger")

    shared = sobol_sphere(
        model.tangent_dimension,
        initial_sobol,
        phase8_seed(candidate_id, radius, "shared-sobol"),
    )
    holdout = sobol_sphere(
        model.tangent_dimension,
        sobol_holdout,
        phase8_seed(candidate_id, radius, "sobol-holdout"),
    )
    pool = sobol_sphere(
        model.tangent_dimension,
        acquisition_pool_size,
        phase8_seed(candidate_id, radius, "acquisition-pool"),
    )

    evaluations: list[Phase8Evaluation] = []

    def evaluate_direction(direction: np.ndarray, arm: str, step: int) -> float:
        deformation = fixed_radius_deformation(model, direction, radius)
        result = compute_relative_systole(
            polarization,
            deformation.metric,
            metric_convention=MetricConvention.POLARIZATION_SCALED,
        )
        ell = float(result.squared_systole)
        evaluations.append(
            Phase8Evaluation(
                candidate_id=candidate_id,
                polarization_type=polarization_type,
                radius=float(radius),
                arm=arm,
                step=step,
                direction=deformation.direction,
                ell_squared=ell,
                ell_ratio_to_cm=ell / cm_ell,
                radial_scale=deformation.radial_scale,
                achieved_distance=deformation.achieved_distance,
                polarization_residual=deformation.polarization_residual,
                log_volume_residual=deformation.log_volume_residual,
            )
        )
        return ell / cm_ell

    shared_values = np.asarray(
        [
            evaluate_direction(direction, "shared_sobol", index)
            for index, direction in enumerate(shared)
        ],
        dtype=float,
    )
    holdout_values = [
        evaluate_direction(direction, "sobol_holdout", index)
        for index, direction in enumerate(holdout)
    ]

    bo_directions, _bo_values = _bayesian_directions(
        shared,
        shared_values,
        pool,
        lambda direction: evaluate_direction(
            direction,
            "bayesian_ucb",
            sum(record.arm == "bayesian_ucb" for record in evaluations),
        ),
        bayesian_steps,
        ucb_kappa,
    )
    if len(bo_directions) != bayesian_steps:
        raise ArithmeticError("Bayesian search returned the wrong evaluation count")

    sobol_records = [
        record
        for record in evaluations
        if record.arm in {"shared_sobol", "sobol_holdout"}
    ]
    bayesian_records = [
        record
        for record in evaluations
        if record.arm in {"shared_sobol", "bayesian_ucb"}
    ]
    sobol_best = max(record.ell_ratio_to_cm for record in sobol_records)
    bayesian_best = max(record.ell_ratio_to_cm for record in bayesian_records)
    best = max(evaluations, key=lambda record: record.ell_ratio_to_cm)
    tolerance = 2e-10 * max(1.0, abs(cm_ell), abs(best.ell_squared))
    summary = Phase8SearchResult(
        protocol_version=PHASE8_PROTOCOL_VERSION,
        candidate_id=candidate_id,
        dimension_g=int(population_row["dimension_g"]),
        polarization_type=polarization_type,
        discriminant=int(population_row["discriminant"]),
        radius=float(radius),
        tangent_dimension=model.tangent_dimension,
        cm_ell_squared=cm_ell,
        cm_logical_image_order=int(population_row["logical_image_order"]),
        generic_minimal_image_order=int(population_row["generic_minimal_image_order"]),
        initial_sobol_evaluations=initial_sobol,
        sobol_holdout_evaluations=sobol_holdout,
        bayesian_evaluations=bayesian_steps,
        sobol_best_ratio=float(sobol_best),
        bayesian_best_ratio=float(bayesian_best),
        overall_best_ratio=float(best.ell_ratio_to_cm),
        overall_best_ell_squared=float(best.ell_squared),
        best_arm=best.arm,
        best_direction=best.direction,
        counterexample_found=best.ell_squared > cm_ell + tolerance,
        maximum_distance_error=max(
            abs(record.achieved_distance - radius) for record in evaluations
        ),
        maximum_polarization_residual=max(
            record.polarization_residual for record in evaluations
        ),
        maximum_log_volume_residual=max(
            record.log_volume_residual for record in evaluations
        ),
    )
    return summary, tuple(evaluations)


def phase8_champion_rows(
    population_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Select the highest-ell Phase-5 record for every surveyed type."""

    by_type: dict[tuple[int, ...], list[dict[str, object]]] = {}
    for row in population_rows:
        by_type.setdefault(tuple(int(v) for v in row["polarization_type"]), []).append(row)
    return [
        sorted(
            group,
            key=lambda row: (
                -float(row["ell_squared_numeric"]),
                -int(row["logical_image_order"]),
                str(row["candidate_id"]),
            ),
        )[0]
        for _polarization_type, group in sorted(by_type.items())
    ]


def write_phase8_results(
    summaries: Sequence[Phase8SearchResult],
    evaluations: Sequence[Phase8Evaluation],
    output_directory: str | Path,
) -> tuple[Path, Path, Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary_rows = [summary.as_dict() for summary in summaries]
    evaluation_rows = [evaluation.as_dict() for evaluation in evaluations]
    protocol = {
        "protocol_version": PHASE8_PROTOCOL_VERSION,
        "selection": (
            "highest Phase-5 ell^2 record within each polarization type; "
            "ties prefer larger logical image then lexicographically earliest stable ID"
        ),
        "radii": list(PHASE8_RADII),
        "tangent_coordinates": (
            "SVD-orthonormalized effective Sp(A)/U(G) tangent; dimension g(g+1)"
        ),
        "distance": (
            "RMS affine-invariant SPD distance "
            "||log(G0^-1/2 G1 G0^-1/2)||_F/sqrt(2g)"
        ),
        "sobol_budget": PHASE8_INITIAL_SOBOL + PHASE8_SOBOL_HOLDOUT,
        "bayesian_budget": PHASE8_INITIAL_SOBOL + PHASE8_BO_STEPS,
        "initial_sobol": PHASE8_INITIAL_SOBOL,
        "sobol_holdout": PHASE8_SOBOL_HOLDOUT,
        "bayesian_steps": PHASE8_BO_STEPS,
        "acquisition_pool": PHASE8_ACQUISITION_POOL,
        "surrogate": "fixed Matern-5/2 Gaussian process, length scale 0.65",
        "acquisition": f"UCB mean + {PHASE8_UCB_KAPPA} * posterior standard deviation",
        "seeds": "SHA256(protocol_version|candidate_id|radius|component)",
        "adaptive_resampling": "only the preregistered Bayesian UCB choices",
        "primary_failure_condition": "any evaluated ell^2/ell^2_CM > 1 beyond tolerance",
        "claim_boundary": (
            "adversarial numerical search, not exhaustive local or global certification"
        ),
    }
    summary_path = output / "phase8_adversarial_search_summary.json"
    evaluation_path = output / "phase8_adversarial_search_evaluations.json"
    csv_path = output / "phase8_adversarial_search_summary.csv"
    protocol_path = output / "phase8_adversarial_protocol.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2) + "\n")
    evaluation_path.write_text(json.dumps(evaluation_rows, indent=2) + "\n")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )
    return summary_path, evaluation_path, csv_path, protocol_path
