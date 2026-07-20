"""Phase 10 blind global searches on fixed-polarization moduli spaces.

The search oracle exposes only the relative-systole objective ``ell^2``,
computed by exhaustive finite-kernel enumeration and floating-point CVP.
Every run starts from the canonical product metric ``diag(D,D)`` and uses the
full ``g(g+1)``-dimensional compatible-metric chart.  CM labels, passive-gate
data, and the Phase-5 champions are deliberately absent from the search
functions and enter only in :func:`compare_phase10_with_cm`.

The chart is global (the exponential Cartan parametrization of ``Sp/U``), but
the experiments are bounded by an affine-invariant RMS distance from the
canonical product metric.  Consequently these are reproducible bounded
global searches, not proofs of global optimality on a noncompact moduli space.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from math import ceil, log, log2, sqrt
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from gkp_systole import (
    CompatibleMetricFamily,
    MetricConvention,
    canonical_alternating,
    compute_relative_systole,
)

from .normalized_controls import rms_affine_invariant_distance
from .preregistered_controls import form_from_population_row


PHASE10_PROTOCOL_VERSION = "phase10-v1-blind-bounded-global-moduli-search"
PHASE10_POLARIZATION_TYPES = (
    (1, 3),
    (1, 5),
    (1, 1, 2),
    (1, 1, 3),
    (1, 2, 2),
)
PHASE10_RADII = (0.25, 0.50, 1.00, 1.50)
PHASE10_METHOD_BUDGET = 96
PHASE10_BO_INITIAL = 32
PHASE10_BO_STEPS = 64
PHASE10_ACQUISITION_POOL = 2048
PHASE10_UCB_KAPPA = 2.0
PHASE10_CMA_RESTARTS = 2


@dataclass(frozen=True)
class Phase10Evaluation:
    protocol_version: str
    polarization_type: tuple[int, ...]
    radius: float
    method: str
    query: int
    coordinates: tuple[float, ...]
    coordinate_norm: float
    achieved_rms_distance: float
    boundary_fraction: float
    ell_squared: float
    class_multiplicity: int
    lift_multiplicity: int
    kernel_distance_spectrum: tuple[float, ...]
    certified: bool

    def as_dict(self) -> dict[str, object]:
        record = asdict(self)
        record["polarization_type"] = list(self.polarization_type)
        record["coordinates"] = list(self.coordinates)
        record["kernel_distance_spectrum"] = list(self.kernel_distance_spectrum)
        return record


@dataclass(frozen=True)
class Phase10BlindSummary:
    protocol_version: str
    polarization_type: tuple[int, ...]
    dimension_g: int
    coordinate_dimension: int
    radius: float
    method: str
    budget: int
    best_ell_squared: float
    best_query: int
    best_coordinates: tuple[float, ...]
    best_achieved_rms_distance: float
    best_boundary_fraction: float
    best_class_multiplicity: int
    best_lift_multiplicity: int
    best_kernel_distance_spectrum: tuple[float, ...]
    maximum_distance_overrun: float
    all_certified: bool

    def as_dict(self) -> dict[str, object]:
        record = asdict(self)
        record["polarization_type"] = list(self.polarization_type)
        record["best_coordinates"] = list(self.best_coordinates)
        record["best_kernel_distance_spectrum"] = list(
            self.best_kernel_distance_spectrum
        )
        return record


@dataclass(frozen=True)
class Phase10CMComparison:
    protocol_version: str
    polarization_type: tuple[int, ...]
    radius: float
    best_method: str
    best_blind_ell_squared: float
    cm_candidate_id: str
    cm_discriminant: int
    cm_ell_squared: float
    cm_logical_image_order: int
    cm_automorphism_order: int
    blind_to_cm_ratio: float
    blind_beats_cm: bool
    blind_ties_cm: bool
    spectrum_rms_difference: float
    spectrum_max_difference: float
    same_kernel_spectrum_at_tolerance: bool

    def as_dict(self) -> dict[str, object]:
        record = asdict(self)
        record["polarization_type"] = list(self.polarization_type)
        return record


def phase10_seed(
    polarization_type: Sequence[int], radius: float, component: str
) -> int:
    type_token = ",".join(str(int(value)) for value in polarization_type)
    payload = (
        f"{PHASE10_PROTOCOL_VERSION}|{type_token}|{radius:.12g}|{component}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def canonical_product_family(
    polarization_type: Sequence[int],
) -> CompatibleMetricFamily:
    """Return the blind canonical ``diag(D,D)`` compatible family."""

    values = tuple(int(value) for value in polarization_type)
    alternating = canonical_alternating(values)
    metric = np.diag(values + values)
    baseline = compute_relative_systole(
        alternating,
        metric,
        metric_convention=MetricConvention.POLARIZATION_SCALED,
    )
    return CompatibleMetricFamily.from_reference(
        name=f"canonical_product_type_{'_'.join(map(str, values))}",
        alternating=alternating,
        reference_metric=metric,
        reference_exact_ell_squared=str(baseline.squared_systole),
        reference_ell_squared=float(baseline.squared_systole),
        reference_cm="none: canonical product reference selected from D only",
    )


def intrinsic_coordinate_radius(dimension_g: int, radius: float) -> float:
    """Convert affine-invariant RMS radius to Cartan-coordinate radius."""

    if dimension_g <= 0 or radius <= 0:
        raise ValueError("dimension and radius must be positive")
    return float(radius) * sqrt(2 * dimension_g) / 2.0


def sobol_ball(
    dimension: int,
    count: int,
    seed: int,
    coordinate_radius: float,
) -> np.ndarray:
    """Deterministic scrambled-Sobol points uniform in a Euclidean ball."""

    from scipy.stats import norm, qmc

    if dimension <= 0 or count <= 0 or coordinate_radius <= 0:
        raise ValueError("dimension, count, and radius must be positive")
    power = int(ceil(log2(count)))
    points = qmc.Sobol(d=dimension + 1, scramble=True, seed=seed).random_base2(power)
    points = points[:count]
    gaussian = norm.ppf(np.clip(points[:, :dimension], 1e-12, 1.0 - 1e-12))
    directions = gaussian / np.linalg.norm(gaussian, axis=1)[:, None]
    radii = coordinate_radius * points[:, dimension] ** (1.0 / dimension)
    return directions * radii[:, None]


def _project_to_ball(vector: np.ndarray, radius: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= radius:
        return vector
    return vector * (radius / norm)


def _kernel_spectrum(result) -> tuple[float, ...]:
    return tuple(
        sorted(float(item.squared_distance) for item in result.class_results)
    )


def _evaluate_coordinates(
    family: CompatibleMetricFamily,
    polarization_type: tuple[int, ...],
    radius: float,
    coordinate_radius: float,
    method: str,
    query: int,
    coordinates: Sequence[float],
) -> Phase10Evaluation:
    vector = np.asarray(coordinates, dtype=float)
    if vector.shape != (family.coordinate_dimension,):
        raise ValueError("coordinate vector has the wrong shape")
    if float(np.linalg.norm(vector)) > coordinate_radius + 2e-10:
        raise ValueError("query lies outside the preregistered intrinsic ball")
    metric = family.metric(vector)
    result = compute_relative_systole(
        family.alternating,
        metric,
        metric_convention=MetricConvention.POLARIZATION_SCALED,
    )
    distance = rms_affine_invariant_distance(family.reference_metric, metric)
    return Phase10Evaluation(
        protocol_version=PHASE10_PROTOCOL_VERSION,
        polarization_type=polarization_type,
        radius=float(radius),
        method=method,
        query=int(query),
        coordinates=tuple(float(value) for value in vector),
        coordinate_norm=float(np.linalg.norm(vector)),
        achieved_rms_distance=float(distance),
        boundary_fraction=float(distance / radius),
        ell_squared=float(result.squared_systole),
        class_multiplicity=result.class_multiplicity,
        lift_multiplicity=result.lift_multiplicity,
        kernel_distance_spectrum=_kernel_spectrum(result),
        certified=result.certified,
    )


def _run_sobol(
    family: CompatibleMetricFamily,
    polarization_type: tuple[int, ...],
    radius: float,
    budget: int,
) -> list[Phase10Evaluation]:
    coordinate_radius = intrinsic_coordinate_radius(len(polarization_type), radius)
    points = sobol_ball(
        family.coordinate_dimension,
        budget,
        phase10_seed(polarization_type, radius, "sobol"),
        coordinate_radius,
    )
    return [
        _evaluate_coordinates(
            family,
            polarization_type,
            radius,
            coordinate_radius,
            "sobol",
            index,
            point,
        )
        for index, point in enumerate(points)
    ]


def _run_bayesian_ucb(
    family: CompatibleMetricFamily,
    polarization_type: tuple[int, ...],
    radius: float,
    initial: int,
    steps: int,
    acquisition_pool_size: int,
    kappa: float,
) -> list[Phase10Evaluation]:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    coordinate_radius = intrinsic_coordinate_radius(len(polarization_type), radius)
    initial_points = sobol_ball(
        family.coordinate_dimension,
        initial,
        phase10_seed(polarization_type, radius, "bo-initial"),
        coordinate_radius,
    )
    pool = sobol_ball(
        family.coordinate_dimension,
        acquisition_pool_size,
        phase10_seed(polarization_type, radius, "bo-pool"),
        coordinate_radius,
    )
    evaluations = [
        _evaluate_coordinates(
            family,
            polarization_type,
            radius,
            coordinate_radius,
            "bayesian_ucb",
            index,
            point,
        )
        for index, point in enumerate(initial_points)
    ]
    x = [np.asarray(record.coordinates) / coordinate_radius for record in evaluations]
    y = [record.ell_squared for record in evaluations]
    available = np.ones(len(pool), dtype=bool)
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
        length_scale=0.65,
        length_scale_bounds="fixed",
        nu=2.5,
    )
    for step in range(steps):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=True,
            optimizer=None,
        )
        gp.fit(np.asarray(x), np.asarray(y))
        indices = np.flatnonzero(available)
        candidates = pool[indices]
        mean, std = gp.predict(candidates / coordinate_radius, return_std=True)
        selected_local = int(np.argmax(mean + kappa * std))
        selected_index = int(indices[selected_local])
        record = _evaluate_coordinates(
            family,
            polarization_type,
            radius,
            coordinate_radius,
            "bayesian_ucb",
            initial + step,
            pool[selected_index],
        )
        available[selected_index] = False
        evaluations.append(record)
        x.append(np.asarray(record.coordinates) / coordinate_radius)
        y.append(record.ell_squared)
    return evaluations


def _cma_es_points(
    dimension: int,
    budget: int,
    coordinate_radius: float,
    seed: int,
    evaluate: Callable[[np.ndarray], float],
    restarts: int,
) -> list[np.ndarray]:
    """Small deterministic implementation of standard rank-mu CMA-ES."""

    if budget < restarts or restarts <= 0:
        raise ValueError("CMA budget must accommodate every restart")
    rng = np.random.default_rng(seed)
    lambda_ = 4 + int(3 * log(dimension))
    mu = lambda_ // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mueff = 1.0 / float(np.sum(weights**2))
    cc = (4.0 + mueff / dimension) / (
        dimension + 4.0 + 2.0 * mueff / dimension
    )
    cs = (mueff + 2.0) / (dimension + mueff + 5.0)
    c1 = 2.0 / ((dimension + 1.3) ** 2 + mueff)
    cmu = min(
        1.0 - c1,
        2.0 * (mueff - 2.0 + 1.0 / mueff)
        / ((dimension + 2.0) ** 2 + mueff),
    )
    damps = 1.0 + 2.0 * max(
        0.0, sqrt((mueff - 1.0) / (dimension + 1.0)) - 1.0
    ) + cs
    chi_n = sqrt(dimension) * (
        1.0 - 1.0 / (4.0 * dimension) + 1.0 / (21.0 * dimension**2)
    )
    restart_budgets = [budget // restarts] * restarts
    for index in range(budget % restarts):
        restart_budgets[index] += 1

    evaluated_points: list[np.ndarray] = []
    for restart, restart_budget in enumerate(restart_budgets):
        if restart == 0:
            mean = np.zeros(dimension)
        else:
            direction = rng.normal(size=dimension)
            direction /= np.linalg.norm(direction)
            mean = direction * coordinate_radius * 0.35
        sigma = coordinate_radius * (0.30 if restart == 0 else 0.45)
        covariance = np.eye(dimension)
        path_c = np.zeros(dimension)
        path_sigma = np.zeros(dimension)
        used = 0
        generation = 0
        while used < restart_budget:
            population = min(lambda_, restart_budget - used)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-14)
            sqrt_covariance = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
            inverse_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
            z = rng.normal(size=(population, dimension))
            y = z @ sqrt_covariance.T
            points = np.asarray(
                [_project_to_ball(mean + sigma * row, coordinate_radius) for row in y]
            )
            values = np.asarray([evaluate(point) for point in points])
            evaluated_points.extend(np.asarray(point) for point in points)
            used += population
            selected_count = min(mu, population)
            selected_indices = np.argsort(values)[::-1][:selected_count]
            selected_weights = weights[:selected_count].copy()
            selected_weights /= np.sum(selected_weights)
            old_mean = mean.copy()
            mean = np.sum(points[selected_indices] * selected_weights[:, None], axis=0)
            mean = _project_to_ball(mean, coordinate_radius)
            y_selected = (points[selected_indices] - old_mean) / max(sigma, 1e-15)
            y_weighted = np.sum(y_selected * selected_weights[:, None], axis=0)
            path_sigma = (1.0 - cs) * path_sigma + sqrt(
                cs * (2.0 - cs) * mueff
            ) * (inverse_sqrt @ y_weighted)
            generation += 1
            norm_path = float(np.linalg.norm(path_sigma))
            denominator = sqrt(max(1e-15, 1.0 - (1.0 - cs) ** (2 * generation)))
            hsig = float(
                norm_path / denominator
                < (1.4 + 2.0 / (dimension + 1.0)) * chi_n
            )
            path_c = (1.0 - cc) * path_c + hsig * sqrt(
                cc * (2.0 - cc) * mueff
            ) * y_weighted
            rank_mu = sum(
                weight * np.outer(row, row)
                for weight, row in zip(selected_weights, y_selected)
            )
            covariance = (
                (1.0 - c1 - cmu + c1 * (1.0 - hsig) * cc * (2.0 - cc))
                * covariance
                + c1 * np.outer(path_c, path_c)
                + cmu * rank_mu
            )
            covariance = (covariance + covariance.T) / 2.0
            sigma *= np.exp((cs / damps) * (norm_path / chi_n - 1.0))
            sigma = float(np.clip(sigma, coordinate_radius * 1e-5, coordinate_radius))
    if len(evaluated_points) != budget:
        raise ArithmeticError("CMA-ES consumed the wrong query budget")
    return evaluated_points


def _run_cma_es(
    family: CompatibleMetricFamily,
    polarization_type: tuple[int, ...],
    radius: float,
    budget: int,
    restarts: int,
) -> list[Phase10Evaluation]:
    coordinate_radius = intrinsic_coordinate_radius(len(polarization_type), radius)
    evaluations: list[Phase10Evaluation] = []

    def objective(point: np.ndarray) -> float:
        record = _evaluate_coordinates(
            family,
            polarization_type,
            radius,
            coordinate_radius,
            "cma_es",
            len(evaluations),
            point,
        )
        evaluations.append(record)
        return record.ell_squared

    _cma_es_points(
        family.coordinate_dimension,
        budget,
        coordinate_radius,
        phase10_seed(polarization_type, radius, "cma-es"),
        objective,
        restarts,
    )
    return evaluations


def _summarize_method(
    family: CompatibleMetricFamily,
    polarization_type: tuple[int, ...],
    radius: float,
    method: str,
    evaluations: Sequence[Phase10Evaluation],
) -> Phase10BlindSummary:
    records = [record for record in evaluations if record.method == method]
    if not records:
        raise ValueError(f"no evaluations for method {method}")
    best = max(records, key=lambda item: (item.ell_squared, -item.query))
    return Phase10BlindSummary(
        protocol_version=PHASE10_PROTOCOL_VERSION,
        polarization_type=polarization_type,
        dimension_g=len(polarization_type),
        coordinate_dimension=family.coordinate_dimension,
        radius=float(radius),
        method=method,
        budget=len(records),
        best_ell_squared=best.ell_squared,
        best_query=best.query,
        best_coordinates=best.coordinates,
        best_achieved_rms_distance=best.achieved_rms_distance,
        best_boundary_fraction=best.boundary_fraction,
        best_class_multiplicity=best.class_multiplicity,
        best_lift_multiplicity=best.lift_multiplicity,
        best_kernel_distance_spectrum=best.kernel_distance_spectrum,
        maximum_distance_overrun=max(
            0.0, max(record.achieved_rms_distance - radius for record in records)
        ),
        all_certified=all(record.certified for record in records),
    )


def run_phase10_blind_search(
    polarization_type: Sequence[int],
    radius: float,
    *,
    method_budget: int = PHASE10_METHOD_BUDGET,
    bo_initial: int = PHASE10_BO_INITIAL,
    bo_steps: int = PHASE10_BO_STEPS,
    acquisition_pool_size: int = PHASE10_ACQUISITION_POOL,
    ucb_kappa: float = PHASE10_UCB_KAPPA,
    cma_restarts: int = PHASE10_CMA_RESTARTS,
) -> tuple[tuple[Phase10BlindSummary, ...], tuple[Phase10Evaluation, ...]]:
    """Run three blind equal-budget optimizers for one fixed ``(D,radius)``."""

    values = tuple(int(value) for value in polarization_type)
    if bo_initial + bo_steps != method_budget:
        raise ValueError("Bayesian initial and adaptive budgets must equal method budget")
    family = canonical_product_family(values)
    records: list[Phase10Evaluation] = []
    records.extend(_run_sobol(family, values, radius, method_budget))
    records.extend(
        _run_cma_es(family, values, radius, method_budget, cma_restarts)
    )
    records.extend(
        _run_bayesian_ucb(
            family,
            values,
            radius,
            bo_initial,
            bo_steps,
            acquisition_pool_size,
            ucb_kappa,
        )
    )
    summaries = tuple(
        _summarize_method(family, values, radius, method, records)
        for method in ("sobol", "cma_es", "bayesian_ucb")
    )
    return summaries, tuple(records)


def _physical_metric(form) -> np.ndarray:
    return np.asarray(form.metric_core, dtype=float) / sqrt(form.order.radicand)


def compare_phase10_with_cm(
    blind_summaries: Sequence[Phase10BlindSummary],
    champion_rows: Sequence[dict[str, object]],
    *,
    tolerance: float = 2e-9,
) -> tuple[Phase10CMComparison, ...]:
    """Post-hoc comparison; this is the only function that reads CM data."""

    champions = {
        tuple(int(value) for value in row["polarization_type"]): row
        for row in champion_rows
    }
    grouped: dict[tuple[tuple[int, ...], float], list[Phase10BlindSummary]] = {}
    for summary in blind_summaries:
        grouped.setdefault((summary.polarization_type, summary.radius), []).append(summary)
    comparisons = []
    for (polarization_type, radius), rows in sorted(grouped.items()):
        best = max(rows, key=lambda item: item.best_ell_squared)
        champion = champions[polarization_type]
        form = form_from_population_row(champion)
        cm_result = compute_relative_systole(
            form.alternating,
            _physical_metric(form),
            metric_convention=MetricConvention.POLARIZATION_SCALED,
        )
        cm_spectrum = np.asarray(_kernel_spectrum(cm_result))
        blind_spectrum = np.asarray(best.best_kernel_distance_spectrum)
        if cm_spectrum.shape != blind_spectrum.shape:
            raise ArithmeticError("kernel spectra have incompatible shapes")
        differences = blind_spectrum - cm_spectrum
        cm_ell = float(champion["ell_squared_numeric"])
        scale = max(1.0, abs(cm_ell), abs(best.best_ell_squared))
        comparisons.append(
            Phase10CMComparison(
                protocol_version=PHASE10_PROTOCOL_VERSION,
                polarization_type=polarization_type,
                radius=float(radius),
                best_method=best.method,
                best_blind_ell_squared=best.best_ell_squared,
                cm_candidate_id=str(champion["candidate_id"]),
                cm_discriminant=int(champion["discriminant"]),
                cm_ell_squared=cm_ell,
                cm_logical_image_order=int(champion["logical_image_order"]),
                cm_automorphism_order=int(champion["polarized_automorphism_order"]),
                blind_to_cm_ratio=best.best_ell_squared / cm_ell,
                blind_beats_cm=best.best_ell_squared > cm_ell + tolerance * scale,
                blind_ties_cm=abs(best.best_ell_squared - cm_ell) <= tolerance * scale,
                spectrum_rms_difference=float(np.sqrt(np.mean(differences**2))),
                spectrum_max_difference=float(np.max(np.abs(differences))),
                same_kernel_spectrum_at_tolerance=bool(
                    np.max(np.abs(differences)) <= tolerance * max(1.0, float(np.max(cm_spectrum)))
                ),
            )
        )
    return tuple(comparisons)


def write_phase10_results(
    blind_summaries: Sequence[Phase10BlindSummary],
    evaluations: Sequence[Phase10Evaluation],
    comparisons: Sequence[Phase10CMComparison],
    output_directory: str | Path,
) -> tuple[Path, ...]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary_rows = [item.as_dict() for item in blind_summaries]
    evaluation_rows = [item.as_dict() for item in evaluations]
    comparison_rows = [item.as_dict() for item in comparisons]
    protocol = {
        "protocol_version": PHASE10_PROTOCOL_VERSION,
        "polarization_types": [list(item) for item in PHASE10_POLARIZATION_TYPES],
        "intrinsic_radii": list(PHASE10_RADII),
        "reference": "canonical product metric diag(D,D), selected from D only",
        "chart": "full g(g+1)-dimensional exponential Cartan chart on Sp/U",
        "domain": "closed RMS affine-invariant ball about the canonical product metric",
        "objective_visible_to_optimizers": (
            "relative systole squared ell^2 only, via exhaustive finite-kernel "
            "enumeration and floating-point branch-and-bound CVP"
        ),
        "method_budget": PHASE10_METHOD_BUDGET,
        "methods": {
            "sobol": "scrambled Sobol uniform-in-ball design",
            "cma_es": f"rank-mu CMA-ES with {PHASE10_CMA_RESTARTS} restarts",
            "bayesian_ucb": (
                f"{PHASE10_BO_INITIAL} initial Sobol-ball points + "
                f"{PHASE10_BO_STEPS} fixed-Matern-5/2 GP-UCB queries"
            ),
        },
        "acquisition_pool": PHASE10_ACQUISITION_POOL,
        "ucb_kappa": PHASE10_UCB_KAPPA,
        "seeds": "SHA256(protocol_version|D|radius|component)",
        "blinding": (
            "blind summaries/evaluations are completed without loading CM or gate data; "
            "CM comparisons are generated afterward in a separate function and ledger"
        ),
        "claim_boundary": (
            "bounded deterministic numerical search; no proof of global optimality "
            "and no arithmetic recognition of a floating-point endpoint"
        ),
    }
    files = {
        "summary_json": output / "phase10_blind_search_summary.json",
        "evaluation_json": output / "phase10_blind_search_evaluations.json",
        "comparison_json": output / "phase10_posthoc_cm_comparison.json",
        "summary_csv": output / "phase10_blind_search_summary.csv",
        "comparison_csv": output / "phase10_posthoc_cm_comparison.csv",
        "protocol": output / "phase10_blind_search_protocol.json",
    }
    files["summary_json"].write_text(json.dumps(summary_rows, indent=2) + "\n")
    files["evaluation_json"].write_text(json.dumps(evaluation_rows, indent=2) + "\n")
    files["comparison_json"].write_text(json.dumps(comparison_rows, indent=2) + "\n")
    files["protocol"].write_text(json.dumps(protocol, indent=2) + "\n")
    for key, rows in (("summary_csv", summary_rows), ("comparison_csv", comparison_rows)):
        with files[key].open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        name: json.dumps(value) if isinstance(value, list) else value
                        for name, value in row.items()
                    }
                )
    return tuple(files.values())
