"""Phase 9 radial robustness of CM passive logical Clifford actions."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from gkp_systole import KernelGroup, MetricConvention, Polarization, compute_relative_systole

from .adversarial_search import (
    PHASE8_ACQUISITION_POOL,
    PHASE8_BO_STEPS,
    PHASE8_INITIAL_SOBOL,
    PHASE8_RADII,
    PHASE8_SOBOL_HOLDOUT,
    PHASE8_UCB_KAPPA,
    _bayesian_directions,
    compatible_tangent_model,
    fixed_radius_deformation,
    phase8_champion_rows,
    sobol_sphere,
)
from .hermitian_automorphisms import enumerate_hermitian_cm_automorphisms
from .kernel_action import act_on_kernel_element
from .preregistered_controls import form_from_population_row


PHASE9_PROTOCOL_VERSION = "phase9-v1-radial-passive-gate-robustness"
PHASE9_RADII = PHASE8_RADII
PHASE9_TAU = 0.02
PHASE9_EPSILONS = (1e-8, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)
PHASE9_INITIAL_SOBOL = PHASE8_INITIAL_SOBOL
PHASE9_SOBOL_HOLDOUT = PHASE8_SOBOL_HOLDOUT
PHASE9_BO_STEPS = PHASE8_BO_STEPS
PHASE9_ACQUISITION_POOL = PHASE8_ACQUISITION_POOL
PHASE9_UCB_KAPPA = PHASE8_UCB_KAPPA


@dataclass(frozen=True)
class LogicalGateDefectModel:
    alternating: np.ndarray
    baseline_metric: np.ndarray
    automorphisms: np.ndarray
    action_groups: tuple[tuple[int, ...], ...]
    identity_action_index: int
    generic_action_indices: tuple[int, ...]
    baseline_automorphism_order: int

    @property
    def logical_image_order(self) -> int:
        return len(self.action_groups)


@dataclass(frozen=True)
class GateRobustnessMeasurement:
    action_defects: tuple[float, ...]
    retention_score: float
    epsilon_counts: tuple[tuple[float, int], ...]
    enhanced_epsilon_counts: tuple[tuple[float, int], ...]

    def epsilon_count(self, epsilon: float) -> int:
        return dict(self.epsilon_counts)[epsilon]

    def enhanced_epsilon_count(self, epsilon: float) -> int:
        return dict(self.enhanced_epsilon_counts)[epsilon]


@dataclass(frozen=True)
class Phase9Evaluation:
    candidate_id: str
    polarization_type: tuple[int, ...]
    radius: float
    arm: str
    step: int
    direction: tuple[float, ...]
    retention_score: float
    enhanced_mean_defect: float
    enhanced_maximum_defect: float
    exact_retained_logical_actions: int
    exact_retained_enhanced_actions: int
    epsilon_counts: tuple[tuple[float, int], ...]
    enhanced_epsilon_counts: tuple[tuple[float, int], ...]
    action_defects: tuple[float, ...]
    ell_squared: float
    ell_ratio_to_cm: float
    achieved_distance: float
    polarization_residual: float
    log_volume_residual: float

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["polarization_type"] = list(self.polarization_type)
        result["direction"] = list(self.direction)
        result["epsilon_counts"] = {
            format(epsilon, ".12g"): count for epsilon, count in self.epsilon_counts
        }
        result["enhanced_epsilon_counts"] = {
            format(epsilon, ".12g"): count
            for epsilon, count in self.enhanced_epsilon_counts
        }
        result["action_defects"] = list(self.action_defects)
        return result


@dataclass(frozen=True)
class Phase9SearchResult:
    protocol_version: str
    candidate_id: str
    dimension_g: int
    polarization_type: tuple[int, ...]
    discriminant: int
    radius: float
    tangent_dimension: int
    cm_ell_squared: float
    logical_image_order: int
    polarized_automorphism_order: int
    generic_minimal_image_order: int
    tau: float
    sobol_best_retention: float
    bayesian_best_retention: float
    sobol_worst_retention: float
    bayesian_worst_retention: float
    overall_best_retention: float
    overall_worst_retention: float
    best_retention_ell_ratio: float
    worst_retention_ell_ratio: float
    best_retention_exact_action_count: int
    worst_retention_exact_action_count: int
    best_retention_exact_enhanced_action_count: int
    worst_retention_exact_enhanced_action_count: int
    best_retention_epsilon_counts: tuple[tuple[float, int], ...]
    worst_retention_epsilon_counts: tuple[tuple[float, int], ...]
    best_retention_enhanced_epsilon_counts: tuple[tuple[float, int], ...]
    worst_retention_enhanced_epsilon_counts: tuple[tuple[float, int], ...]
    best_retention_arm: str
    worst_retention_arm: str
    best_retention_direction: tuple[float, ...]
    worst_retention_direction: tuple[float, ...]
    maximum_distance_error: float
    maximum_polarization_residual: float
    maximum_log_volume_residual: float

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["polarization_type"] = list(self.polarization_type)
        result["best_retention_direction"] = list(self.best_retention_direction)
        result["worst_retention_direction"] = list(self.worst_retention_direction)
        result["best_retention_epsilon_counts"] = {
            format(epsilon, ".12g"): count
            for epsilon, count in self.best_retention_epsilon_counts
        }
        result["worst_retention_epsilon_counts"] = {
            format(epsilon, ".12g"): count
            for epsilon, count in self.worst_retention_epsilon_counts
        }
        result["best_retention_enhanced_epsilon_counts"] = {
            format(epsilon, ".12g"): count
            for epsilon, count in self.best_retention_enhanced_epsilon_counts
        }
        result["worst_retention_enhanced_epsilon_counts"] = {
            format(epsilon, ".12g"): count
            for epsilon, count in self.worst_retention_enhanced_epsilon_counts
        }
        return result


def _physical_metric(form) -> np.ndarray:
    return np.asarray(form.metric_core, dtype=float) / np.sqrt(form.order.radicand)


def logical_gate_defect_model(form) -> LogicalGateDefectModel:
    """Group exact CM automorphisms by their induced logical action."""

    group = enumerate_hermitian_cm_automorphisms(form)
    kernel = KernelGroup.from_polarization(group.problem.polarization)
    generators = kernel.generators
    identity_signature = tuple(generators)
    minus_identity = tuple(
        tuple(-int(row == column) for column in range(len(form.alternating)))
        for row in range(len(form.alternating))
    )
    minus_identity_signature = tuple(
        act_on_kernel_element(minus_identity, generator) for generator in generators
    )
    grouped: dict[tuple[object, ...], list[int]] = {}
    matrices = np.asarray(group.elements, dtype=float)
    for index, matrix in enumerate(group.elements):
        signature = tuple(
            act_on_kernel_element(matrix, generator) for generator in generators
        )
        grouped.setdefault(signature, []).append(index)
    signatures = sorted(grouped, key=repr)
    identity_index = signatures.index(identity_signature)
    generic_indices = tuple(
        sorted(
            {
                signatures.index(identity_signature),
                signatures.index(minus_identity_signature),
            }
        )
    )
    action_groups = tuple(tuple(grouped[signature]) for signature in signatures)
    return LogicalGateDefectModel(
        alternating=np.asarray(form.alternating, dtype=float),
        baseline_metric=_physical_metric(form),
        automorphisms=matrices,
        action_groups=action_groups,
        identity_action_index=identity_index,
        generic_action_indices=generic_indices,
        baseline_automorphism_order=group.order,
    )


def automorphism_metric_defects(
    model: LogicalGateDefectModel,
    metric: Sequence[Sequence[float]],
) -> np.ndarray:
    """Compute coordinate-invariant RMS metric defects for all automorphisms."""

    g = np.asarray(metric, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(g)
    if float(np.min(eigenvalues)) <= 0:
        raise ValueError("metric must be positive definite")
    inverse_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    transformations = model.automorphisms
    congruences = np.transpose(transformations, (0, 2, 1)) @ (g @ transformations)
    differences = congruences - g
    whitened = inverse_sqrt @ differences @ inverse_sqrt
    return np.linalg.norm(whitened, axis=(1, 2)) / np.sqrt(g.shape[0])


def measure_gate_robustness(
    model: LogicalGateDefectModel,
    metric: Sequence[Sequence[float]],
    *,
    tau: float = PHASE9_TAU,
    epsilons: Sequence[float] = PHASE9_EPSILONS,
) -> GateRobustnessMeasurement:
    """Minimize defects over representatives of every logical action."""

    if tau <= 0 or any(epsilon < 0 for epsilon in epsilons):
        raise ValueError("tau must be positive and epsilons nonnegative")
    automorphism_defects = automorphism_metric_defects(model, metric)
    action_defects = np.asarray(
        [min(automorphism_defects[list(indices)]) for indices in model.action_groups],
        dtype=float,
    )
    enhanced_indices = tuple(
        index
        for index in range(model.logical_image_order)
        if index not in model.generic_action_indices
    )
    if not enhanced_indices:
        raise ValueError("gate-retention score requires CM-enhanced logical actions")
    enhanced = action_defects[list(enhanced_indices)]
    retention = float(np.mean(np.exp(-((enhanced / tau) ** 2))))
    epsilon_counts = tuple(
        (float(epsilon), int(np.sum(action_defects <= float(epsilon))))
        for epsilon in epsilons
    )
    enhanced_epsilon_counts = tuple(
        (float(epsilon), int(np.sum(enhanced <= float(epsilon))))
        for epsilon in epsilons
    )
    return GateRobustnessMeasurement(
        action_defects=tuple(float(value) for value in action_defects),
        retention_score=retention,
        epsilon_counts=epsilon_counts,
        enhanced_epsilon_counts=enhanced_epsilon_counts,
    )


def phase9_seed(candidate_id: str, radius: float, component: str) -> int:
    digest = hashlib.sha256(
        f"{PHASE9_PROTOCOL_VERSION}|{candidate_id}|{radius:.12g}|{component}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def evaluate_phase9_search(
    population_row: dict[str, object],
    radius: float,
    *,
    initial_sobol: int = PHASE9_INITIAL_SOBOL,
    sobol_holdout: int = PHASE9_SOBOL_HOLDOUT,
    bayesian_steps: int = PHASE9_BO_STEPS,
    acquisition_pool_size: int = PHASE9_ACQUISITION_POOL,
    ucb_kappa: float = PHASE9_UCB_KAPPA,
) -> tuple[Phase9SearchResult, tuple[Phase9Evaluation, ...]]:
    form = form_from_population_row(population_row)
    metric = _physical_metric(form)
    tangent_model = compatible_tangent_model(form.alternating, metric)
    gate_model = logical_gate_defect_model(form)
    baseline_measurement = measure_gate_robustness(gate_model, metric)
    if max(baseline_measurement.action_defects) > 2e-10:
        raise ArithmeticError("CM automorphisms do not preserve the baseline metric")
    if gate_model.logical_image_order != int(population_row["logical_image_order"]):
        raise ArithmeticError("logical image disagrees with Phase-5 ledger")
    polarization = Polarization(form.alternating)
    cm_ell = float(population_row["ell_squared_numeric"])
    candidate_id = str(population_row["candidate_id"])
    polarization_type = tuple(int(value) for value in population_row["polarization_type"])

    shared = sobol_sphere(
        tangent_model.tangent_dimension,
        initial_sobol,
        phase9_seed(candidate_id, radius, "shared-sobol"),
    )
    holdout = sobol_sphere(
        tangent_model.tangent_dimension,
        sobol_holdout,
        phase9_seed(candidate_id, radius, "sobol-holdout"),
    )
    best_pool = sobol_sphere(
        tangent_model.tangent_dimension,
        acquisition_pool_size,
        phase9_seed(candidate_id, radius, "best-retention-pool"),
    )
    worst_pool = sobol_sphere(
        tangent_model.tangent_dimension,
        acquisition_pool_size,
        phase9_seed(candidate_id, radius, "worst-retention-pool"),
    )
    evaluations: list[Phase9Evaluation] = []

    def evaluate_direction(direction: np.ndarray, arm: str, step: int) -> float:
        deformation = fixed_radius_deformation(tangent_model, direction, radius)
        measurement = measure_gate_robustness(gate_model, deformation.metric)
        ell = compute_relative_systole(
            polarization,
            deformation.metric,
            metric_convention=MetricConvention.POLARIZATION_SCALED,
        )
        defects = np.asarray(measurement.action_defects)
        enhanced_indices = tuple(
            index
            for index in range(gate_model.logical_image_order)
            if index not in gate_model.generic_action_indices
        )
        enhanced = defects[list(enhanced_indices)]
        evaluations.append(
            Phase9Evaluation(
                candidate_id=candidate_id,
                polarization_type=polarization_type,
                radius=float(radius),
                arm=arm,
                step=step,
                direction=deformation.direction,
                retention_score=measurement.retention_score,
                enhanced_mean_defect=float(np.mean(enhanced)),
                enhanced_maximum_defect=float(np.max(enhanced)),
                exact_retained_logical_actions=measurement.epsilon_count(1e-8),
                exact_retained_enhanced_actions=measurement.enhanced_epsilon_count(1e-8),
                epsilon_counts=measurement.epsilon_counts,
                enhanced_epsilon_counts=measurement.enhanced_epsilon_counts,
                action_defects=measurement.action_defects,
                ell_squared=float(ell.squared_systole),
                ell_ratio_to_cm=float(ell.squared_systole) / cm_ell,
                achieved_distance=deformation.achieved_distance,
                polarization_residual=deformation.polarization_residual,
                log_volume_residual=deformation.log_volume_residual,
            )
        )
        return measurement.retention_score

    shared_values = np.asarray(
        [
            evaluate_direction(direction, "shared_sobol", index)
            for index, direction in enumerate(shared)
        ]
    )
    for index, direction in enumerate(holdout):
        evaluate_direction(direction, "sobol_holdout", index)

    _bayesian_directions(
        shared,
        shared_values,
        best_pool,
        lambda direction: evaluate_direction(
            direction,
            "bayesian_best_retention",
            sum(record.arm == "bayesian_best_retention" for record in evaluations),
        ),
        bayesian_steps,
        ucb_kappa,
    )
    _bayesian_directions(
        shared,
        -shared_values,
        worst_pool,
        lambda direction: -evaluate_direction(
            direction,
            "bayesian_worst_retention",
            sum(record.arm == "bayesian_worst_retention" for record in evaluations),
        ),
        bayesian_steps,
        ucb_kappa,
    )

    sobol_records = [
        record for record in evaluations if record.arm in {"shared_sobol", "sobol_holdout"}
    ]
    best_records = [
        record
        for record in evaluations
        if record.arm in {"shared_sobol", "bayesian_best_retention"}
    ]
    worst_records = [
        record
        for record in evaluations
        if record.arm in {"shared_sobol", "bayesian_worst_retention"}
    ]
    sobol_best = max(sobol_records, key=lambda record: record.retention_score)
    bayesian_best = max(best_records, key=lambda record: record.retention_score)
    sobol_worst = min(sobol_records, key=lambda record: record.retention_score)
    bayesian_worst = min(worst_records, key=lambda record: record.retention_score)
    overall_best = max(evaluations, key=lambda record: record.retention_score)
    overall_worst = min(evaluations, key=lambda record: record.retention_score)
    summary = Phase9SearchResult(
        protocol_version=PHASE9_PROTOCOL_VERSION,
        candidate_id=candidate_id,
        dimension_g=int(population_row["dimension_g"]),
        polarization_type=polarization_type,
        discriminant=int(population_row["discriminant"]),
        radius=float(radius),
        tangent_dimension=tangent_model.tangent_dimension,
        cm_ell_squared=cm_ell,
        logical_image_order=gate_model.logical_image_order,
        polarized_automorphism_order=gate_model.baseline_automorphism_order,
        generic_minimal_image_order=int(population_row["generic_minimal_image_order"]),
        tau=PHASE9_TAU,
        sobol_best_retention=sobol_best.retention_score,
        bayesian_best_retention=bayesian_best.retention_score,
        sobol_worst_retention=sobol_worst.retention_score,
        bayesian_worst_retention=bayesian_worst.retention_score,
        overall_best_retention=overall_best.retention_score,
        overall_worst_retention=overall_worst.retention_score,
        best_retention_ell_ratio=overall_best.ell_ratio_to_cm,
        worst_retention_ell_ratio=overall_worst.ell_ratio_to_cm,
        best_retention_exact_action_count=overall_best.exact_retained_logical_actions,
        worst_retention_exact_action_count=overall_worst.exact_retained_logical_actions,
        best_retention_exact_enhanced_action_count=overall_best.exact_retained_enhanced_actions,
        worst_retention_exact_enhanced_action_count=overall_worst.exact_retained_enhanced_actions,
        best_retention_epsilon_counts=overall_best.epsilon_counts,
        worst_retention_epsilon_counts=overall_worst.epsilon_counts,
        best_retention_enhanced_epsilon_counts=overall_best.enhanced_epsilon_counts,
        worst_retention_enhanced_epsilon_counts=overall_worst.enhanced_epsilon_counts,
        best_retention_arm=overall_best.arm,
        worst_retention_arm=overall_worst.arm,
        best_retention_direction=overall_best.direction,
        worst_retention_direction=overall_worst.direction,
        maximum_distance_error=max(abs(record.achieved_distance - radius) for record in evaluations),
        maximum_polarization_residual=max(record.polarization_residual for record in evaluations),
        maximum_log_volume_residual=max(record.log_volume_residual for record in evaluations),
    )
    return summary, tuple(evaluations)


def write_phase9_results(
    summaries: Sequence[Phase9SearchResult],
    evaluations: Sequence[Phase9Evaluation],
    output_directory: str | Path,
) -> tuple[Path, Path, Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary_rows = [summary.as_dict() for summary in summaries]
    evaluation_rows = [evaluation.as_dict() for evaluation in evaluations]
    protocol = {
        "protocol_version": PHASE9_PROTOCOL_VERSION,
        "candidate_selection": (
            "same five within-type CM champions as Phase 8; ell^2 ties prefer larger "
            "logical image then lexicographically earliest stable ID"
        ),
        "radii": list(PHASE9_RADII),
        "tau": PHASE9_TAU,
        "epsilons": list(PHASE9_EPSILONS),
        "logical_action_defect": (
            "minimum over CM automorphism representatives of "
            "||G^-1/2(U^T G U-G)G^-1/2||_F/sqrt(2g)"
        ),
        "generic_action_image": "logical actions induced by {+I,-I}",
        "retention_score": (
            "mean exp(-(defect/tau)^2) over CM-enhanced logical actions, "
            "excluding the complete generic {+I,-I} image"
        ),
        "sobol_budget": PHASE9_INITIAL_SOBOL + PHASE9_SOBOL_HOLDOUT,
        "best_retention_bayesian_budget": PHASE9_INITIAL_SOBOL + PHASE9_BO_STEPS,
        "worst_retention_bayesian_budget": PHASE9_INITIAL_SOBOL + PHASE9_BO_STEPS,
        "surrogate": "fixed Matern-5/2 GP with UCB mean+2*std",
        "seeds": "SHA256(protocol_version|candidate_id|radius|component)",
        "claim_boundary": (
            "tracks approximate retention of logical actions present at the CM baseline; "
            "does not enumerate new automorphisms of every deformed metric"
        ),
    }
    summary_path = output / "phase9_gate_robustness_summary.json"
    evaluation_path = output / "phase9_gate_robustness_evaluations.json"
    csv_path = output / "phase9_gate_robustness_summary.csv"
    protocol_path = output / "phase9_gate_robustness_protocol.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2) + "\n")
    evaluation_path.write_text(json.dumps(evaluation_rows, indent=2) + "\n")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )
    return summary_path, evaluation_path, csv_path, protocol_path


def phase9_champion_rows(population_rows: Sequence[dict[str, object]]):
    return phase8_champion_rows(population_rows)
