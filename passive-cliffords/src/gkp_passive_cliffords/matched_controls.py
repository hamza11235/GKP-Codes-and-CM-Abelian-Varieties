"""Matched generic-real controls for the exact nonuniform CM records.

The controls keep the polarization matrix (and hence its type ``D``) fixed
and deform only the compatible metric by small real symplectic
transformations.  The deformation parameters are rational multiples of
``pi``.  They therefore leave the rational orbit used by the CM
construction.  A generic point of this family is non-CM, but this module
does not certify the endomorphism ring of each individual control.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import sqrt

import numpy as np

from gkp_systole import scan_pi_symplectic_deformations
from gkp_systole.deformation_scan import PiDeformationSample

from .cm_records import (
    PHASE3_CM_CANDIDATES,
    ExactCMGateCandidate,
    cm_candidate_logical_action,
)
from .finite_symplectic import elementary_prime_symplectic_order
from .kernel_action import compute_logical_action
from .numerical_automorphisms import (
    NumericalPolarizedAutomorphismProblem,
    enumerate_numerical_polarized_automorphisms,
)


DEFAULT_PHASE4_SEEDS = {
    candidate.candidate_id: 4100 + index
    for index, candidate in enumerate(PHASE3_CM_CANDIDATES)
}


def scaled_metric(candidate: ExactCMGateCandidate) -> tuple[tuple[float, ...], ...]:
    """Return the polarization-scaled metric used by the systole engine."""

    scale = sqrt(candidate.metric_scale_radicand)
    return tuple(
        tuple(float(value) / scale for value in row)
        for row in candidate.metric_core
    )


@dataclass(frozen=True)
class MatchedGenericControl:
    candidate_id: str
    dimension_g: int
    polarization_type: tuple[int, ...]
    control_index: int
    parameters: tuple[tuple[tuple[int, ...], Fraction], ...]
    metric: tuple[tuple[float, ...], ...]
    ell_squared: float
    class_multiplicity: int
    lift_multiplicity: int
    polarized_automorphism_order: int
    logical_image_order: int
    action_kernel_order: int
    full_symplectic_target_order: int
    maximum_metric_residual: float
    relative_metric_displacement: float
    cm_status: str = "generic-real control; non-CM almost surely, not individually certified"

    @property
    def target_coverage(self) -> Fraction:
        return Fraction(self.logical_image_order, self.full_symplectic_target_order)

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "dimension_g": self.dimension_g,
            "polarization_type": self.polarization_type,
            "control_index": self.control_index,
            "control_status": self.cm_status,
            "ell_squared": self.ell_squared,
            "class_multiplicity": self.class_multiplicity,
            "lift_multiplicity": self.lift_multiplicity,
            "polarized_automorphism_order": self.polarized_automorphism_order,
            "logical_image_order": self.logical_image_order,
            "action_kernel_order": self.action_kernel_order,
            "full_symplectic_target_order": self.full_symplectic_target_order,
            "target_coverage": self.target_coverage,
            "maximum_metric_residual": self.maximum_metric_residual,
            "relative_metric_displacement": self.relative_metric_displacement,
        }


@dataclass(frozen=True)
class MatchedControlScan:
    candidate: ExactCMGateCandidate
    seed: int
    amplitude: float
    steps: int
    baseline_ell_squared: float
    controls: tuple[MatchedGenericControl, ...]

    @property
    def cm_automorphism_order(self) -> int:
        return cm_candidate_logical_action(self.candidate).automorphism_order

    @property
    def cm_logical_image_order(self) -> int:
        return cm_candidate_logical_action(self.candidate).image_order

    @property
    def largest_control_automorphism_order(self) -> int:
        return max(control.polarized_automorphism_order for control in self.controls)

    @property
    def largest_control_image_order(self) -> int:
        return max(control.logical_image_order for control in self.controls)

    @property
    def automorphism_enhancement(self) -> Fraction:
        return Fraction(self.cm_automorphism_order, self.largest_control_automorphism_order)

    @property
    def logical_image_enhancement(self) -> Fraction:
        return Fraction(self.cm_logical_image_order, self.largest_control_image_order)


def _evaluate_control(
    candidate: ExactCMGateCandidate,
    baseline_metric: tuple[tuple[float, ...], ...],
    sample: PiDeformationSample,
) -> MatchedGenericControl:
    problem = NumericalPolarizedAutomorphismProblem(
        polarization=candidate.polarization,
        metric=sample.metric,
        tolerance=1e-8,
    )
    group = enumerate_numerical_polarized_automorphisms(problem)
    action = compute_logical_action(group)
    baseline = np.asarray(baseline_metric, dtype=float)
    deformed = np.asarray(sample.metric, dtype=float)
    displacement = float(np.linalg.norm(deformed - baseline) / np.linalg.norm(baseline))
    target_order = elementary_prime_symplectic_order(candidate.polarization.type)
    return MatchedGenericControl(
        candidate_id=candidate.candidate_id,
        dimension_g=candidate.polarization.dimension,
        polarization_type=candidate.polarization.type,
        control_index=sample.index,
        parameters=sample.parameters,
        metric=sample.metric,
        ell_squared=float(sample.systole_result.squared_systole),
        class_multiplicity=sample.systole_result.class_multiplicity,
        lift_multiplicity=sample.systole_result.lift_multiplicity,
        polarized_automorphism_order=action.automorphism_order,
        logical_image_order=action.image_order,
        action_kernel_order=action.action_kernel_order,
        full_symplectic_target_order=target_order,
        maximum_metric_residual=group.maximum_metric_residual,
        relative_metric_displacement=displacement,
    )


@lru_cache(maxsize=None)
def matched_control_scan(
    candidate: ExactCMGateCandidate,
    *,
    sample_count: int = 3,
    amplitude: float = 5e-4,
    steps: int = 4,
    vector_bound: int = 2,
) -> MatchedControlScan:
    """Construct and evaluate deterministic local controls for one CM record."""

    baseline_metric = scaled_metric(candidate)
    seed = DEFAULT_PHASE4_SEEDS[candidate.candidate_id]
    deformation_scan = scan_pi_symplectic_deformations(
        candidate.alternating,
        baseline_metric,
        sample_count=sample_count,
        seed=seed,
        amplitude=amplitude,
        steps=steps,
        vector_bound=vector_bound,
    )
    controls = tuple(
        _evaluate_control(candidate, baseline_metric, sample)
        for sample in deformation_scan.samples
    )
    return MatchedControlScan(
        candidate=candidate,
        seed=seed,
        amplitude=amplitude,
        steps=steps,
        baseline_ell_squared=float(deformation_scan.baseline_result.squared_systole),
        controls=controls,
    )


def phase4_matched_control_scans(
    *,
    sample_count: int = 3,
    amplitude: float = 5e-4,
) -> tuple[MatchedControlScan, ...]:
    return tuple(
        matched_control_scan(
            candidate,
            sample_count=sample_count,
            amplitude=amplitude,
        )
        for candidate in PHASE3_CM_CANDIDATES
    )


def phase4_control_table(
    *,
    sample_count: int = 3,
    amplitude: float = 5e-4,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scan in phase4_matched_control_scans(
        sample_count=sample_count,
        amplitude=amplitude,
    ):
        for control in scan.controls:
            rows.append(control.as_dict())
    return rows


def phase4_comparison_table(
    *,
    sample_count: int = 3,
    amplitude: float = 5e-4,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scan in phase4_matched_control_scans(
        sample_count=sample_count,
        amplitude=amplitude,
    ):
        cm_action = cm_candidate_logical_action(scan.candidate)
        target_order = elementary_prime_symplectic_order(scan.candidate.polarization.type)
        rows.append(
            {
                "candidate_id": scan.candidate.candidate_id,
                "dimension_g": scan.candidate.polarization.dimension,
                "polarization_type": scan.candidate.polarization.type,
                "cm_field": scan.candidate.cm_field,
                "cm_ell_squared": scan.baseline_ell_squared,
                "control_ell_squared_min": min(c.ell_squared for c in scan.controls),
                "control_ell_squared_max": max(c.ell_squared for c in scan.controls),
                "cm_automorphism_order": cm_action.automorphism_order,
                "control_automorphism_orders": tuple(
                    c.polarized_automorphism_order for c in scan.controls
                ),
                "cm_logical_image_order": cm_action.image_order,
                "control_logical_image_orders": tuple(
                    c.logical_image_order for c in scan.controls
                ),
                "full_symplectic_target_order": target_order,
                "cm_target_coverage": Fraction(cm_action.image_order, target_order),
                "automorphism_enhancement": scan.automorphism_enhancement,
                "logical_image_enhancement": scan.logical_image_enhancement,
                "control_status": scan.controls[0].cm_status,
            }
        )
    return rows
