"""Passive-Clifford actions for the existing exact nonuniform CM records."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from gkp_systole import (
    G3_TYPE_112_GAUSSIAN_FORM,
    G3_TYPE_113_EISENSTEIN_FORM,
    G3_TYPE_122_GAUSSIAN_FORM,
    TYPE_15_EXACT_MODEL,
    TYPE_112_EXACT_MODEL,
    Polarization,
)
from gkp_systole.quadratic_hermitian import ImaginaryQuadraticOrder, QuadraticHermitianForm

from .automorphisms import PolarizedAutomorphismProblem, enumerate_polarized_automorphisms
from .finite_symplectic import elementary_prime_symplectic_order
from .hermitian_automorphisms import enumerate_hermitian_cm_automorphisms
from .kernel_action import LogicalActionResult, compute_logical_action


TYPE_13_CM_FORM = QuadraticHermitianForm(
    ImaginaryQuadraticOrder(-24),
    6,
    6,
    3,
    -2,
)


@dataclass(frozen=True)
class ExactCMGateCandidate:
    candidate_id: str
    label: str
    cm_field: str
    provenance: str
    alternating: tuple[tuple[int, ...], ...]
    metric_core: tuple[tuple[object, ...], ...]
    complex_structure_numerator: tuple[tuple[object, ...], ...]
    metric_scale_radicand: int
    ell_squared_exact: str
    hermitian_form: object | None = None

    @property
    def polarization(self) -> Polarization:
        return Polarization(self.alternating)


def _candidate_from_hermitian_form(
    *,
    candidate_id: str,
    label: str,
    cm_field: str,
    provenance: str,
    ell_squared_exact: str,
    metric_scale_radicand: int,
    form: object,
) -> ExactCMGateCandidate:
    structure = getattr(form, "complex_structure_numerator", None)
    if structure is None:
        structure = form.order.complex_structure_numerator
    return ExactCMGateCandidate(
        candidate_id=candidate_id,
        label=label,
        cm_field=cm_field,
        provenance=provenance,
        alternating=tuple(tuple(int(value) for value in row) for row in form.alternating),
        metric_core=tuple(tuple(value for value in row) for row in form.metric_core),
        complex_structure_numerator=tuple(
            tuple(value for value in row) for row in structure
        ),
        metric_scale_radicand=metric_scale_radicand,
        ell_squared_exact=ell_squared_exact,
        hermitian_form=form,
    )


TYPE_13_CM_CANDIDATE = _candidate_from_hermitian_form(
    candidate_id="g2_type_13_delta_24",
    label="bounded type-(1,3) CM surface record",
    cm_field="Q(sqrt(-6)), order discriminant -24",
    provenance="best exact bounded quadratic-Hermitian Phase-6 record",
    ell_squared_exact="4/sqrt(24)",
    metric_scale_radicand=24,
    form=TYPE_13_CM_FORM,
)

TYPE_15_RECONSTRUCTED_CANDIDATE = ExactCMGateCandidate(
    candidate_id="g2_type_15_reconstructed",
    label="reconstructed type-(1,5) CM surface record",
    cm_field="Q(sqrt(-10))",
    provenance="exact reconstruction of the best Phase-7 compatible-metric point",
    alternating=TYPE_15_EXACT_MODEL.alternating,
    metric_core=TYPE_15_EXACT_MODEL.metric_core,
    complex_structure_numerator=TYPE_15_EXACT_MODEL.complex_structure_numerator,
    metric_scale_radicand=10,
    ell_squared_exact="2/sqrt(10)",
)

TYPE_112_RECONSTRUCTED_CANDIDATE = ExactCMGateCandidate(
    candidate_id="g3_type_112_reconstructed",
    label="reconstructed type-(1,1,2) CM threefold record",
    cm_field="Q(sqrt(-3))",
    provenance="exact reconstruction of the best Phase-8 compatible-metric point",
    alternating=TYPE_112_EXACT_MODEL.alternating,
    metric_core=TYPE_112_EXACT_MODEL.metric_core,
    complex_structure_numerator=TYPE_112_EXACT_MODEL.complex_structure_numerator,
    metric_scale_radicand=3,
    ell_squared_exact="2/sqrt(3)",
)

G3_TYPE_112_BOUNDED_CANDIDATE = _candidate_from_hermitian_form(
    candidate_id="g3_type_112_gaussian_bounded",
    label="bounded Gaussian type-(1,1,2) CM benchmark",
    cm_field="Q(i)",
    provenance="exact high-scoring bounded ternary-Hermitian representative",
    ell_squared_exact="2/sqrt(4) = 1",
    metric_scale_radicand=4,
    form=G3_TYPE_112_GAUSSIAN_FORM,
)

G3_TYPE_113_BOUNDED_CANDIDATE = _candidate_from_hermitian_form(
    candidate_id="g3_type_113_eisenstein_bounded",
    label="bounded Eisenstein type-(1,1,3) CM record",
    cm_field="Q(sqrt(-3))",
    provenance="best exact bounded ternary-Hermitian Phase-8 record",
    ell_squared_exact="2/sqrt(3)",
    metric_scale_radicand=3,
    form=G3_TYPE_113_EISENSTEIN_FORM,
)

G3_TYPE_122_BOUNDED_CANDIDATE = _candidate_from_hermitian_form(
    candidate_id="g3_type_122_gaussian_bounded",
    label="bounded Gaussian type-(1,2,2) CM record",
    cm_field="Q(i)",
    provenance="best exact bounded ternary-Hermitian Phase-8 record",
    ell_squared_exact="2/sqrt(4) = 1",
    metric_scale_radicand=4,
    form=G3_TYPE_122_GAUSSIAN_FORM,
)


PHASE3_CM_CANDIDATES = (
    TYPE_13_CM_CANDIDATE,
    TYPE_15_RECONSTRUCTED_CANDIDATE,
    TYPE_112_RECONSTRUCTED_CANDIDATE,
    G3_TYPE_112_BOUNDED_CANDIDATE,
    G3_TYPE_113_BOUNDED_CANDIDATE,
    G3_TYPE_122_BOUNDED_CANDIDATE,
)


@lru_cache(maxsize=None)
def cm_candidate_logical_action(candidate: ExactCMGateCandidate) -> LogicalActionResult:
    if candidate.hermitian_form is not None:
        group = enumerate_hermitian_cm_automorphisms(candidate.hermitian_form)
    else:
        problem = PolarizedAutomorphismProblem(
            polarization=candidate.polarization,
            metric=candidate.metric_core,
            complex_structure=candidate.complex_structure_numerator,
        )
        group = enumerate_polarized_automorphisms(problem)
    return compute_logical_action(group)


def phase3_cm_action_table() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in PHASE3_CM_CANDIDATES:
        action = cm_candidate_logical_action(candidate)
        target_order = elementary_prime_symplectic_order(candidate.polarization.type)
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "label": candidate.label,
                "cm_field": candidate.cm_field,
                "provenance": candidate.provenance,
                "ell_squared_exact": candidate.ell_squared_exact,
                **action.as_dict(),
                "full_symplectic_target_order": target_order,
                "target_coverage": Fraction(action.image_order, target_order),
                "target_saturated": action.image_order == target_order,
            }
        )
    return rows
