"""Exact reconstruction of the Phase-8 type-``(1,1,2)`` record.

The full 12-dimensional numerical search converges near a half-integral metric
``G_core/sqrt(3)``.  This module stores that recognition exactly, certifies the
polarized abelian threefold and its relative systole, and supplies a rational
isogeny certificate showing that the reconstructed point is CM.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import sqrt
from typing import Sequence

from .conventions import MetricConvention
from .kernel import invert_rational_matrix
from .polarization import Polarization, determinant
from .ppav import PPAVValidationResult, validate_polarized_abelian_data
from .systole import RelativeSystoleResult, compute_relative_systole
from .type15 import rational_commutant_dimension


RationalMatrix = tuple[tuple[Fraction, ...], ...]


TYPE_112_ALTERNATING = (
    (0, 0, 0, 1, 0, 0),
    (0, 0, -1, 0, 2, -1),
    (0, 1, 0, 0, -1, 2),
    (-1, 0, 0, 0, 0, 0),
    (0, -2, 1, 0, 0, -1),
    (0, 1, -2, 0, 1, 0),
)

TYPE_112_METRIC_CORE = (
    (12, -12, 7, 4, 6, -11),
    (-12, 20, -5, -4, -10, 15),
    (7, -5, 9, Fraction(1, 2), 0, -4),
    (4, -4, Fraction(1, 2), 3, 4, Fraction(-11, 2)),
    (6, -10, 0, 4, 8, -10),
    (-11, 15, -4, Fraction(-11, 2), -10, 15),
)

TYPE_112_COMPLEX_STRUCTURE_NUMERATOR = (
    (4, -4, Fraction(1, 2), 3, 4, Fraction(-11, 2)),
    (4, -5, Fraction(5, 2), Fraction(3, 2), 3, Fraction(-9, 2)),
    (-2, 0, Fraction(-3, 2), Fraction(-3, 2), -1, Fraction(5, 2)),
    (-12, 12, -7, -4, -6, 11),
    (3, -10, Fraction(-3, 2), 1, 5, Fraction(-11, 2)),
    (-4, 0, Fraction(-13, 2), Fraction(-1, 2), 1, Fraction(3, 2)),
)

TYPE_112_CM_ISOGENY_CHANGE = (
    (1, 4, 0, -4, 0, 4),
    (0, 4, 1, -5, 0, 3),
    (0, -2, 0, 0, 0, -1),
    (0, -12, 0, 12, 0, -6),
    (0, 3, 0, -10, 1, 5),
    (0, -4, 0, 0, 0, 1),
)

TYPE_112_CM_BLOCK = (
    (0, -3, 0, 0, 0, 0),
    (1, 0, 0, 0, 0, 0),
    (0, 0, 0, -3, 0, 0),
    (0, 0, 1, 0, 0, 0),
    (0, 0, 0, 0, 0, -3),
    (0, 0, 0, 0, 1, 0),
)


def _fraction_matrix(matrix) -> RationalMatrix:
    return tuple(tuple(Fraction(value) for value in row) for row in matrix)


def _multiply(left, right) -> RationalMatrix:
    a = _fraction_matrix(left)
    b = _fraction_matrix(right)
    return tuple(
        tuple(
            sum((a[row][inner] * b[inner][column] for inner in range(len(b))), Fraction(0))
            for column in range(len(b[0]))
        )
        for row in range(len(a))
    )


def reconstruct_type112_metric_core(
    numerical_metric: Sequence[Sequence[float]],
    *,
    tolerance: float = 1e-3,
) -> RationalMatrix:
    """Recognize the half-integral core after multiplying by ``sqrt(3)``.

    This is an algebraic-recognition step, not the final certificate.  The
    returned matrix must subsequently pass the exact PPAV and CVP checks in
    :class:`Type112ExactModel`.
    """

    if len(numerical_metric) != 6 or any(len(row) != 6 for row in numerical_metric):
        raise ValueError("type-(1,1,2) reconstruction requires a 6 by 6 metric")
    reconstructed = tuple(
        tuple(Fraction(float(value) * sqrt(3)).limit_denominator(2) for value in row)
        for row in numerical_metric
    )
    maximum_error = max(
        abs(float(reconstructed[row][column]) / sqrt(3) - float(numerical_metric[row][column]))
        for row in range(6)
        for column in range(6)
    )
    if maximum_error > tolerance:
        raise ArithmeticError(
            f"numerical metric is not within the reconstruction tolerance: {maximum_error}"
        )
    return reconstructed


@dataclass(frozen=True)
class Type112CMCertificate:
    field: str
    order_discriminant: int
    elliptic_period: str
    commutant_dimension: int
    rational_isogeny_degree: int
    block_matrix: tuple[tuple[int, ...], ...]
    is_cm: bool


@dataclass(frozen=True)
class Type112ExactModel:
    """Exact model ``G=G_core/sqrt(3)``, ``J=S/sqrt(3)``."""

    alternating: tuple[tuple[int, ...], ...] = TYPE_112_ALTERNATING
    metric_core: RationalMatrix = _fraction_matrix(TYPE_112_METRIC_CORE)
    complex_structure_numerator: RationalMatrix = _fraction_matrix(
        TYPE_112_COMPLEX_STRUCTURE_NUMERATOR
    )
    scale_radicand: int = 3

    @property
    def polarization(self) -> Polarization:
        return Polarization(self.alternating)

    @property
    def squared_systole(self) -> float:
        return 2.0 / sqrt(3.0)

    @property
    def exact_squared_systole(self) -> str:
        return "2/sqrt(3)"

    @property
    def metric_numeric(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(float(value) / sqrt(self.scale_radicand) for value in row)
            for row in self.metric_core
        )

    def validation_certificate(self) -> PPAVValidationResult:
        return validate_polarized_abelian_data(
            self.metric_core,
            self.complex_structure_numerator,
            self.alternating,
            scale_radicand=self.scale_radicand,
            expected_type=(1, 1, 2),
        )

    def core_relative_systole(self) -> RelativeSystoleResult:
        self.validation_certificate()
        return compute_relative_systole(
            self.alternating,
            self.metric_core,
            metric_convention=MetricConvention.POLARIZATION_SCALED,
        )

    def cm_certificate(self) -> Type112CMCertificate:
        structure = self.complex_structure_numerator
        squared = _multiply(structure, structure)
        expected = tuple(
            tuple(Fraction(-3 if row == column else 0) for column in range(6))
            for row in range(6)
        )
        if squared != expected:
            raise ArithmeticError("rational CM endomorphism does not square to -3")
        change = _fraction_matrix(TYPE_112_CM_ISOGENY_CHANGE)
        conjugated = _multiply(
            _multiply(invert_rational_matrix(change), structure),
            change,
        )
        if conjugated != _fraction_matrix(TYPE_112_CM_BLOCK):
            raise ArithmeticError("CM isogeny did not split the rational representation")
        commutant = rational_commutant_dimension(structure)
        degree = abs(determinant(TYPE_112_CM_ISOGENY_CHANGE))
        return Type112CMCertificate(
            field="Q(sqrt(-3))",
            order_discriminant=-12,
            elliptic_period="i*sqrt(3)",
            commutant_dimension=commutant,
            rational_isogeny_degree=degree,
            block_matrix=TYPE_112_CM_BLOCK,
            is_cm=commutant == 18 and degree == 72,
        )


TYPE_112_EXACT_MODEL = Type112ExactModel()
