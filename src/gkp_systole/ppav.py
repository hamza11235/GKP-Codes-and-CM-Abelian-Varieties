"""Exact Riemann-compatibility certificates for polarized complex tori.

The repository frequently stores algebraic data in the scaled form

``G = G_core / sqrt(r)`` and ``J = J_num / sqrt(r)``.

In that representation all compatibility conditions can be checked over the
rationals:

* ``J_num^2 = -r I``;
* ``J_num^T G_core J_num = r G_core``;
* ``A = G_core J_num / r`` is integral and alternating; and
* ``det(G_core) / r^g = |det(A)|``.

The last equality is the covolume identity for a compatible polarization.
For a principal polarization both sides equal one.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Rational
from typing import Sequence

from .metric import Metric
from .polarization import Polarization


RationalMatrix = tuple[tuple[Fraction, ...], ...]
IntMatrix = tuple[tuple[int, ...], ...]


class PPAVValidationError(ValueError):
    """Raised when exact lattice data fail a Riemann compatibility check."""


def _exact_matrix(
    matrix: Sequence[Sequence[int | Fraction]],
    *,
    name: str,
) -> RationalMatrix:
    rows = tuple(tuple(row) for row in matrix)
    if not rows:
        raise PPAVValidationError(f"{name} must be nonempty")
    size = len(rows)
    if any(len(row) != size for row in rows):
        raise PPAVValidationError(f"{name} must be square")
    if any(
        not isinstance(value, (Integral, Rational))
        for row in rows
        for value in row
    ):
        raise PPAVValidationError(
            f"{name} must use exact integer or rational entries; floats are not certified"
        )
    return tuple(
        tuple(
            Fraction(int(value))
            if isinstance(value, Integral)
            else Fraction(value.numerator, value.denominator)
            for value in row
        )
        for row in rows
    )


def _transpose(matrix: RationalMatrix) -> RationalMatrix:
    return tuple(
        tuple(matrix[row][column] for row in range(len(matrix)))
        for column in range(len(matrix))
    )


def _multiply(left: RationalMatrix, right: RationalMatrix) -> RationalMatrix:
    return tuple(
        tuple(
            sum(
                (
                    left[row][inner] * right[inner][column]
                    for inner in range(len(right))
                ),
                Fraction(0),
            )
            for column in range(len(right))
        )
        for row in range(len(left))
    )


def _scale(matrix: RationalMatrix, scalar: Fraction) -> RationalMatrix:
    return tuple(tuple(scalar * value for value in row) for row in matrix)


def _identity(size: int, diagonal: Fraction = Fraction(1)) -> RationalMatrix:
    return tuple(
        tuple(diagonal if row == column else Fraction(0) for column in range(size))
        for row in range(size)
    )


def _integral_matrix(matrix: RationalMatrix, *, name: str) -> IntMatrix:
    if any(value.denominator != 1 for row in matrix for value in row):
        raise PPAVValidationError(f"{name} is not integral")
    return tuple(tuple(value.numerator for value in row) for row in matrix)


@dataclass(frozen=True)
class PPAVValidationResult:
    """A successful exact compatibility certificate.

    Despite the historical ``PPAV`` name, ``principal`` may be false when the
    general :func:`validate_polarized_abelian_data` entry point is used.
    """

    metric_core: RationalMatrix
    complex_structure_numerator: RationalMatrix
    scale_radicand: int
    polarization: Polarization
    checks: tuple[str, ...]

    @property
    def dimension(self) -> int:
        return self.polarization.dimension

    @property
    def polarization_type(self) -> tuple[int, ...]:
        return self.polarization.type

    @property
    def principal(self) -> bool:
        return self.polarization_type == (1,) * self.dimension

    @property
    def metric_core_determinant(self) -> Fraction:
        value = Metric(self.metric_core).determinant
        if not isinstance(value, Fraction):
            raise ArithmeticError("an exact certificate produced an inexact determinant")
        return value

    @property
    def physical_metric_determinant(self) -> Fraction:
        return self.metric_core_determinant / self.scale_radicand**self.dimension

    @property
    def certified(self) -> bool:
        return True

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension_g": self.dimension,
            "polarization_type": self.polarization_type,
            "principal": self.principal,
            "scale_radicand": self.scale_radicand,
            "metric_core_determinant": self.metric_core_determinant,
            "physical_metric_determinant": self.physical_metric_determinant,
            "polarization_determinant": self.polarization.determinant,
            "certified": self.certified,
            "checks": self.checks,
        }


def validate_polarized_abelian_data(
    metric_core: Sequence[Sequence[int | Fraction]] | Metric,
    complex_structure_numerator: Sequence[Sequence[int | Fraction]],
    alternating: Sequence[Sequence[int]] | None = None,
    *,
    scale_radicand: int = 1,
    expected_type: Sequence[int] | None = None,
) -> PPAVValidationResult:
    """Certify exact polarized-abelian compatibility.

    Args:
        metric_core: Exact positive-definite ``G_core``.
        complex_structure_numerator: Exact ``J_num``.
        alternating: Optional claimed integral Riemann form.  When omitted it
            is derived as ``G_core J_num / r``.
        scale_radicand: Positive integer ``r`` in the scaled representation.
        expected_type: Optional polarization type to require.
    """

    if not isinstance(scale_radicand, Integral) or scale_radicand <= 0:
        raise PPAVValidationError("scale_radicand must be a positive integer")
    radicand = int(scale_radicand)

    metric = metric_core if isinstance(metric_core, Metric) else Metric(metric_core)
    if not metric.is_exact:
        raise PPAVValidationError(
            "metric_core must use exact integer or rational entries; floats are not certified"
        )
    metric_matrix = tuple(
        tuple(Fraction(value) for value in row) for row in metric.matrix
    )
    size = metric.dimension
    if size % 2:
        raise PPAVValidationError("the real lattice dimension must be even")

    complex_matrix = _exact_matrix(
        complex_structure_numerator,
        name="complex_structure_numerator",
    )
    if len(complex_matrix) != size:
        raise PPAVValidationError("metric and complex structure dimensions differ")

    square = _multiply(complex_matrix, complex_matrix)
    if square != _identity(size, Fraction(-radicand)):
        raise PPAVValidationError("complex structure does not satisfy J_num^2 = -r I")

    preserved_metric = _multiply(
        _multiply(_transpose(complex_matrix), metric_matrix),
        complex_matrix,
    )
    if preserved_metric != _scale(metric_matrix, Fraction(radicand)):
        raise PPAVValidationError(
            "complex structure is not an isometry of the scaled metric"
        )

    derived_rational = _scale(
        _multiply(metric_matrix, complex_matrix),
        Fraction(1, radicand),
    )
    derived_alternating = _integral_matrix(
        derived_rational,
        name="derived Riemann form G_core J_num / r",
    )
    derived_polarization = Polarization(derived_alternating)

    if alternating is not None:
        claimed = Polarization(alternating)
        if claimed.matrix != derived_polarization.matrix:
            raise PPAVValidationError(
                "claimed polarization does not equal G_core J_num / r"
            )
        polarization = claimed
    else:
        polarization = derived_polarization

    if expected_type is not None:
        required_type = tuple(int(value) for value in expected_type)
        if polarization.type != required_type:
            raise PPAVValidationError(
                f"polarization type {polarization.type} does not match expected {required_type}"
            )

    metric_determinant = metric.determinant
    if not isinstance(metric_determinant, Fraction):
        raise ArithmeticError("exact metric unexpectedly had an inexact determinant")
    physical_determinant = metric_determinant / radicand ** (size // 2)
    if physical_determinant != abs(polarization.determinant):
        raise PPAVValidationError(
            "metric/polarization covolume identity failed: "
            "det(G_core)/r^g != |det(A)|"
        )

    return PPAVValidationResult(
        metric_core=metric_matrix,
        complex_structure_numerator=complex_matrix,
        scale_radicand=radicand,
        polarization=polarization,
        checks=(
            "positive-definite exact metric",
            "J_num^2 = -r I",
            "J_num^T G_core J_num = r G_core",
            "A = G_core J_num / r is integral alternating",
            "polarization type verified",
            "det(G_core)/r^g = |det(A)|",
        ),
    )


def validate_ppav_data(
    metric_core: Sequence[Sequence[int | Fraction]] | Metric,
    complex_structure_numerator: Sequence[Sequence[int | Fraction]],
    alternating: Sequence[Sequence[int]] | None = None,
    *,
    scale_radicand: int = 1,
) -> PPAVValidationResult:
    """Certify a *principally* polarized abelian variety (PPAV)."""

    result = validate_polarized_abelian_data(
        metric_core,
        complex_structure_numerator,
        alternating,
        scale_radicand=scale_radicand,
    )
    expected = (1,) * result.dimension
    if result.polarization_type != expected:
        raise PPAVValidationError(
            f"expected a principal polarization of type {expected}, "
            f"got {result.polarization_type}"
        )
    return result
