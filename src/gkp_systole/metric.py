"""Positive-definite metric data for lattice-coordinate computations."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import sqrt
from numbers import Integral, Rational, Real
from typing import Sequence, Union


Scalar = Union[Fraction, float]
MetricMatrix = tuple[tuple[Scalar, ...], ...]


class MetricError(ValueError):
    """Raised when a matrix is not a valid positive-definite metric."""


def _coerce_scalar(value: Real) -> Scalar:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, Rational):
        return Fraction(value.numerator, value.denominator)
    if isinstance(value, Real):
        return float(value)
    raise MetricError("metric entries must be real numbers")


def cholesky_upper(matrix: Sequence[Sequence[Scalar]]) -> tuple[tuple[float, ...], ...]:
    """Return upper triangular R with G = R^T R."""

    size = len(matrix)
    lower = [[0.0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            subtotal = sum(
                lower[row][inner] * lower[column][inner]
                for inner in range(column)
            )
            if row == column:
                diagonal = float(matrix[row][row]) - subtotal
                if diagonal <= 0.0:
                    raise MetricError("metric must be positive definite")
                lower[row][column] = sqrt(diagonal)
            else:
                lower[row][column] = (
                    float(matrix[row][column]) - subtotal
                ) / lower[column][column]
    return tuple(
        tuple(lower[column][row] for column in range(size))
        for row in range(size)
    )


@dataclass(frozen=True)
class Metric:
    """A symmetric positive-definite Gram matrix in lattice coordinates."""

    matrix: Sequence[Sequence[Real]]
    _matrix: MetricMatrix = field(init=False, repr=False)
    _cholesky_upper: tuple[tuple[float, ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rows = tuple(tuple(_coerce_scalar(value) for value in row) for row in self.matrix)
        if not rows:
            raise MetricError("metric must be nonempty")
        size = len(rows)
        if any(len(row) != size for row in rows):
            raise MetricError("metric must be square")
        if any(rows[row][column] != rows[column][row] for row in range(size) for column in range(size)):
            raise MetricError("metric must be symmetric")
        factor = cholesky_upper(rows)
        object.__setattr__(self, "matrix", rows)
        object.__setattr__(self, "_matrix", rows)
        object.__setattr__(self, "_cholesky_upper", factor)

    @property
    def dimension(self) -> int:
        return len(self._matrix)

    @property
    def is_exact(self) -> bool:
        return all(
            isinstance(value, Fraction)
            for row in self._matrix
            for value in row
        )

    @property
    def cholesky_upper(self) -> tuple[tuple[float, ...], ...]:
        return self._cholesky_upper

    @property
    def determinant(self) -> Scalar:
        """Return the determinant, exactly for rational metrics."""

        if self.is_exact:
            work = [list(map(Fraction, row)) for row in self._matrix]
            result = Fraction(1)
            sign = 1
            for column in range(self.dimension):
                pivot_row = next(
                    (
                        row
                        for row in range(column, self.dimension)
                        if work[row][column] != 0
                    ),
                    None,
                )
                if pivot_row is None:
                    return Fraction(0)
                if pivot_row != column:
                    work[column], work[pivot_row] = work[pivot_row], work[column]
                    sign *= -1
                pivot = work[column][column]
                result *= pivot
                for row in range(column + 1, self.dimension):
                    factor = work[row][column] / pivot
                    for inner in range(column + 1, self.dimension):
                        work[row][inner] -= factor * work[column][inner]
            return sign * result

        # det(G) = det(R)^2 for the cached factor G = R^T R.
        diagonal_product = 1.0
        for index in range(self.dimension):
            diagonal_product *= self._cholesky_upper[index][index]
        return diagonal_product * diagonal_product

    def quadratic(self, vector: Sequence[Fraction | int | float]) -> Scalar:
        if len(vector) != self.dimension:
            raise ValueError("vector dimension does not match metric")
        if self.is_exact and all(isinstance(value, (Fraction, Integral)) for value in vector):
            return sum(
                (
                    Fraction(vector[row])
                    * Fraction(self._matrix[row][column])
                    * Fraction(vector[column])
                    for row in range(self.dimension)
                    for column in range(self.dimension)
                ),
                Fraction(0),
            )
        return sum(
            float(vector[row])
            * float(self._matrix[row][column])
            * float(vector[column])
            for row in range(self.dimension)
            for column in range(self.dimension)
        )
